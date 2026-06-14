from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
from collections import Counter


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0
        

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

        # reward shaping
        self.isDamageReward = args.isDamageReward
        if self.isDamageReward:
            self.rDamageCoff = DecayThenFlatSchedule(args.beta_start,args.beta_finish,args.beta_anneal_time)
        self.isActionReward = args.isActionReward
        if self.isActionReward:
            self.rActionCoff = args.actionRewardCoff

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        # for smacv2
        for parent_conn in self.parent_conns:
            parent_conn.send(("save_replay", None))
        # pass # for grf

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        save_probs = getattr(self.args, "save_probs", False)
        ac = 0.0
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                
            if self.isActionReward:
                action_rewards = (actions > 5).float().mean(dim=1) # [batch_size]
            cpu_actions = actions.to("cpu").numpy()

            _count = Counter(cpu_actions.tolist()[0]).most_common(1)[0][1]
            _count = 0 if _count==1 else _count
            ac += _count / actions.shape[1]
            

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            
            if self.isDamageReward:
                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_enemies",None))
                _enemies = []
                for parent_conn in self.parent_conns:
                    _stat = parent_conn.recv()
                    _enemies.append(_stat)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            if self.isActionReward:
                action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    
                    #reward shaping
                    if self.isActionReward:
                        data["reward"] += action_rewards[action_idx].item() * self.rActionCoff
                        action_idx += 1
                    if self.isDamageReward:
                        cur_health = [ene.health for ene in _enemies[idx].values() if int(ene.unit_type)==54]
                        r_ext = (1 - np.array(cur_health) / 150).mean()
                        data["reward"] += r_ext * self.rDamageCoff.eval(self.t_env)
                    
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        _count = Counter(cpu_actions.tolist()[0]).most_common(1)[0][1]
        _count = 0 if _count==1 else _count
        ac += _count / actions.shape[1]
        # ac += (Counter(cpu_actions.tolist()[0]).most_common(1)[0][1] - 1) / (cpu_actions.shape[1] - 1)
        cur_stats["ac_sum"] = ac + cur_stats.get("ac_sum", 0.0)
        cur_stats["ac_count"] = self.t + 1 + cur_stats.get("ac_count", 0.0)
        # cur_stats["action_similar"] = ac / (self.t + 1) + cur_stats.get("action_similar", 0.0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k == "ac_sum":
                self.logger.log_stat(prefix + "action_similar_mean" , stats["ac_sum"]/stats["ac_count"], self.t_env)
            elif k != "n_episodes" and k!="ac_count":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "save_replay":
            remote.send(env.save_replay())
        elif cmd == "get_agents":
            remote.send(env.get_agents())
        elif cmd == "get_enemies":
            remote.send(env.get_enemies())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
            