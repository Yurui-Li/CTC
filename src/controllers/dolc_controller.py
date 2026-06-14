from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class DOLCMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(DOLCMAC, self).__init__(scheme, groups, args)
        self.hidden_states = []

    def init_hidden(self, batch_size):
        # self.hidden_states = self.agent.init_hidden()
        # if self.hidden_states is not None:
        self.hidden_states = [self.agent.init_hidden(i).unsqueeze(0).expand(batch_size, self.n_agents, -1) 
        for i in range(self.args.n_experts)]
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals,_ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        # avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states,g = self.agent(agent_inputs, self.hidden_states,t)

        return agent_outs,g