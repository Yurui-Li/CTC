for VAR in $( seq 1 5 )
do
    echo "[$(date +%Y%m%d_%H%M%S)] 当前run次数为：$VAR"
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=HeA_Defence2Group agent=dolc mac=dolc learner=dolc use_tensorboard=True save_model=True isActionReward=True isDamageReward=True name=DOLC_D2G
done
