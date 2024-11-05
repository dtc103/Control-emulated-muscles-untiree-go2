import gymnasium as gym
from .flat_terrain_muscle import Go2VelocityMuscleTaskCfg
from .config import agents

gym.register(
    id="Muscle-Walk-Task", 
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2VelocityMuscleTaskCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2MuscleRunnerCfg"
    }
)