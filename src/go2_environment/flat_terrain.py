from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG



class Go2Task(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
    
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
        

        

        