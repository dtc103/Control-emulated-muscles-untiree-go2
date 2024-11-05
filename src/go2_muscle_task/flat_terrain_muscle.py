from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG
from omni.isaac.lab.utils import configclass

from .mdp.actions.actions_cfg import MuscleJointActionCfg

@configclass
class Go2VelocityMuscleTaskCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
    
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        self.curriculum.terrain_levels = None

        self.actions.joint_pos = MuscleJointActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)

@configclass
class Go2VelocityMuscleTaskCfg_PLAY(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None



    
        

        

        