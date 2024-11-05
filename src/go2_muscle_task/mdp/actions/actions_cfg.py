from dataclasses import MISSING
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from . import joint_actions


@configclass
class MuscleJointActionCfg(ActionTermCfg):
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False
    
    class_type = joint_actions.MuscleJointAction
