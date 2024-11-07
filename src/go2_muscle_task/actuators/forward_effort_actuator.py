from omni.isaac.lab.actuators import ActuatorBase

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ActuatorBaseCfg
from typing import TYPE_CHECKING




class FowardEffortActuator(ActuatorBase):
    cfg: "ForwardEffortActuatorCfg"
    
    def reset(self, env_ids):
        pass

    
    # just forward the effort, which we will calculate from the action later
    def compute(self, control_action, joint_pos, joint_vel):
        control_action.joint_positions = None
        control_action.joint_velocities = None

        self.computed_effort = control_action.joint_efforts
        self.applied_effort = self._clip_effort(self.computed_effort)
        
        control_action.joint_efforts = self.applied_effort
        return control_action

@configclass
class ForwardEffortActuatorCfg(ActuatorBaseCfg):
    class_type = FowardEffortActuator