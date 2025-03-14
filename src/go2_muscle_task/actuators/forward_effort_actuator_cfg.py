from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ActuatorBaseCfg

from . import forward_effort_actuator
@configclass
class ForwardEffortActuatorCfg(ActuatorBaseCfg):
    class_type = forward_effort_actuator.FowardEffortActuator