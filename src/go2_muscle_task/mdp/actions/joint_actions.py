from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.envs.mdp.actions import JointAction
from go2_muscle_task.actuators.actuator_muscle import MuscleModel

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    from . import actions_cfg

muscle_params_calf = {
    "lmin":0.2,
    "lmax":2.0,
    "fvmax": 0.8,
    "fpmax": 2,
    "lce_min": 0.75,
    "lce_max": 0.92,
    "phi_min": -2.7227,
    "phi_max": -0.8,
    "vmax": 30.0, # TODO pierre fragen, ob das das ricthige ist (taken from unitre.py UNITREE_GO2_CFG)
    "peak_force": 20.0,
    "eps": 10e-6 # eps is just a smal number for numerical purpouses
}

muscle_params_thigh_front = {
    "lmin":0.2,
    "lmax":2.0,
    "fvmax": 0.8,
    "fpmax": 2,
    "lce_min": 0.75,
    "lce_max": 0.92,
    "phi_min": -1.5708,
    "phi_max": 3.4907,
    "vmax": 30.0, # TODO pierre fragen, ob das das ricthige ist (taken from unitre.py UNITREE_GO2_CFG)
    "peak_force": 20.0,
    "eps": 10e-6 # eps is just a smal number for numerical purpouses
}

muscle_params_thigh_back = {
    "lmin":0.2,
    "lmax":2.0,
    "fvmax": 0.8,
    "fpmax": 2,
    "lce_min": 0.75,
    "lce_max": 0.92,
    "phi_min": -0.5236,
    "phi_max": 4.5378,
    "vmax": 30.0, # TODO pierre fragen, ob das das ricthige ist (taken from unitre.py UNITREE_GO2_CFG)
    "peak_force": 20.0,
    "eps": 10e-6 # eps is just a smal number for numerical purpouses
}

muscle_params_hip = {
    "lmin":0.2,
    "lmax":2.0,
    "fvmax": 0.8,
    "fpmax": 2,
    "lce_min": 0.75,
    "lce_max": 0.92,
    "phi_min": -1.0471,
    "phi_max": 1.0471,
    "vmax": 30.0, # TODO pierre fragen, ob das das ricthige ist (taken from unitre.py UNITREE_GO2_CFG)
    "peak_force": 20.0,
    "eps": 10e-6 # eps is just a smal number for numerical purpouses
}

class MuscleJointAction(JointAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        print(self._processed_actions.shape)
        print(self._asset.data.joint_vel.shape)
        print(self._asset.data.joint_pos.shape)

        self.muscles_hip = MuscleModel(muscle_params=muscle_params_hip,
                                    action_tensor=self._processed_actions[:, 0:8], 
                                    nenvironments=self.num_envs, 
                                    options=None)
        self.muscles_thigh_front = MuscleModel(muscle_params=muscle_params_thigh_front,
                                    action_tensor=self._processed_actions[:, 8:12], 
                                    nenvironments=self.num_envs, 
                                    options=None)
        self.muscles_thigh_back = MuscleModel(muscle_params=muscle_params_thigh_back,
                                    action_tensor=self._processed_actions[:, 12:16], 
                                    nenvironments=self.num_envs, 
                                    options=None)
        
        self.muscles_calf = MuscleModel(muscle_params=muscle_params_calf,
                                    action_tensor=self._processed_actions[:, 16:], 
                                    nenvironments=self.num_envs, 
                                    options=None)

    def apply_actions(self):
        toque_hip = self.muscles_hip.compute_torques(self._asset.data.joint_pos[:, 0:4], self._asset.data.joint_vel[:, 0:4], self._processed_actions[:, 0:8])
        torques_th_front = self.muscles_thigh_front.compute_torques(self._asset.data.joint_pos[:, 4:6], self._asset.data.joint_vel[:, 4:6], self._processed_actions[:, 8:12])
        torques_th_back = self.muscles_thigh_back.compute_torques(self._asset.data.joint_pos[:, 6:8], self._asset.data.joint_vel[:, 6:8], self._processed_actions[:, 12:16])
        torques_calf = self.muscles_calf.compute_torques(self._asset.data.joint_pos[:, 8:], self._asset.data.joint_vel[:, 8:], self._processed_actions[:, 16:])

        torques = torch.cat((toque_hip, torques_th_front, torques_th_back, torques_calf), dim=1)

        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)

    @property
    def action_dim(self):
        return 2 * self._num_joints # 2 times joint, so we can apply the actions
    
    @property
    def raw_actions(self):
        return self._raw_actions
    


