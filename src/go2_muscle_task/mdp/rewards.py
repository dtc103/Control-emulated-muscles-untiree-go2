from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def track_joint_pos(
    env: ManagerBasedRLEnv, std: float, command_name: str, joint_names="[.*]", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    current_joint_angles = asset.data.joint_pos[:, asset.find_joints(joint_names)[0]]
    goal_joint_angles = env.command_manager.get_command(command_name)
    joint_ang_error = torch.sum(
        torch.square(current_joint_angles - goal_joint_angles), dim=-1
    )
    #print("squared: ", joint_ang_error)
    return torch.exp(-joint_ang_error/std**2)

def body_height(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    #print(asset.find_bodies("base")[0])
    print(asset.data.joint_names)
    height = asset.data.root_pos_w[:, 2]
    #print(height)
    height_norm = torch.mean(height)
    #print(height_norm)
    
    #print(height_norm)

    return torch.exp(height_norm)

def hop(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b[:, 2]
    return torch.sum(torch.clip(torch.exp(torch.clamp(lin_vel, min=0)), min = 0.0, max=10.0))
