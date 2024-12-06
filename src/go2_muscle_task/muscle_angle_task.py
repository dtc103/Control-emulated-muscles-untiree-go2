# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp import track_joint_pos, body_height
from .mdp.commands.commands_cfg import JointAngleCommandCfg
from .actuators import ForwardEffortActuatorCfg

from go2_muscle_task.asset.unitree import UNITREE_GO2_BODY_FIX_CFG



##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    #command []
    pose_command = JointAngleCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=JointAngleCommandCfg.Ranges(joint_angles=[(-1.0471, 1.0471), (-1.0471, 1.0471), (-1.0471, 1.0471), (-1.0471, 1.0471), #set hip to 0
                (-1.5708, 3.4907), (-1.5708, 3.4907), (-0.5236, 4.5378), (-0.5236, 4.5378), #thigh front, thigh back
                (-math.pi, -0.8), (-math.pi, -0.8), (-math.pi, -0.8), (-math.pi, -0.8)]) #set calf to random (calf is always between 0 and math.pi)
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    #joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    joint_pos = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*(hip|thigh|calf)"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


    # -- penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-2)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- optional penalties
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.2)

    # body_height = RewTerm(
    #     func=mdp.base_height_l2, 
    #     params={"target_height": 0.4},
    #     weight=1.5
    # )
    #flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)

    # --task--
    joint_pos = RewTerm(
        func=track_joint_pos,
        params={"command_name": "pose_command", "std": math.sqrt(10), "joint_names": ".*_(thigh|calf|hip)_joint"},
        weight=2.0
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.1},
    # )

##
# Environment configuration
##


@configclass
class MuscleAngleCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        self.scene.robot = UNITREE_GO2_BODY_FIX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators = {
            "base_legs": ForwardEffortActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=45.0,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
        }
        self.scene.robot.spawn.articulation_props.fix_root_link = True


@configclass
class MuscleAngleCfg_PLAY(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.observations.policy.enable_corruption = False
        
        #self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.robot = UNITREE_GO2_BODY_FIX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators = {
            "base_legs": ForwardEffortActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=45.0,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
        }
        self.scene.robot.spawn.articulation_props.fix_root_link = True

