import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")

# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)

# parse the arguments

args_cli = parser.parse_args()


# launch omniverse app

app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

import torch
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from go2_muscle_task.actuators import MuscleModel, ForwardEffortActuatorCfg
from go2_muscle_task.asset.unitree import UNITREE_GO2_BODY_FIX_CFG
from omni.isaac.lab.actuators import DCMotorCfg
import pandas as pd


import numpy as np

muscle_params = {
    "lmin":0.2,
    "lmax":2.0,
    "fvmax": 1.8,
    "fpmax": 2,
    "lce_min": 0.75,
    "lce_max": 0.93,
    "phi_min": -2.7227,
    "phi_max": -0.8,
    "vmax": 30.0, # TODO pierre fragen, ob das das ricthige ist (taken from unitre.py UNITREE_GO2_CFG)
    "peak_force": 20.0,
    "eps": 10e-6 # eps is just a smal number for numerical purpouses
}

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    origins = [[0.0, 0.0, 1.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    robot_cfg = UNITREE_GO2_BODY_FIX_CFG.replace(prim_path="/World/Origin1/Robot")
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.init_state.pos = (0.0, 0.0, 0.1)
    robot_cfg.actuators = {
        "calf": ForwardEffortActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "remaining": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    }
    
    robot = Articulation(cfg=robot_cfg)

    scene_entities = {"robot": robot}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    robot = entities["robot"]
    sim_dt = sim.get_physics_dt()
    count = 1

    #fvmaxs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    #i = 0

    muscle_params["fvmax"] = 0.9
    muscle = MuscleModel(muscle_params, torch.zeros(1, 24, device="cuda:0"), 1, None)

    pos_log = []

    # Simulation loop
    while simulation_app.is_running():
        if count % 500 == 0:
            pd.DataFrame(pos_log).to_csv(f"free_running.csv", sep=";", header=None, index=False)
            break
            pos_log = []
            count = 1
            # i += 1
            # muscle_params["fvmax"] = fvmaxs[i]
            # muscle = MuscleModel(muscle_params, torch.zeros(1, 24, device="cuda:0"), 1, None)

            # robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())

        positions = torch.zeros(1, 12, device="cuda:0")
        # keep the hip at 0 
        positions[0, 0:4] = 0
        #put the thighs a little bit back, so the calfs have room to move
        positions[0, 4:8] = 1.0
        #positions[0, 8:] = -1p.pi + count * 0.001 
        robot.set_joint_position_target(positions)
        #robot.write_joint_state_to_sim(positions, torch.zeros(1, 12, device="cuda:0"))

        # joint position indexing
        #[FL_HIP, FR_HIP, RL_HIP, RR_HIP, FL_THIGH, FR_THIGH, RL_THIGH, RR_THIGH, FL_CALF, FR_CALF, RL_CALF, RR_THIGH]
        activation = torch.zeros(1, 24, device="cuda:0") #only for calfs
        activation[0, -1] = 0.20 #min((count) * 0.0001 + 0.15, 1.0) #np.sin(count * 0.001) * 0.5 + 0.5 #acts[pos] #
        activation[0, 11] = 0.1
        effort = muscle.compute_torques(robot.data.joint_pos, robot.data.joint_vel, activation)
        effort[0, 0:4] = 0
        effort[0, 4:8] = 0

        
        print("activation", activation)
        
        robot.set_joint_effort_target(effort)

        robot.write_data_to_sim()
        print("jointpos: ", robot.data.joint_pos)
        print("applied torque", robot.data.applied_torque)
        
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)

        pos_log.append(robot.data.joint_pos[0, -1].cpu().numpy())
        #activation_log.append(activation[0, -1].cpu().numpy())


    
    
    #pd.DataFrame(activation_log).to_csv(f"muscle_activation_activation_experiment.csv", sep=";", header=None, index=False)

    print(len(pos_log))


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.0, -2.0, 2.0], [0.0, 0.0, 0.75])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
