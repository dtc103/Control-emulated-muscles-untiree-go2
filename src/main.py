import argparse
import sys

from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Training environment for training a muscle leg model on the unitree go2")
parser.add_argument("--video", action="store_true", default=False, help="record videos during training")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
    