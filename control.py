"""Script for controling the project."""
# /usr/bin/env python3
import os
from engine.misc.config import config
import argparse
import shutil


def remove_data(*arg, **kwarg):
    """Remove the contents of the models directory."""
    path = os.path.join(os.getcwd(), config["data"]["path_to_database"])
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)


def remove_models(*arg, **kwarg):
    """Remove the contents of the models directory."""
    path = os.path.join(os.getcwd(), config["model"]["path_to_model_cache"])
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A basic script for controlling the processes"
    )

    parser.add_argument("--remove", type=str, default="", help="what to remove")

    parser.add_argument(
        "--train",
        type=int,
        default=0,
        help="wether or not to begin training, and how many genrations to train.",
    )
    args = parser.parse_args()

    if "m" in args.remove:
        remove_models()
    if "d" in args.remove:
        remove_data()
