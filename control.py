"""Script for controling the project."""
# /usr/bin/env python3
import os
from engine.misc.config import config
from engine.training import self_play, train
from engine.model import load_latest_model, path_to_cache, save_model
from engine.data import load_dataset
from engine.test import compute_model_winrate_against
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


def train_models(generations: int):
    """Trains models for a certain number of generations."""
    # Try to load the latest model.
    model = load_latest_model()
    starting_generaiton = len(os.listdir(path_to_cache))

    # Perform self play
    for gen in range(starting_generaiton, starting_generaiton + generations):
        print(
            f"Generation: currently {gen} / {generations}, and there has in total been {len(os.listdir(path_to_cache)) - 1} improved models."
        )
        self_play(model, config["game"]["size"], config["game"]["komi"])
        # TODO: Implement a window function for this.
        new_model = train(load_dataset(generations=2), model)

        # Test new_model vs old model.
        if compute_model_winrate_against(new_model, model) > 0.5:
            save_model(new_model)
            model = new_model


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

    if args.train != 0:
        train_models(args.train)
