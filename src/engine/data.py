"""This script contains functions to turn a list of moves into usable data for training."""
# /usr/bin/env python3
import numpy as np
from typing import List
from engine.misc.types import (
    Matrix,
    Dataset,
    Datapoint,
    GameOutcome,
    Point,
    Player,
    Game,
    Pass,
)
from engine.state import State
from engine.symmertrices import extract_symertries
from engine.misc.config import cfg
import json
import os

base_path_to_data = os.path.join(os.getcwd(), cfg.data.path_to_database)


def get_datapoints(
    pre_moves: List[Point],
    moves: List[Point],
    policies: List[Matrix],
    outcome: GameOutcome,
) -> List[Datapoint]:
    """Play the moves and extract datapoints, along with their symertries."""
    datapoints = []

    # Intialize state
    state = State(cfg.game.size, cfg.game.komi)
    for move in pre_moves:
        state.play_move(move)

    for policy, move in zip(policies, moves):
        # Is this how good the current state is for the player or other whise
        # (i think its how good it is for the opponent)
        value = (
            outcome.point_difference
            if outcome.winner == state.current_opponent
            else -outcome.point_difference
        )
        # Get model inputs
        inputs = state.convert_to_input_tensor()

        # TODO: the move value should drop of over time?
        datapoints.extend(extract_symertries(inputs, policy, value))

        state.play_move(move)

    return datapoints


def load_data_from_file(path_to_file: str) -> List[Datapoint]:
    """Load the data from the file, and convert it to individual datapoints."""
    size = cfg.game.size
    with open(path_to_file, "r") as file:
        json_object = json.load(file)
        pre_moves = map(lambda m: tuple(m), json_object["pre_moves"])
        # NOTE: move is represented as a tuple, but serialized as a list.
        moves = [
            tuple(move) if move != [size, size] else Pass
            for move in json_object["moves"]
        ]
        policies = [np.array(pol) for pol in json_object["policies"]]
        winner = [
            Player.WHITE if winner == "white" else Player.BLACK
            for winner in json_object["winner"]
        ]
        difference = json_object["point_difference"]

        outcome = GameOutcome(
            difference, winner=Player.WHITE if winner == "white" else Player.BLACK
        )

        return get_datapoints(pre_moves, moves, policies, outcome)


def load_dataset(generations: int) -> Dataset:
    """Load the dataset, containing the datapoints from the latest generations, from the data directory."""
    directories = sorted(os.listdir(base_path_to_data))

    # Load datapoints from directories
    datapoints = []
    for idx, dir in enumerate(reversed(directories)):
        path_to_dir = os.path.join(base_path_to_data, dir)
        for file in os.listdir(path_to_dir):
            path_to_file = os.path.join(path_to_dir, file)
            datapoints.extend(load_data_from_file(path_to_file))

        if idx >= generations:
            break

    # Convert datapoints to datasets
    return create_dataset_from_datapoints(datapoints)


def create_dataset_from_datapoints(datapoints: List[Datapoint]) -> Dataset:
    """Combine the datapoints into a single dataset used for training."""
    # NOTE: stack arrays accrodingly.
    inputs = np.stack([datapoint.inputs for datapoint in datapoints])
    values = np.stack([datapoint.value for datapoint in datapoints])
    policies = np.stack([datapoint.policy for datapoint in datapoints])

    return Dataset(inputs, values, policies)


def save_games(games: List[Game]):
    """Save the games to the data directory."""
    number_of_folders = len(os.listdir(base_path_to_data))

    folder_dir = os.path.join(base_path_to_data, str(number_of_folders))
    os.mkdir(folder_dir)
    # Write the games to the appropriete folder
    for idx, game in enumerate(games):
        with open(os.path.join(folder_dir, f"{idx}.json"), "w+") as file:
            json_str = json.dumps(game.to_dict())
            file.write(json_str)


if __name__ == "__main__":
    dataset = load_dataset(2)
