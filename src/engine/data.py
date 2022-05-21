"""This script contains functions to turn a list of moves into usable data for training."""
# /usr/bin/env python3
import numpy as np
from typing import List, Tuple
from engine.misc.types import (
    Board,
    Matrix,
    Dataset,
    Datapoint,
    GameOutcome,
    Point,
    Player,
    Game,
)
from numpy.typing import ArrayLike
from engine.state import State
from engine.misc.config import config
import json
import os

base_path_to_data = os.path.join(os.getcwd(), config["data"]["path_to_database"])


def extract_symertries(
    inputs: ArrayLike, policy: Matrix, value: float
) -> List[Datapoint]:
    """Extract extra data, from the fact that the board has 8 symertries."""
    # along main diagonal: flip horizontally, flip vertically, rot90 flip
    reflections = [
        # Along the middle
        Datapoint(np.flip(inputs, axis=1), np.flip(policy, axis=0), value),
        Datapoint(np.flip(inputs, axis=2), np.flip(policy, axis=1), value),
        # TODO: Along the diagonals
        # Datapoint(
        #     np.flip(np.flip(inputs, axis=1), axis=2),
        #     np.flip(np.flip(policy, axis=0), axis=1),
        #     value,
        # ),
        # Datapoint(
        #     np.flip(np.flip(inputs, axis=1), axis=2),
        #     np.flip(np.flip(policy, axis=0), axis=1),
        #     value,
        # ),
    ]

    rotations = [
        Datapoint(
            np.rot90(inputs, k, axes=(1, 2)), np.rot90(policy, k, axes=(0, 1)), value
        )
        for k in [1, 2, 3]
    ]

    normal = [Datapoint(inputs, policy, value)]

    return rotations + reflections + normal


def get_datapoints(
    pre_moves: List[Point],
    moves: List[Point],
    policies: List[Matrix],
    outcome: GameOutcome,
) -> List[Datapoint]:
    """Play the moves and extract datapoints, along with their symertries."""
    datapoints = []

    # Intialize state
    state = State(config["game"]["size"], config["game"]["komi"])
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
        legal_moves = np.expand_dims(
            state.get_avalible_moves(state.current_player), axis=0
        )
        move_history = state.get_move_tensor()
        board = state.board
        board_mask = np.expand_dims(state.board_mask, axis=0)
        liberties = np.zeros((1, config["game"]["size"], config["game"]["size"]))
        inputs = np.concatenate(
            [board, board_mask, liberties, legal_moves, move_history], axis=0
        )

        datapoints.extend(extract_symertries(inputs, policy, value))

        state.play_move(move)

    return datapoints


def load_data_from_file(path_to_file: str) -> List[Datapoint]:
    """Load the data from the file, and convert it to individual datapoints."""
    with open(path_to_file, "r") as file:
        json_object = json.load(file)
        pre_moves = json_object["pre_moves"]
        moves = json_object["moves"]
        policies = json_object["policies"]
        winner = json_object["winner"]
        difference = json_object["point_difference"]

        outcome = GameOutcome(
            difference, winner=Player.WHITE if winner == "white" else Player.BLACK
        )

        return get_datapoints(moves, policies, outcome)


def load_dataset(window: int) -> Dataset:
    """Load the dataset, containing the latest datapoints, from the data directory."""
    directories = sorted(os.listdir(base_path_to_data))

    # Load datapoints from directories
    datapoints = []
    for idx, dir in enumerate(reversed(directories)):
        path_to_dir = os.path.join(base_path_to_data, dir)
        for file in os.listdir(path_to_dir):
            path_to_file = os.path.join(path_to_dir, file)
            datapoints.extend(load_data_from_file(path_to_file))

        if idx >= window:
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

    folder_dir = os.path.join(base_path_to_data, number_of_folders)
    os.mkdir(folder_dir)
    # Write the games to the appropriete folder
    for idx, game in enumerate(games):
        with open(os.path.join(number_of_folders, f"{idx}.json"), "w+") as file:
            json.dump(game.to_dict(), file)


if __name__ == "__main__":
    dataset = load_dataset(2)
