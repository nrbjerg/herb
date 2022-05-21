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
)
from numpy.typing import ArrayLike
from engine.state import State
from engine.misc.config import config


def extract_symertries(
    inputs: ArrayLike, policy: Matrix, value: float
) -> List[Datapoint]:
    """Extract extra data, from the fact that the board has 8 symertries."""
    print(inputs.shape, policy.shape)
    # TODO:
    reflections_along_diagonals = []

    reflections_along_axises = [
        Datapoint(np.flip(inputs, axis=1), np.flip(policy, axis=0), value),
        Datapoint(np.flip(inputs, axis=2), np.flip(policy, axis=1), value),
    ]

    rotations = [
        Datapoint(np.rot90(inputs, axes=(1, 2)), np.rot90(policy, axes=(0, 1)), value)
        for k in [1, 2, 3]
    ]

    normal = [Datapoint(inputs, policy, value)]

    return rotations + reflections_along_diagonals + reflections_along_axises + normal


def get_datapoints(
    moves: List[Point], policies: List[Matrix], outcome: GameOutcome
) -> List[Datapoint]:
    """Play the moves and extract datapoints, along with their symertries."""
    datapoints = []
    state = State(config["game_parameters"]["size"], config["game_parameters"]["komi"])
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
        move_history = state.create_move_tensor()
        board = state.board
        board_mask = np.expand_dims(state.board_mask, axis=0)
        liberties = np.zeros(
            (1, config["game_parameters"]["size"], config["game_parameters"]["size"])
        )
        inputs = np.concatenate(
            [board, board_mask, liberties, legal_moves, move_history], axis=0
        )

        datapoints.extend(extract_symertries(inputs, policy, value))

        state.play_move(move)

    return datapoints


def create_dataset_from_datapoints(datapoints: List[Datapoint]) -> Dataset:
    """Combine the datapoints into a single dataset used for training."""
    pass


if __name__ == "__main__":
    print(get_datapoints([(1, 1)], [np.zeros((9, 9))], GameOutcome(0, Player.WHITE)))
