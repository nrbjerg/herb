"""The types used by the engine."""
# /usr/bin/env python3
from typing import Tuple, List
from numpy.typing import ArrayLike
from enum import IntEnum
from dataclasses import dataclass

Point = Tuple[int, int]
Matrix = ArrayLike
Board = ArrayLike


class Player(IntEnum):
    """Players & their indicies."""

    BLACK = 0
    WHITE = 1


@dataclass()
class Datapoint:
    """A single datapoint for training"""

    inputs: ArrayLike
    policy: Matrix
    value: float


@dataclass()
class Dataset:
    """The dataset for training"""

    inputs: ArrayLike
    values: ArrayLike
    policies: ArrayLike

    def trim_dataset(window: int):
        """Remove old enteries from the dataset."""
        pass


@dataclass()
class GameOutcome:
    """Models a game outcome."""

    point_difference: float
    winner: Player


@dataclass()
class Game:
    """Models an actual game."""

    # Game setup
    size: int
    komi: float
    pre_game_moves: List[Point]

    # General game information
    moves: List[Point]
    outcome: GameOutcome

    # For training.
    policy_activations: List[ArrayLike]
