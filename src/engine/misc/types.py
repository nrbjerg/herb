"""The types used by the engine."""
# /usr/bin/env python3
from typing import Tuple, List, Dict, Any
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

    @staticmethod
    def Oppoent(player: int):
        """Get the opponent of the player."""
        return Player.BLACK if player == Player.WHITE else Player.WHITE


@dataclass()
class Datapoint:
    """A single datapoint for training."""

    inputs: ArrayLike
    policy: Matrix
    value: float


@dataclass()
class Dataset:
    """The dataset for training."""

    inputs: ArrayLike
    values: ArrayLike
    policies: ArrayLike


@dataclass()
class GameOutcome:
    """Models a game outcome."""

    point_difference: float
    winner: Player


@dataclass()
class Game:
    """Models an actual game."""

    # Game setup
    pre_moves: List[Point]

    # General game information
    moves: List[Point]
    outcome: GameOutcome

    # For training.
    policies: List[ArrayLike]

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version of the game, ready to be saved."""
        return {
            "pre_moves": self.pre_moves,
            "moves": self.moves,
            "winner": self.outcome.winner,
            "point_difference": self.outcome.point_difference,
            "policies": self.policies,
        }
