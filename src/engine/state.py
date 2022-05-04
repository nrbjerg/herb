#!/usr/bin/env python3
import numpy as np
import numba as nb
from typing import List, Tuple, Set, Generator
from numpy.typing import ArrayLike
from enum import IntEnum

Point = Tuple[int]


class GameOutcome(IntEnum):
    """ Used to model the outcome of the game (suprice!) """

    loss = -1
    draw = 0
    win = 1


class State:
    def __init__(self, dim: int):
        """Initialize a new state, with no handicaps."""
        self.dim = dim
        self.board = np.zeros((2, dim, dim), dtype=np.int64)
        self.player = 0

    @staticmethod()
    def _bfs(
        mask: ArrayLike, starting_point: Point, points: List[Point]
    ) -> List[Point]:
        """Perform breath first search on the mask and returns
           a list of all the points connected to the starting point."""
        pass

    def _is_adjecent_to(self, point: Point, player_index: int) -> bool:
        """Checks if the point is adjecent to a stone of the given player."""
        pass

    def _get_adjecent_points(self, point: Point) -> Generator[Point]:
        """Iterate over adjecent points"""
        pass

    # NOTE: The scoring is based uppon REF: https://senseis.xmp.net/?Scoring
    def score_relative_to(self, player_index: int) -> GameOutcome:
        """Compute the final score of the game."""
        pass

    def score(self, player_index) -> int:
        """Compute the score of the given player"""
        pass
