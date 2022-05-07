#!/usr/bin/env python3
import numpy as np
import numba as nb
from typing import List, Tuple, Set, Iterator
from numpy.typing import ArrayLike
from enum import IntEnum
from itertools import product


Point = Tuple[int]


class GameOutcome(IntEnum):
    """ Used to model the outcome of the game (suprice!) """

    loss = -1
    draw = 0
    win = 1


class State:
    @property
    def points(self) -> List[Point]:
        """Return a list of all of the points, on the board."""
        return product(range(self.dim), range(self.dim))

    @property
    def board_mask(self) -> ArrayLike:
        """Return a mask of the board, with 1 on the intersections with stones and 0 on the free intersections."""
        return np.sum(self.board, axis=0)

    def __init__(self, dim: int):
        """Initialize a new state, with no handicaps."""
        self.dim = dim
        self.board = np.zeros((2, dim, dim), dtype=np.int64)
        self.player = 0
        self.number_of_passes = 0

    def _check_move(self, point: Point) -> None:
        """Check wether a move is valid, note that this might throw an expection"""
        if point[0] < 0 or point[0] >= self.dim or point[1] < 0 or point[1] >= self.dim:
            raise ValueError(
                f"invalid move: the point {point} is not in the range {self.dim} x {self.dim}"
            )
        elif self.board_mask[point] != 0:
            raise ValueError(f"Can't make move on {point}, a stone is already there")

        # TODO: HANDLE SURCIDE MOVES

    def play_move(self, point: Point = None):
        """Play a move on the board, note that a point of none is used to inidicate a pass."""
        if point != None:
            self._check_move(point)
            self.board[self.player][point] = 1
            # Check if the move kills stones TODO:

        else:
            self.number_of_passes += 1

        self.player = (self.player + 1) % 2

    def _check_for_surcide(self, point: Point, opponent_index: int) -> bool:
        """Return True if there is an adjecent point where the mask is 0."""
        for adjecent in self._get_adjecent_points(point):
            if self.board[opponent_index][adjecent[0]][adjecent[1]] == 0:
                return False

        return True

    def get_avalible_moves(self, player_index: int) -> ArrayLike:
        """Return a numpy array of avalible moves."""
        opponent = (player_index + 1) % 2
        moves = (self.board_mask + 1) % 2  # NOTE: This just flips 0 to 1 and 1 to 0

        for point in self.points:
            # Check that it's not a surcide move
            if moves[point] == 1 and self._check_for_surcide(point, opponent):
                moves[point] = 0

        return moves

    def _flod_fill(
        self, mask: ArrayLike, starting_point: Point, points: List[Point] = None
    ) -> List[Point]:
        """Perform breath first search on the mask and returns
           a list of all the points connected to the starting point."""
        if points == None:
            points = []

        # Only perform recursion if there is nothing at the starting point
        if mask[starting_point] == 0:
            points.append(starting_point)
        else:
            return points

        for point in self._get_adjecent_points(starting_point):
            if point in points:  # Avoid infinite loops
                continue
            else:
                points = self._flod_fill(mask, point, points=points)

        return points

    def _is_adjecent_to(self, point: Point, player_index: int) -> bool:
        """Check if the point is adjecent to a stone of the given player."""
        # Loop over each edjecent point, and checks if there is a stone there.
        for adjecent in self._get_adjecent_points(point):
            if self.board[player_index][adjecent[0]][adjecent[1]] == 1:
                return True

        return False

    def _get_adjecent_points(self, point: Point) -> Iterator[Point]:
        """Iterate over adjecent points."""
        if point[0] > 0:
            yield (point[0] - 1, point[1])

        if point[1] > 0:
            yield (point[0], point[1] - 1)

        if point[0] < self.dim - 1:
            yield (point[0] + 1, point[1])

        if point[1] < self.dim - 1:
            yield (point[0], point[1] + 1)

    # REF: https://senseis.xmp.net/?Scoring
    # TODO: implement half move scoring
    def score_relative_to(self, player_index: int) -> int:
        """Compute the final score of the game."""
        empty_points_strings = []
        for point in self.points:
            # Skip the point, if it has already been visited or there is a stone
            if point in sum(empty_points_strings, []) or self.board_mask[point] == 1:
                continue

            empty_points_strings.append(self._flod_fill(self.board_mask, point))

        # Compute the scores of each player
        player_score = self.score(player_index, empty_points_strings)
        oponent_score = self.score((player_index + 1) % 2, empty_points_strings)

        return player_score - oponent_score

    def score(self, player_index: int, empty_points_strings: List[List[Point]]) -> int:
        """Compute the score of the given player, using the chinese scoring method."""
        stones = {
            point
            for point in self.points
            if self.board[player_index][point[0]][point[1]] == 1
        }

        points = len(stones)
        for string in empty_points_strings:
            # NOTE: Only add adjecent points if they aren't in the string
            border = set()
            for point in string:
                for adjecent in self._get_adjecent_points(point):
                    if (adjecent in string) == False:
                        border.add(adjecent)

            if border.issubset(stones):
                points += len(string)

        return points


def from_SGF(sgf: str) -> State:
    """Produce a State, by playing the moves in the sgf format."""
    pass
