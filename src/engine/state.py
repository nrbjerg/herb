#!/usr/bin/env python3
import numpy as np
from typing import List, Tuple, Iterator
from numpy.typing import ArrayLike
from enum import IntEnum
from itertools import product
from engine.misc.types import Point, Board, Matrix
from engine.misc.config import config
import numba as nb


@nb.njit()
def _get_adjecent_points(point: Point, size: int) -> Iterator[Point]:
    """Iterate over adjecent points."""
    if point[0] > 0:
        yield (point[0] - 1, point[1])

    if point[1] > 0:
        yield (point[0], point[1] - 1)

    if point[0] < size - 1:
        yield (point[0] + 1, point[1])

    if point[1] < size - 1:
        yield (point[0], point[1] + 1)


# TODO: Maybe create a liberties mask & update that, in stead of computing liberties on the fly?
class State:
    """Store information about a game state"""

    @property
    def points(self) -> List[Point]:
        """Return a list of all of the points, on the board."""
        return product(range(self.size), range(self.size))

    @property
    def board_mask(self) -> Board:
        """
            Return a mask of the board, with 1 on the intersections with
            stones and 0 on the free intersections.
        """
        return np.sum(self.board, axis=0)

    @property
    def current_opponent(self) -> int:
        """Get the index of the opponent of the current player."""
        return (self.current_player + 1) % 2

    @staticmethod
    def invert_bitmap(bitmap: Matrix) -> Matrix:
        """Return an inverted bitmap, ie. 1 gets swaped to 0, and 0 to 1."""
        return (bitmap + 1) % 2

    def __init__(self, size: int, komi: float = 0):
        """Initialize a new state, with no handicaps."""
        self.size = size
        self.board = np.zeros((2, size, size), dtype=np.int64)
        self.komi = komi
        self.current_player = 0
        self.number_of_passes = 0
        self.moves = []

    # NOTE: This is only used when the human player has to make a move
    def check_move(self, point: Point) -> None:
        """Check wether a move is valid, note that this might throw an expection."""
        if (
            point[0] < 0
            or point[0] >= self.size
            or point[1] < 0
            or point[1] >= self.size
        ):
            raise ValueError(
                f"invalid move: the point {point} is not in the range {self.size} x {self.size}"
            )
        elif self.board_mask[point] != 0:
            raise ValueError(f"Can't make move on {point}, a stone is already there")

        elif self._check_for_surcide(
            point, (self.current_player + 1) % 2
        ) and not self._captures(point, self.current_opponent):
            raise ValueError(f"Move: {point} is a surcide move!")

        elif self._ko(point, self.current_opponent):
            raise ValueError(f"Move: {point} dosn't respect the ko rule.")

    def remove_string(self, string: List[Point], string_player_index: int):
        """Remove the string from the mask and return the mask."""
        for point in string:
            self.board[string_player_index][point] = 0

    def capture_stones_after_move(self, point: Point):
        """Remove dead stones from the board after playing at a point."""
        for adjecent in _get_adjecent_points(point, self.size):
            # Check if the strings have 0 liberties and remove them if so
            if self.board[self.current_opponent][adjecent] == 1:
                string = self._flod_fill(
                    self.invert_bitmap(self.board[self.current_opponent]), adjecent
                )
                if self.count_liberties(string, self.current_player) == 0:
                    self.remove_string(string, self.current_opponent)

    def play_move(self, point: Point = None):
        """Play a move on the board, note that a point of none is used to inidicate a pass."""
        if point is not None:
            self.board[self.current_player][point] = 1
            self.capture_stones_after_move(point)

        else:
            self.number_of_passes += 1

        # Update variables
        self.moves.append(point)
        print(self.moves)
        self.current_player = self.current_opponent

    def _check_for_surcide(self, point: Point, opponent_index: int) -> bool:
        """Return True if there is an adjecent point where the mask is 0."""
        for adjecent in _get_adjecent_points(point, self.size):
            if self.board[opponent_index][adjecent[0]][adjecent[1]] == 0:
                return False

        return True

    # NOTE: Here oponent referes to the opponent of the string.
    def count_liberties(self, string: List[Point], opponent_index: int) -> int:
        """Count the number of liberites of a string of stones."""
        n = 0
        # Loop over each point in the string & count it's "liberties".
        for point in string:
            for adjecent in _get_adjecent_points(point, self.size):
                # If the adjecent point is in string, dont count it.
                if (adjecent in string) is False and self.board[opponent_index][
                    adjecent
                ] == 0:
                    n += 1

        return n

    # TODO: Keep track of the strings during computations (this sould make it alot more effective.)
    def _captures(self, point: Point, opponent_index: int) -> bool:
        """Return true if playing uppon that point causes a capture."""
        for adjecent in _get_adjecent_points(point, self.size):
            # Count the number of liberties of the adjecent points, if it's 1 then its 0 after the move.
            if self.board[opponent_index][adjecent] == 1:
                string = self._flod_fill(
                    self.invert_bitmap(self.board[opponent_index]), adjecent
                )
                if self.count_liberties(string, (opponent_index + 1) % 2) == 1:
                    return True

        return False

    # NOTE: statement: One may not capture just one stone if that stone was played on the previous move and that move also captured just one stone.
    def _ko(self, point: Point, opponent: int) -> bool:
        """Check that the point is a ko point."""
        if len(self.moves) == 0 or self.moves[-1] is None:
            return False

        elif point in _get_adjecent_points(
            self.moves[-1], self.size
        ):  # Check if the last move is captured as the only stone.
            string = self._flod_fill(
                self.invert_bitmap(self.board[opponent]), self.moves[-1]
            )

            if (
                self.count_liberties(string, (opponent + 1) % 2) == 1
                and len(string) == 1
            ):
                return True  # This is the only point where capturing is not allowed.

        return False

    def get_avalible_moves(self, player_index: int) -> Matrix:
        """Return a numpy array of avalible moves."""
        opponent = (player_index + 1) % 2
        moves = self.invert_bitmap(self.board_mask)

        for point in self.points:
            if moves[point] == 1 and self._check_for_surcide(point, opponent):
                # Check if playing uppon that point, captures stones, and isn't a ko
                if self._captures(point, opponent) and not self._ko(point, opponent):
                    continue

                else:
                    moves[point] = 0

        return moves

    def _flod_fill(
        self, mask: Matrix, starting_point: Point, points: List[Point] = None
    ) -> List[Point]:
        """Perform breath first search on the mask and returns
           a list of all the points connected to the starting point."""
        if points is None:
            points = []

        # Only perform recursion if there is nothing at the starting point
        if mask[starting_point] == 0:
            points.append(starting_point)
        else:
            return points

        for point in _get_adjecent_points(starting_point, self.size):
            if point in points:  # Avoid infinite loops
                continue
            else:
                points = self._flod_fill(mask, point, points=points)

        return points

    def _is_adjecent_to(self, point: Point, player_index: int) -> bool:
        """Check if the point is adjecent to a stone of the given player."""
        # Loop over each edjecent point, and checks if there is a stone there.
        for adjecent in _get_adjecent_points(point, self.size):
            if self.board[player_index][adjecent[0]][adjecent[1]] == 1:
                return True

        return False

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

        points = len(stones) + (self.komi if player_index == 1 else 0)
        for string in empty_points_strings:
            # NOTE: Only add adjecent points if they aren't in the string
            border = set()
            for point in string:
                for adjecent in _get_adjecent_points(point, self.size):
                    if adjecent not in string:
                        border.add(adjecent)

            if border.issubset(stones):
                points += len(string)

        return points

    def create_move_tensor(self) -> Matrix:
        """Create a 3d tensor, to mask the latest moves."""
        mask = np.zeros(
            (config["game_parameters"]["moves_given_to_model"], self.size, self.size)
        )

        for idx, move in enumerate(
            reversed(self.moves[-config["game_parameters"]["moves_given_to_model"] :])
        ):
            print(move)
            mask[idx][move] = 1

        return mask
