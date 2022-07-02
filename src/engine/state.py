"""Source for modeling a go state, getting model inputs ect."""
# /usr/bin/env python3
import numpy as np
from typing import List, Iterator
from numpy.typing import ArrayLike
from itertools import product
import random
from engine.misc.types import Point, Board, Matrix, Player, Move
from engine.misc.config import cfg
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


def _flod_fill(
    mask: Matrix, starting_point: Point, size: int, points: List[Point] = None
) -> List[Point]:
    """Perform breath first search on the mask and returns a list of all the points connected to the starting point."""
    if points is None:
        points = []

    # Only perform recursion if there is nothing at the starting point
    if mask[starting_point] == 0:
        points.append(starting_point)
    else:
        return points

    for point in _get_adjecent_points(starting_point, size):
        if point in points:  # Avoid infinite loops
            continue
        else:
            points = _flod_fill(mask, point, size, points=points)

    return points


# TODO: Maybe create a liberties mask & update that, in stead of computing liberties on the fly?
class State:
    """Store information about a game state."""

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
        return Player.Oppoent(self.current_player)

    @staticmethod
    def invert_bitmap(bitmap: Matrix) -> Matrix:
        """Return an inverted bitmap, ie. 1 gets swaped to 0, and 0 to 1."""
        return (bitmap + 1) % 2

    def __init__(self, size: int, komi: float = 0):
        """Initialize a new state, with no handicaps."""
        self.size = size
        self.board = np.zeros((2, size, size), dtype=np.int64)
        self.komi = komi
        self.current_player = Player.BLACK
        self.number_of_passes = 0
        self.moves = []

    # NOTE: This is only used when the human player has to make a move
    def check_move(self, move: Move):
        """Check wether a move is valid, note that this might throw an expection."""
        # It is always legal to pass
        if move == None:
            return

        if move[0] < 0 or move[0] >= self.size or move[1] < 0 or move[1] >= self.size:
            raise ValueError(
                f"invalid move: the point {move} is not in the range {self.size} x {self.size}"
            )
        if self.board_mask[move] != 0:
            raise ValueError(f"Can't make move on {move}, a stone is already there")

        if self._check_for_surcide(move, self.current_opponent) and not self._captures(
            move, self.current_opponent
        ):
            raise ValueError(f"Move: {move} is a surcide move!")

        if self._ko(move, self.current_opponent):
            raise ValueError(f"Move: {move} dosn't respect the ko rule.")

    def remove_group(self, group: List[Point], group_player_index: int):
        """Remove the group from the mask and return the mask."""
        for point in group:
            self.board[group_player_index][point] = 0

    def capture_stones_after_move(self, move: Move):
        """Remove dead stones from the board after playing at a point."""
        for adjecent in _get_adjecent_points(move, self.size):
            # Check if the groups have 0 liberties and remove them if so
            if self.board[self.current_opponent][adjecent] == 1:
                group = _flod_fill(
                    self.invert_bitmap(self.board[self.current_opponent]),
                    adjecent,
                    self.size,
                )
                if self.count_liberties(group, self.current_player) == 0:
                    self.remove_group(group, self.current_opponent)

    def play_move(self, move: Move = None):
        """Play a move on the board, note that a point of none is used to inidicate a pass."""
        if move is not None:
            self.board[self.current_player][move] = 1
            self.capture_stones_after_move(move)

        else:
            self.number_of_passes += 1

        # Update variables
        self.moves.append(move)
        self.current_player = self.current_opponent

    def _check_for_surcide(self, point: Point, opponent_index: int) -> bool:
        """Return True if there is an adjecent point where the mask is 0."""
        for adjecent in _get_adjecent_points(point, self.size):
            if self.board[opponent_index][adjecent[0]][adjecent[1]] == 0:
                return False

        return True

    # NOTE: Here oponent referes to the opponent of the group.
    def count_liberties(self, group: List[Point], opponent_index: int) -> int:
        """Count the number of liberites of a group of stones."""
        n = 0
        # Loop over each point in the group & count it's "liberties".
        for point in group:
            for adjecent in _get_adjecent_points(point, self.size):
                # If the adjecent point is in group, dont count it.
                if (adjecent in group) is False and self.board[opponent_index][
                    adjecent
                ] == 0:
                    n += 1

        return n

    # TODO: Keep track of the groups during computations (this sould make it alot more effective.)
    def _captures(self, point: Point, opponent_index: int) -> bool:
        """Return true if playing uppon that point causes a capture."""
        for adjecent in _get_adjecent_points(point, self.size):
            # Count the number of liberties of the adjecent points, if it's 1 then its 0 after the move.
            if self.board[opponent_index][adjecent] == 1:
                group = _flod_fill(
                    self.invert_bitmap(self.board[opponent_index]), adjecent, self.size
                )
                if self.count_liberties(group, Player.Oppoent(opponent_index)) == 1:
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
            group = _flod_fill(
                self.invert_bitmap(self.board[opponent]), self.moves[-1], self.size
            )

            if (
                self.count_liberties(group, Player.Oppoent(opponent)) == 1
                and len(group) == 1
            ):
                return True  # This is the only point where capturing is not allowed.

        return False

    def get_avalible_moves(self, player_index: int) -> Matrix:
        """Return a numpy array of avalible moves."""
        opponent = Player.Oppoent(player_index)
        moves = self.invert_bitmap(self.board_mask)

        for point in self.points:
            if moves[point] == 1 and self._check_for_surcide(point, opponent):
                # Check if playing uppon that point, captures stones, and isn't a ko
                if self._captures(point, opponent) and not self._ko(point, opponent):
                    continue

                else:
                    moves[point] = 0

        return moves

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
        empty_points_groups = []
        for point in self.points:
            # Skip the point, if it has already been visited or there is a stone
            if point in sum(empty_points_groups, []) or self.board_mask[point] == 1:
                continue

            empty_points_groups.append(_flod_fill(self.board_mask, point, self.size))

        # Compute the scores of each player
        player_score = self._score(player_index, empty_points_groups)
        oponent_score = self._score(Player.Oppoent(player_index), empty_points_groups)

        return player_score - oponent_score

    def _score(self, player_index: int, empty_points_groups: List[List[Point]]) -> int:
        """Compute the score of the given player, using the chinese scoring method."""
        stones = {
            point
            for point in self.points
            if self.board[player_index][point[0]][point[1]] == 1
        }

        points = len(stones) + (self.komi if player_index == 1 else 0)
        for group in empty_points_groups:
            # NOTE: Only add adjecent points if they aren't in the group
            border = set()
            for point in group:
                for adjecent in _get_adjecent_points(point, self.size):
                    if adjecent not in group:
                        border.add(adjecent)

            if border.issubset(stones):
                points += len(group)

        return points

    def get_move_tensor(self) -> Matrix:
        """Create a 3d tensor, to mask the latest moves."""
        mask = np.zeros((cfg.game.moves_given_to_model, self.size, self.size))

        for idx, move in enumerate(
            reversed(self.moves[-cfg.game.moves_given_to_model :])
        ):
            mask[idx][move] = 1

        return mask

    # TODO: there must be a faster way to do this, maybe keep track of the groups seperatly?
    def get_liberties_matrix(self) -> Matrix:
        """ Create a tensor with entery i,j corresponding the number of liberties that the group containing the stone at i,j has, 0 if i,j is an empty point."""
        mask = np.zeros((self.size, self.size))
        points = set()
        for point in self.points:
            # Skip already visited points
            if point in points:
                continue

            # Figure out the index of the player
            player = None
            for i in range(2):
                if self.board[i][point] == 1:
                    player = i
                    break

            if player is not None:
                group = _flod_fill(
                    self.invert_bitmap(self.board[player]), point, self.size
                )
                liberties = self.count_liberties(group, Player.Oppoent(player))

                for connected in group:
                    mask[connected] = liberties

        return mask

    def convert_to_input_tensor(self) -> ArrayLike:
        """Convert the state into a tensor, that is feedable to the model."""
        legal_moves = np.expand_dims(
            self.get_avalible_moves(self.current_player), axis=0
        )
        move_history = self.get_move_tensor()
        board = self.board
        board_mask = np.expand_dims(self.board_mask, axis=0)
        liberties = np.expand_dims(self.get_liberties_matrix(), axis=0)
        inputs = np.concatenate(
            [board, board_mask, liberties, legal_moves, move_history], axis=0
        )

        return inputs

    def has_terminated(self) -> bool:
        """Check if the game has terminated."""
        # The game board is full
        if np.sum(np.sum(self.board_mask, axis=1), axis=0) == self.size * self.size:
            return True

        # Last to moves where passes
        elif len(self.moves) > 2 and self.moves[-1] is None and self.moves[-2] is None:
            return True

        else:
            return False

    def __hash__(self):
        """Make the state hashable, this is based uppon the latest move (respecting ko) and the current position."""
        # NOTE: Take the current board state, the current player and
        # the latest move into acount, this makes sure that the hashing repects ko
        if len(self.moves) > 0:
            return hash((self.board.tobytes(), self.current_player, self.moves[-1]))
        else:
            return hash((self.board.tobytes(), self.current_player))


def get_initial_moves(size: int, number_of_initial_moves: int) -> List[Move]:
    """Get a random list of initial moves."""
    valid_moves = product(range(size), range(size))
    moves = random.choices(list(valid_moves), k=number_of_initial_moves)
    return moves
