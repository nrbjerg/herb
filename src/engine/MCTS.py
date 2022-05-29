"""Source code for implementing monte carlo tree search."""
# /usr/bin/env python3
from engine.model import Model
from engine.state import State
from engine.misc.types import Matrix, Move, Hash, Pass
from typing import Tuple
from engine.misc.config import config
import numpy as np
import torch
from copy import deepcopy


size = config["game"]["size"]
c_puct = 1.1


def convert_index_to_move(index: int) -> Move:
    """Convert the output from the model to an actual point."""
    if index == size ** 2:
        return Pass
    else:
        return (index // size, index % size)


def add_noise(policy: Matrix):
    """Add noise to the policy matrix."""
    pass


class MCTS:
    """A parallel montecarlo tree search algorithm."""

    def __init__(self, model: Model):
        """Intialize the MCTS tree."""
        self.size = config["game"]["size"]
        self.maximum_depth = 64
        self.model = model
        # Cache model predictions for speed & efficiency.
        self.predictions = {}

        # NOTE: THE FUCK IS A Q VALUE?
        self.Qsa = {}  # The Q values for s,a
        self.Nsa = {}  # The number of times the action a has been taken from state s
        self.Ns = {}  # The number of times the state has been visited

        self.Ps = {}  # Stores the initial policy (from the neural network)

        # Stores if the current state has terminated (1 for win, -1 for not terminated)
        self.Es = {}
        self.Vs = {}  # Stores the valid moves for the state s

    def get_move_probabilities(
        self, state: State, roolouts: int, temperature: float = 1.0
    ) -> Matrix:
        """Compute the action probabilities."""
        # Populate dictionaries
        for _ in range(roolouts):
            self.search(state)

        key = state.__hash__()

        counts = np.array(
            [[self.Nsa.get((key, move), 0) for move in range(self.size ** 2 + 1)]],
            dtype="float32",
        )

        counts[0][: self.size ** 2] *= state.get_avalible_moves(
            state.current_player
        ).flatten()
        # TEST: Its very weird that the highest number seems to move, with the times
        # this function is called, could be a quincidence or maybe its the neural network

        if temperature == 0.0:
            # Play deterministicly
            probs = np.zeros((1, size * size + 1), dtype="float32")
            probs[0][np.argmax(counts)] = 1.0
            return probs

        else:
            return counts / np.sum(counts)

    def get_best_move(self, state: State, roolouts: int) -> Move:
        """Get the best move for testing purposes."""
        move_index = np.argmax(
            self.get_move_probabilities(state, roolouts, temperature=0.0)
        )
        return convert_index_to_move(move_index)

    def get_training_move(self, state: State, roolouts: int) -> Move:
        """Get a move for trainig purposes."""
        probs = self.get_move_probabilities(state, roolouts).flatten()
        return convert_index_to_move(np.random.choice(len(probs), p=probs))

    def compute_UCB_score(self, move: int, key: Hash, prior: float) -> float:
        """Compute the usb score."""
        if (
            key,
            move,
        ) in self.Qsa:  # NOTE: if its in the Qsa dict, then its also in the Nsa dict.
            return self.Qsa[(key, move)] + c_puct * prior * np.sqrt(self.Ns[key]) / (
                1 + self.Nsa[(key, move)]
            )
        else:
            return (
                c_puct * prior * np.sqrt(self.Ns[key] + 1e-8)
            )  # the 1e-8 makes sure that its never 0

    def pick_move_index_with_highest_UCB_score(self, key: Hash) -> int:
        """Return the index of the move with the highest UCB score."""
        # NOTE: That the pass encoded as (size * size) is always valid
        scores = [
            self.compute_UCB_score(move, key, self.Ps[key][move])
            if self.Vs[key][move] == 1
            else 0
            for move in range(self.size ** 2 + 1)
        ]

        move_index = np.argmax(scores)

        return move_index

    def pick_move_with_highest_UCB_score(self, key: Hash) -> Move:
        """Pick the move with the highest UCB score."""
        return convert_index_to_move(self.pick_move_index_with_highest_UCB_score(key))

    def predict_with_model(self, state: State, key: Hash) -> Tuple[Matrix, float]:
        """Perform predictions with the moved, if needed."""
        if key not in self.predictions:
            policy, value = self.model(
                torch.from_numpy(
                    np.expand_dims(state.convert_to_input_tensor(), axis=0)
                ).float()
            )

            # NOTE: The indicies is because the model returns 3d tensors.
            self.predictions[key] = (
                policy[0].detach().numpy(),
                value[0][0].detach().numpy(),
            )

        return self.predictions[key]

    def search(self, state: State, depth: int = 0):
        """Search through the posibilities."""
        if depth == self.maximum_depth:
            return 0

        state = deepcopy(state)  # NOTE: Don't mutate the original state.
        key = state.__hash__()

        # Check if the state has been terminated
        if key not in self.Es:
            self.Es[key] = state.has_terminated()

        # If the state has been terminated, return the score of who won
        if self.Es[key] is True:
            return state.score_relative_to(
                state.current_opponent
            )  # The result, based uppon the last player

        if key not in self.Ps:  # We have hit a leaf node
            self.Ps[key], v = self.predict_with_model(state, key)

            # Add the pass to this.
            valid_moves = np.ones((self.size ** 2 + 1,))
            valid_moves[: self.size ** 2] = np.reshape(
                state.get_avalible_moves(state.current_player), (self.size ** 2),
            )

            self.Ps[key] *= valid_moves

            # Normalize the policy
            k = np.sum(self.Ps[key])
            if k > 0:
                self.Ps[key] = self.Ps[key] / k

            else:
                print("Warning: all moves where masked.")
                self.Ps[key] = valid_moves / np.sum(valid_moves)

            # This is not a problem since this is a leaf node then.
            self.Vs[key] = valid_moves
            self.Ns[key] = 0

            # This should be from the view of there current player, therefore -v is backprobagated.
            return -v

        move_index = self.pick_move_index_with_highest_UCB_score(key)
        new_state = deepcopy(state)
        new_state.play_move(convert_index_to_move(move_index))
        v = self.search(
            new_state, depth=depth + 1
        )  # NOTE: Continue searching until we reach a leaf node

        # Update information
        if (key, move_index) in self.Qsa:
            self.Qsa[(key, move_index)] = (
                self.Nsa[(key, move_index)] * self.Qsa[(key, move_index)] + v
            ) / (self.Nsa[(key, move_index)] + 1)
            self.Nsa[(key, move_index)] += 1

        # Initialize information
        else:
            self.Qsa[(key, move_index)] = v
            self.Nsa[(key, move_index)] = 1

        self.Ns[key] += 1

        return -v

    def reset(self, hard_reset: bool = False):
        """Reset the dictionaries, and if hard == True, also reset the predictions."""
        self.Qsa, self.Nsa, self.Ns, self.Es, self.Ps, self.Vs = {}, {}, {}, {}, {}, {}
        if hard_reset:
            self.predictions = {}
