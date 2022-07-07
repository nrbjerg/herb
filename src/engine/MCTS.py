"""Source code for implementing monte carlo tree search."""
# /usr/bin/env python3
from __future__ import annotations
from engine.model import Model
from engine.state import State
from engine.misc.types import Matrix, Move, Pass
from typing import Tuple, Dict, List
from engine.misc.config import cfg
import numpy as np
import torch
from copy import deepcopy


def convert_index_to_move(index: int) -> Move:
    """Convert the output from the model to an actual point."""
    if index == cfg.game.size ** 2:
        return Pass
    else:
        return (index // cfg.game.size, index % cfg.game.size)


class Node:
    """Models a node in the monte carlo tree."""

    def __init__(self, state: State, parent: Node, prior: float = 0.0):
        """Initialize the node."""
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.total_value = 0
        self.visits = 0
        self.is_expanded = False

    @property
    def Q(self) -> float:
        """Compute the average value."""
        return self.total_value / (1 + self.visits)

    @property
    def U(self) -> float:
        """Compute the upper confidence score."""
        return np.sqrt(self.parent.visits) * self.prior / (1 + self.visits)

    @property
    def child_scores(self) -> List:
        """Return a matrix of the child scores."""
        scores = np.zeros((cfg.game.size ** 2 + 1))
        for move, child in self.children.items():
            scores[move] = child.Q + cfg.mcts.c_puct * child.U

        return scores

    @property
    def child_visits(self) -> Matrix:
        """Return a matrix of the child scores."""
        scores = np.zeros((1, cfg.game.size ** 2 + 1))
        for move, child in self.children.items():
            scores[0][move] = child.visits

        return scores

    def best_move_and_child(self) -> Node:
        """Pick the best move and child."""
        # This sequence can be empty? NOTE: Where is it called, and does expanded mean that there is atleast a child?
        return max(
            self.children.items(),
            key=lambda move_and_child: move_and_child[1].Q
            + cfg.mcts.c_puct * move_and_child[1].U,
        )

    def best_child(self) -> Node:
        """Pick the best child."""
        return self.best_move_and_child()[1]

    @property
    def best_move(self) -> Move:
        """Pick the best move."""
        return self.best_move_and_child()[0]

    def select_leaf(self) -> Node:
        """Select the best child, until a leaf is picked."""
        current = self
        while current.is_expanded:
            current = current.best_child()
        return current

    def expand_node(self, child_priors: Matrix):  # TODO: Add dirichlet noise.
        """Add the children of the node."""
        self.is_expanded = True
        legal_moves = self.state.get_avalible_moves(self.state.current_player)
        child_priors[:-1] = child_priors[-1] * legal_moves.reshape(cfg.game.size ** 2)
        for move, prior in enumerate(child_priors):
            if prior > 0:
                state = deepcopy(self.state)
                state.play_move(convert_index_to_move(move))

                # Back propagatethe outcome of the game if the game has been termintated
                if state.has_terminated():
                    score = state.score_relative_to(state.current_player)
                    value = 1 if score < 0 else -1
                    self.back_prop_value(value)

                # Otherwise add the child.
                else:
                    self.children[move] = Node(state, parent=self, prior=prior)

        # If there are no children, set is expanded to false.
        if len(self.children) == 0:
            self.is_expanded = False

    def back_prop_value(self, value: float):
        """Back propagates the value back up through the tree."""
        self.visits += 1
        self.total_value += value
        if self.parent is not None:  # Back propagate until the root is reached.
            self.parent.back_prop_value(-value)


class MCTS:
    """The monte carlo tree search algorithm."""

    def __init__(self, model: Model):
        """Intialize the algorithm."""
        self.model = model
        self.size = cfg.game.size
        self.evaluated_states: Dict[State, Node] = {}  # state: node
        self.predictions: Dict[State, Tuple[float, Matrix]] = {}

    def get_move_probabilities(
        self, state: State, roolouts: int, temperature: float = 1.0
    ) -> Matrix:
        """Compute the action probabilities."""
        root = self.search(state, roolouts)
        visits = root.child_visits
        total = sum(sum(visits))

        if temperature == 0.0 or total == 0.0:
            # Play deterministicly:
            probs = np.zeros((1, cfg.game.size ** 2 + 1), dtype="float32")
            probs[0][np.argmax(visits)] = 1.0
            return probs

        else:
            # Scale the visists acording to the temperature.
            pi = np.power(visits, 1 / temperature)
            probs = pi / sum(sum(pi))
            return probs

    def get_best_move(self, state: State, roolouts: int) -> Move:
        """Get the best move for testing purposes."""
        root = self.search(state, roolouts)
        return root.best_move

    def search(self, state: State, roolouts: int) -> Node:
        """Perform the actual search. expanding the tree ect. and returns the root of the search."""
        # Load the state if posible.
        if state in self.evaluated_states.keys():
            root = self.evaluated_states[state]
            root.parent = None
        else:
            root = Node(state, None)
            self.evaluated_states[state] = root

        for _ in range(roolouts):
            leaf = root.select_leaf()

            # Perform predictions and backprop
            p, v = self.predict_with_model(state)
            leaf.back_prop_value(v)
            leaf.expand_node(p)

            # Store for later.
            self.evaluated_states[leaf.state.__hash__()] = leaf

        return root

    def get_training_move(self, state: State, roolouts: int) -> Move:
        """Get a move for trainig purposes."""
        return convert_index_to_move(
            np.random.choice(
                cfg.game.szie ** 2 + 1, p=self.get_move_probabilities(state, roolouts)
            )
        )

    def predict_with_model(self, state: State) -> Tuple[Matrix, float]:
        """Perform predictions with the moved, if needed."""
        if state not in self.predictions:
            policy, value = self.model(
                torch.from_numpy(
                    np.expand_dims(state.convert_to_input_tensor(), axis=0)
                ).float()
            )

            # NOTE: The indicies is because the model returns 3d tensors.
            self.predictions[state] = (
                policy[0].detach().numpy(),
                value[0][0].detach().numpy(),
            )

        return self.predictions[state]
