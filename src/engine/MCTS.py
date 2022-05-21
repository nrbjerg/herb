#!/usr/bin/env python3
from engine.model import Model
from engine.misc.types import Matrix


class Node:
    pass


def PUCT(child: Node, c_puct: float, prior: float):
    """Computes the PUCT score of a node."""
    pass


def add_noise(policy: Matrix):
    """Add noise to the policy matrix."""
    pass


class MCTS:
    """A parallel montecarlo tree search algorithm"""

    def __init__(self, model: Model):
        self.games = []
        self.model
