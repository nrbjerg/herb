#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from test_model import play_game, setup_state
from engine.model import Model
from engine.MCTS import MCTS
from engine.misc.config import cfg
from typing import Dict, Any, List, Tuple
import random


def random_pairs(items: List[Any]) -> List[Tuple[Any]]:
    """Pick random pairs out of the list of items."""
    pairs = []
    while len(items) >= 2:
        pairs.append(tuple(random.choices(items, k=2)))
        # Remove picked pairs from list
        x, y = pairs[-1]
        items.remove(x)
        items.remove(y)

    return pairs


def compute_relative_winrates(models: Dict[str, Model]) -> Dict[str, float]:
    """Compute the relative eloes of the models."""
    agents = {gen: MCTS(m) for gen, m in models.items()}
    winrates = {gen: 0.0 for gen in models.keys()}

    for _ in range(cfg.testing.games_played_in_tests_of_saved_models):
        # Create random pairs and match them against each other
        for (g1, g2) in random_pairs(models.keys()):
            state = setup_state()
            outcome1 = play_game(
                deepcopy(state), (agents[g1], agents[g2]), cfg.testing.roolouts,
            )
            outcome2 = play_game(state, (agents[g2], agents[g1]), cfg.testing.roolouts)

            # Update winrates. TODO:
    return winrates


def plot_winrates():
    """Loads the models, computes the winrates and plots them."""
    pass
