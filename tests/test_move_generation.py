#!/usr/bin/env python3
import pytest
from engine.state import State
from .conf_test import load_test_state
import numpy as np


def test_move_generation():
    """Tests that surcide moves are not allowed"""
    state = load_test_state()
    black_avalible_moves = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
        ]
    )
    white_avalible_moves = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
        ]
    )
    assert np.array_equal(state.get_avalible_moves(0), black_avalible_moves)
    assert np.array_equal(state.get_avalible_moves(1), white_avalible_moves)
