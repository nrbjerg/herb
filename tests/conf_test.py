#!/usr/bin/env python3
from engine.state import State
import numpy as np


def load_test_state() -> State:
    """Load a test state, used for testning"""
    # The position is taken from REF: https://senseis.xmp.net/?ChineseCountingExample
    state = State(9)
    white = [
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
    ]

    black = [
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
    ]

    state.board = np.array([black, white])
    return state
