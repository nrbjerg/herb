#!/usr/bin/env python3
import pytest
from engine.state import State
from .conf_test import load_test_state
from tui.repressent_board import get_string_representation
import numpy as np


def load_ko_and_surcide_state():
    # Test for KO & surcide
    white = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    black = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
    ]

    state = State(9)
    state.board = np.array([black, white])

    print("Current state")
    print(get_string_representation(state))

    return state


def test_move_surcide_moves():
    """Tests that surcide moves are not allowed"""
    # Test for basic move generation
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

    print("Check that avalible moves works")
    assert np.array_equal(state.get_avalible_moves(0), black_avalible_moves)
    assert np.array_equal(state.get_avalible_moves(1), white_avalible_moves)

    # Check that it finds the surcide moves and nothing else

    state = load_ko_and_surcide_state()

    print("Testing surcide moves")
    for point in state.points:
        if point in [(8, 7), (5, 0), (2, 5)]:
            assert state._check_for_surcide(point, 0) == True
        else:
            assert state._check_for_surcide(point, 0) == False

    # print(get_string_representation(state))
    print("Testing liberties")
    assert state.count_liberties([(2, 6)], 1) == 1

    print("Testing that capturing surcidal moves are allowed")
    assert state.get_avalible_moves(1)[5][0] == 1
    assert state.get_avalible_moves(1)[2][5] == 1
    assert state.get_avalible_moves(1)[8][7] == 1  # This is just a surcide.


def test_ko():
    """Test that the ko functionallity works."""
    state = load_ko_and_surcide_state()
    state.last_move = (2, 6)

    assert state.get_avalible_moves(1)[2][5] == 0
