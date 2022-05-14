#!/usr/bin/env python3
from engine.state import State
from .conf_test import (
    load_test_state,
    load_ko_and_surcide_state,
    load_string_capture_state,
)
from tui.repressent_board import get_string_representation
import numpy as np


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
            assert state._check_for_surcide(point, 0)
        else:
            assert not state._check_for_surcide(point, 0)

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


def test_captures():
    """Test that the capturing mechanicsms work"""
    # test cases for single stones
    state = load_ko_and_surcide_state()
    state.current_player = 1
    state.play_move((2, 5))
    assert state.board[0][2][5] == 0

    state.current_player = 1
    state.play_move((8, 7))
    assert state.board[0][8][8] == 0

    state.current_player = 1
    state.play_move((5, 0))
    assert state.board[0][4][0] == 0

    # test cases for strings
    state = load_string_capture_state()
    state.current_player = 1
    state.play_move((3, 1))
    assert state.board[0][3][2] == 0
    assert state.board[0][3][3] == 0

    state.play_move((7, 8))
    assert state.board[1][7][8] == 0
    assert state.board[1][8][8] == 0
    assert state.board[1][8][7] == 0
