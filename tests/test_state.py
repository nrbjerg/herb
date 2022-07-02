#!/usr/bin/env python3
from engine.state import State
from .conf_test import (
    load_test_state,
    load_ko_and_surcide_state,
    load_string_capture_state,
)
from engine.misc.config import cfg
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
    state.moves.append((2, 6))

    assert state.get_avalible_moves(1)[2][5] == 0


def test_captures():
    """Test that the capturing mechanicsms work"""
    # test cases for single stones
    state = load_ko_and_surcide_state()
    state.current_player = 1
    state.play_move((2, 5))
    assert state.board[0][2][6] == 0
    assert state.board[1][2][5] == 1

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

    print("testing move tensor")
    mask = np.zeros((cfg.game.moves_given_to_model, cfg.game.size, cfg.game.size))
    mask[0, 7, 8] = 1
    mask[1, 3, 1] = 1
    assert np.array_equal(mask, state.get_move_tensor())


def test_liberites_mask():
    """Tests the liberties mask."""
    state = load_ko_and_surcide_state()
    liberties = state.get_liberties_matrix()

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 2, 0, 0],
            [0, 0, 0, 0, 4, 0, 1, 3, 0],
            [2, 0, 0, 0, 0, 3, 2, 0, 0],
            [1, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3, 1],
            [0, 0, 0, 0, 0, 0, 3, 0, 1],
        ]
    )

    assert np.array_equal(liberties, expected)


def test_hash():
    """Tests that the hashing algorithm is functional."""
    state1 = State(9, 9)
    state2 = State(9, 9)

    assert state1.__hash__() == state2.__hash__()

    state1.play_move((5, 5))
    state2.play_move((5, 5))
    assert state1.__hash__() == state2.__hash__()

    state1.play_move((4, 4))
    state1.play_move((2, 2))
    state1.play_move((3, 3))

    state2.play_move((3, 3))
    state2.play_move((2, 2))
    state2.play_move((4, 4))
    assert state1.__hash__() != state2.__hash__()
