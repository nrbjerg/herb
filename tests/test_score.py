#!/usr/bin/env python3
from engine.state import State
from .conf_test import load_test_state


def test_scoring():
    """Test the scoring method"""
    dim = 9
    # Check that the empty board has score 0
    state = State(dim)
    assert state.score_relative_to(0) == 0

    # Check the test state
    state = load_test_state()
    assert state.score_relative_to(0) == 7


def test_from_SGF():
    """Tests the load from smart game format function, note that the make move function also gets tested"""
    pass
