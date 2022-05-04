#!/usr/bin/env python3
import pytest
from engine.state import State, GameOutcome


def test_scoring():
    """Test the scoring method"""
    dim = 9
    state = State(dim)
    #  After
    assert state.score(0) == 0
