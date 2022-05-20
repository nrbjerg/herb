#!/usr/bin/env python3
from engine.model import Model
import torch


def test_dimensions():
    """Tests that the model output is the expected dimension"""
    batch_size = 32
    size = 9
    model = Model()
    tensor = torch.rand((batch_size, 2, size, size))
    pol, val = model(tensor)
    assert pol.shape == (batch_size, size * size)
    assert val.shape == (batch_size, 1)
