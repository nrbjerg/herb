#!/usr/bin/env python3
from engine.model import Model
from engine.misc.config import config
import torch


def test_dimensions():
    """Tests that the model output is the expected dimension"""
    batch_size = 32
    size = 9
    model = Model()
    tensor = torch.rand(
        (batch_size, 5 + config["game"]["moves_given_to_model"], size, size)
    )
    pol, val = model(tensor)
    assert pol.shape == (batch_size, size * size + 1)
    assert val.shape == (batch_size, 1)
