#!/usr/bin/env python3
import numpy as np
from typing import List, Tuple
from engine.misc.types import Board, Dataset
from engine.model import Model
from engine.misc import config

from tqdm import trange

import torch
from torch.optim import Adam
from torch import Tensor, nn

mse = nn.MSELoss()


def cross_entropy(predictions: Tensor, targets: Tensor) -> float:
    """Compute the cross entropy loss."""
    return torch.mean(-torch.sum(targets * torch.log_softmax(predictions, dim=1), 1))


def loss_function(p, v, p_target, v_target):
    """Compute the loss, given the model outputs & the target outputs."""
    return mse(v, v_target) + cross_entropy(p, p_target)


def train(dataset: Dataset, model: Model) -> Model:
    """Train the model on the dataset"""
    inputs, target_policies, target_values = (
        Dataset.inputs,
        Dataset.policies,
        Dataset.values,
    )
    number_of_datapoints = inputs.size()[0]

    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    with trange(config["training"]["epochs"], unit="epoch") as t:
        for epoch in t:

            permutation = torch.randperm(number_of_datapoints)

            total_loss = 0
            for idx in range(0, number_of_datapoints, config["training"]["batch_size"]):
                optimizer.zero_grad()
                indicies = permutation[idx : idx + config["training"]["batch_size"]]

                # Compute the loss
                p, v = model(inputs[indicies], training=True)

                loss = loss_function(
                    p, v, target_policies[indicies], target_values[indicies]
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Update progress bar?
            t.set_postfix(
                loss=total_loss
                / (number_of_datapoints % config["training"]["batch_size"])
            )

    return model


def self_play(model: Model):
    """Use self play to get a new dataset, which will be saved, in the data directory"""
    pass
