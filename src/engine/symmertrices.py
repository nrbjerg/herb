#!/usr/bin/env python3
from engine.misc.types import Matrix, Datapoint
from engine.misc.config import config

import numpy as np
from numpy.typing import ArrayLike
from typing import List

size = config["game"]["size"]


def _add_pass_to_policy(policy_matrix: Matrix, pass_policy: float) -> Matrix:
    """Add the pass policy reshapes the policy matrix & adds the pass policy to the back."""
    policy = np.zeros((1, size ** 2 + 1))
    policy[0][:-1] = np.reshape(policy_matrix, (size ** 2))
    policy[0][-1] = pass_policy

    return policy


def extract_symertries(
    inputs: ArrayLike, policy: Matrix, value: float
) -> List[Datapoint]:
    """Extract extra data, from the fact that the board has 8 symertries."""
    # along main diagonal: flip horizontally, flip vertically, rot90 flip
    pass_policy = policy[0][-1]
    policy_matrix = np.reshape(policy[0][:-1], (size, size))
    reflections = [
        # Along the middle
        Datapoint(
            np.flip(inputs, axis=1),
            _add_pass_to_policy(np.flip(policy_matrix, axis=0), pass_policy),
            value,
        ),
        Datapoint(
            np.flip(inputs, axis=2),
            _add_pass_to_policy(np.flip(policy_matrix, axis=1), pass_policy),
            value,
        ),
        # TODO: Along the diagonals
        # Datapoint(
        #     np.flip(np.flip(inputs, axis=1), axis=2),
        #     np.flip(np.flip(policy, axis=0), axis=1),
        #     value,
        # ),
        # Datapoint(
        #     np.flip(np.flip(inputs, axis=1), axis=2),
        #     np.flip(np.flip(policy, axis=0), axis=1),
        #     value,
        # ),
    ]

    rotations = [
        Datapoint(
            np.rot90(inputs, k, axes=(1, 2)),
            _add_pass_to_policy(np.rot90(policy_matrix, k, axes=(0, 1)), pass_policy),
            value,
        )
        for k in [1, 2, 3]
    ]

    normal = [Datapoint(inputs, policy, value)]

    return normal
