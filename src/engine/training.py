"""Contains code for self play and training of the neural network based uppon self play data."""
# /usr/bin/env python3
from tqdm import tqdm, trange

from engine.misc.types import Dataset, Game, GameOutcome
from engine.model import Model
from engine.misc.config import config
from engine.state import State, get_initial_moves
from engine.MCTS import MCTS, convert_index_to_move
from engine.data import save_games
from tui.repressent_board import get_string_representation

import numpy as np
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
    """Train the model on the dataset."""
    print("Updating model parameters.")
    inputs, target_policies, target_values = (
        torch.from_numpy(dataset.inputs).float(),
        torch.from_numpy(dataset.policies).float(),
        torch.from_numpy(dataset.values).float(),
    )
    number_of_datapoints = inputs.shape[0]

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
                    p,
                    v,
                    torch.unsqueeze(target_policies[indicies], 1),
                    target_values[indicies],
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Update progress bar
            t.set_postfix(
                loss=total_loss
                / (number_of_datapoints / config["training"]["batch_size"])
            )

    return model


def self_play(model: Model, size: int, komi: float, pre_game_moves: int = 2):
    """Use self play to get a new dataset, which will be saved, in the data directory."""
    mcts = MCTS(model)
    policies = []
    games = []

    # TODO: Implement a roolouts window (ie. pick a random number in a specified range.)
    roolouts = config["training"]["roolouts"]
    moves_before_deterministic_play = config["training"][
        "moves_before_deterministic_play"
    ]
    print("Generating training data.")
    for game in trange(config["training"]["number_of_games"], unit="game"):
        # Setup initial state
        state = State(size, komi)
        initial_moves = get_initial_moves(size, pre_game_moves)
        for move in initial_moves:
            state.play_move(move)

        # Play moves until the game is over
        number_of_moves_played = 0
        while (
            state.has_terminated() is False
            and number_of_moves_played <= config["testing"]["maximum_number_of_moves"]
        ):
            # Get move, ether deterministic or random based on the number of moves played
            # TODO: Get the policies aswell
            policy = mcts.get_move_probabilities(state, config["training"]["roolouts"])

            if number_of_moves_played < moves_before_deterministic_play:
                move = convert_index_to_move(
                    np.random.choice(policy.shape[-1], p=policy.flatten())
                )
            else:
                move = convert_index_to_move(np.argmax(policy))

            policies.append(policy)
            state.play_move(move)
            number_of_moves_played += 1

        # Generate game object.
        difference = state.score_relative_to(state.current_player)
        winner = state.current_player if difference > 0 else state.current_opponent
        outcome = GameOutcome(abs(difference), winner)
        games.append(Game(initial_moves, state.moves[2:], outcome, policies))

    save_games(games)
