"""Contains code for self play and training of the neural network based uppon self play data."""
# /usr/bin/env python3
from tqdm import trange

from engine.misc.types import Dataset, Game, GameOutcome
from engine.model import Model
from engine.misc.config import cfg
from engine.state import State, get_initial_moves
from engine.MCTS import MCTS, convert_index_to_move
from engine.data import save_games

import numpy as np
import torch
from torch.optim import Adam, SGD
from torch import Tensor, nn

mse = nn.MSELoss()


def cross_entropy(predictions: Tensor, targets: Tensor) -> float:
    """Compute the cross entropy loss."""
    return torch.mean(-torch.sum(targets * torch.log_softmax(predictions, dim=1), 1))


def train(dataset: Dataset, model: Model) -> Model:
    """Train the model on the dataset."""
    print("Updating model parameters.")
    inputs, target_policies, target_values = (
        torch.from_numpy(dataset.inputs).float(),
        torch.from_numpy(dataset.policies).float(),
        torch.from_numpy(dataset.values).float(),
    )
    number_of_datapoints = inputs.shape[0]

    optimizer = SGD(model.parameters(), lr=cfg.training.lr,)
    with trange(cfg.training.epochs, unit="epoch") as t:
        for epoch in t:
            permutation = torch.randperm(number_of_datapoints)

            total_p_loss, total_v_loss = 0, 0
            for idx in range(0, number_of_datapoints, cfg.training.batch_size):
                optimizer.zero_grad()

                # Pass batch thorugh model
                indicies = permutation[idx : idx + cfg.training.batch_size]
                p, v = model(inputs[indicies], training=True)

                # Compute the loss
                p_loss = cross_entropy(p, target_policies[indicies])
                v_loss = mse(v, torch.unsqueeze(target_values[indicies], 1))
                loss = v_loss + p_loss

                # Optimize model
                loss.backward()
                optimizer.step()

                total_p_loss += p_loss.item()
                total_v_loss += v_loss.item()

            # Update progress bar
            t.set_postfix(
                p=total_p_loss / (number_of_datapoints / cfg.training.batch_size),
                v=total_v_loss / (number_of_datapoints / cfg.training.batch_size),
            )

    return model


def self_play(model: Model, size: int, komi: float):
    """Use self play to get a new dataset, which will be saved, in the data directory."""
    mcts = MCTS(model)
    games = []

    # TODO: Implement a roolouts window (ie. pick a random number in a specified range.)
    print("Generating training data.")
    for game in trange(cfg.training.number_of_games, unit="game"):
        policies = []
        # Setup initial state
        state = State(size, komi)
        initial_moves = get_initial_moves(size, cfg.training.predetermined_moves)
        for move in initial_moves:
            state.play_move(move)

        # Play moves until the game is over
        number_of_moves_played = cfg.training.predetermined_moves
        moves_played_before_deterministic_play = (
            cfg.training.moves_before_deterministic_play()
        )
        while (
            state.has_terminated() is False
            and number_of_moves_played <= cfg.testing.maximum_number_of_moves
        ):
            # Get move, ether deterministic or random based on the number of moves played
            policy = mcts.get_move_probabilities(state, cfg.training.roolouts)

            if number_of_moves_played < moves_played_before_deterministic_play:
                move = convert_index_to_move(
                    np.random.choice(policy.shape[-1], p=policy.flatten())
                )
            else:
                move = convert_index_to_move(np.argmax(policy))

            policies.append(policy)
            state.play_move(move)
            number_of_moves_played += 1

        # Generate game object. NOTE: Komi makes sure that it's never a tie.
        difference = state.score_relative_to(state.current_player)
        winner = state.current_player if difference > 0 else state.current_opponent
        outcome = GameOutcome(abs(difference), winner)

        games.append(
            Game(
                initial_moves,
                state.moves[cfg.training.predetermined_moves :],
                outcome,
                policies,
            )
        )

    save_games(games)
