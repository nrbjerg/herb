"""This module contains code for testing models against each other."""
# /usr/bin/env python3
from engine.MCTS import MCTS
from engine.model import Model
from engine.state import State, get_initial_moves
from engine.misc.config import config
from engine.misc.types import GameOutcome, Player

from tqdm import trange
from typing import Tuple
from copy import deepcopy


def play_game(state: State, players: Tuple[MCTS], roolouts: int) -> GameOutcome:
    """Play a game and returns the game outcome."""
    for move in range(config["testing"]["maximum_number_of_moves"]):
        player = players[move % 2]
        move = player.get_best_move(state, roolouts)
        state.play_move(move)
        if state.has_terminated():
            difference = state.score_relative_to(state.current_player)
            return GameOutcome(
                abs(difference),
                state.current_player if difference > 0 else state.current_opponent,
            )

    return None


def compute_model_winrate_against(model: Model, opponent: Model) -> float:
    """Compute the models winrate against the oppoent."""
    print("Testing model, against older model.")
    wins = 0
    player, opponent = MCTS(model), MCTS(opponent)

    size = config["game"]["size"]
    pre_game_moves = config["training"]["moves_before_deterministic_play"]
    number_of_games = config["testing"]["number_of_games"]
    roolouts = config["testing"]["roolouts"]

    for game in trange(config["testing"]["number_of_games"], unit="game"):
        # Play initial moves
        state = State(size, config["game"]["komi"])
        for move in get_initial_moves(size, pre_game_moves):
            state.play_move(move)

        # Each player gets a to play each color
        outcome = play_game(deepcopy(state), [player, opponent], roolouts)
        if outcome is not None and outcome.winner == Player.BLACK:
            wins += 1

        outcome = play_game(state, [opponent, player], roolouts)
        if outcome is not None and outcome.winner == Player.WHITE:
            wins += 1

    winrate = wins / (number_of_games * 2)
    print(f"New model winrate: {winrate}")
    return winrate
