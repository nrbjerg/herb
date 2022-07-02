#!/usr/bin/env python3
from typing import Dict, Any


class ModelConfig:
    """Holds the configuration model related information."""

    def __init__(self):
        """Load the model config from the conf.yaml file."""
        # Residual block hyper params
        self.number_of_residual_blocks = 3
        self.number_of_filters = 64

        # Hidden layer hyper params
        self.number_of_hidden_layers = 2
        self.number_of_neurons = 128

        # Head hyper params
        self.value_head_filters = 4
        self.policy_head_filters = 16

        # General hyper params
        self.perform_batch_norm = True
        self.dropout_rate = 0.3

        self.path_to_model_cache = "models/"


class GameConfig:
    """Holds the configuration game related information."""

    def __init__(self):
        """Load the game config from the conf.yaml file."""
        self.size = 5  # NOTE: Testing only works when this is set to 9
        self.komi = 4.5
        self.maximum_number_of_moves = 180
        self.moves_given_to_model = 5


class DataConfig:
    """Holds the configuration data related information."""

    def __init__(self):
        """Load the data config from the conf.yaml file."""
        self.path_to_database = "data/"
        self.window = 100
        self.games_per_generation = 81


class TrainingConfig:
    """Holds the configuration training related information."""

    def __init__(self):
        """Load the training config from the conf.yaml file."""
        self.epochs = 100
        self.lr = 0.005  # TODO: experiment with sliding windows
        self.batch_size = 128
        self.roolouts = 20
        self.weight_decay = 0.005
        self.moves_before_deterministic_play = 20
        self.maximum_number_of_moves = 140
        self.number_of_games = 10


class TestingConfig:
    """Holds the configuration game related information."""

    def __init__(self):
        """Load the game config from the conf.yaml file."""
        self.roolouts = 10
        self.moves_before_deterministic_play = 0
        self.pre_game_moves = 2
        self.number_of_games = 10
        self.maximum_number_of_moves = 140


class Config:
    """Holds the entiere config."""

    def __init__(self):
        """Load each config."""
        self.game_cfg = GameConfig()
        self.data_cfg = DataConfig()
        self.training_cfg = TrainingConfig()
        self.testing_cfg = TestingConfig()
        self.model_cfg = ModelConfig()
