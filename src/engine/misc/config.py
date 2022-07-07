"""Holds the config for the project."""
# /usr/bin/env python3
from numpy.random import geometric


class MCTSConfig:
    """Holds the configuration for the monte carlo tree search."""

    def __init__(self):
        """Initialize config."""
        self.maxmium_depth = 32
        self.c_puct = 1.1


class ModelConfig:
    """Holds the configuration model related information."""

    def __init__(self):
        """Load the model config from the conf.yaml file."""
        # Residual block hyper params
        self.number_of_residual_blocks = 3
        self.number_of_filters = 128

        # Hidden layer hyper params
        self.number_of_hidden_layers = 2
        self.number_of_neurons = 128

        # Head hyper params
        self.value_head_filters = 4
        self.policy_head_filters = 16

        # General hyper params
        self.dropout_rate = 0.3

        self.path_to_model_cache = "models/"


class GameConfig:
    """Holds the configuration game related information."""

    def __init__(self):
        """Load the game config from the conf.yaml file."""
        self.maximum_depth = 32
        self.size = 9  # NOTE: Testing only works when this is set to 9
        self.komi = 4.5
        self.maximum_number_of_moves = 180
        self.moves_given_to_model = 3


class DataConfig:
    """Holds the configuration data related information."""

    def __init__(self):
        """Load the data config from the conf.yaml file."""
        self.path_to_database = "data/"


class TrainingConfig:
    """Holds the configuration training related information."""

    def __init__(self):
        """Load the training config from the conf.yaml file."""
        self.epochs = 10
        self.lr = 0.01  # TODO: experiment with sliding windows
        self.window = 3  # TODO: This should be sliding, (should depend on the generation of the model.)
        self.batch_size = 256
        self.roolouts = 32
        self.weight_decay = 0.005

        self.predetermined_moves = 2
        # Mean is 1 + predetermined moves, with std var appro 9.5
        self.moves_before_deterministic_play = (
            lambda: self.predetermined_moves + geometric(0.1)
        )
        self.maximum_number_of_moves = 140
        self.number_of_games = 20


class TestingConfig:
    """Holds the configuration game related information."""

    def __init__(self):
        """Load the game config from the conf.yaml file."""
        self.roolouts = 8
        self.moves_before_deterministic_play = 0
        self.pre_game_moves = 2
        self.number_of_games = 50
        self.maximum_number_of_moves = 140
        self.games_played_in_tests_of_saved_models = 20


class Config:
    """Holds the entiere config."""

    def __init__(self):
        """Load each config."""
        self.game = GameConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.testing = TestingConfig()
        self.model = ModelConfig()
        self.mcts = MCTSConfig()


cfg = Config()
