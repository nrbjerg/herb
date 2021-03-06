"""Defines the model used in monte carlo tree search."""
# /usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

import os

from engine.misc.config import cfg


# NOTE: Run on cuda if avalible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_to_cache = os.path.join(os.getcwd(), cfg.model.path_to_model_cache)


class ResidualBlock(nn.Module):
    """The building blocks of the residual neural network."""

    def __init__(self):
        """Initialize a residual block."""
        super(ResidualBlock, self).__init__()
        # Initialize convolutional layers
        self.conv1 = nn.Conv2d(
            cfg.model.number_of_filters, cfg.model.number_of_filters, 3, padding=1,
        )
        self.conv2 = nn.Conv2d(
            cfg.model.number_of_filters, cfg.model.number_of_filters, 3, padding=1,
        )

        self.dropout1 = nn.Dropout(p=cfg.model.dropout_rate)
        self.dropout2 = nn.Dropout(p=cfg.model.dropout_rate)
        #
        # Performs batch norm
        self.batch_norm1 = nn.BatchNorm2d(cfg.model.number_of_filters)
        self.batch_norm2 = nn.BatchNorm2d(cfg.model.number_of_filters)

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """Pass the tensor x through the residual block."""
        fx1 = F.relu(self.batch_norm1(self.conv1(x)))
        if training:
            fx1 = self.dropout1(fx1)

        fx2 = self.batch_norm2(self.conv2(x))
        if training:
            fx2 = self.dropout2(x)

        return F.relu(fx2 + x)


class ValueHead(nn.Module):
    """The value head of the model."""

    def __init__(self):
        """Initialize the value head of the network."""
        super(ValueHead, self).__init__()
        # Convolutional filters
        self.conv = nn.Conv2d(
            cfg.model.number_of_filters, cfg.model.value_head_filters, 1
        )
        self.batch_norm = nn.BatchNorm2d(cfg.model.value_head_filters)

        # Dropout layers
        self.dropout_layers = nn.ModuleList(
            [
                nn.Dropout(p=cfg.model.dropout_rate)
                for _ in range(cfg.model.number_of_hidden_layers)
            ]
        )

        # Hidden layers
        hidden_layers = [
            nn.Linear(
                cfg.model.value_head_filters * cfg.game.size * cfg.game.size,
                cfg.model.number_of_neurons,
            )
        ]
        for _ in range(cfg.model.number_of_hidden_layers - 2):
            hidden_layers.append(
                nn.Linear(cfg.model.number_of_neurons, cfg.model.number_of_neurons)
            )

        hidden_layers.append(nn.Linear(cfg.model.number_of_neurons, 1))

        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """Pass data through the value head of the network."""
        # Pass through value filters
        x = torch.flatten(F.relu(self.batch_norm(self.conv(x))), start_dim=1)
        if training:
            x = self.dropout_layers[0](x)

        # Pass through linear layers
        for dropout, hidden in zip(self.dropout_layers[1:], self.hidden_layers[:-1]):
            x = F.relu(hidden(x))
            if training:
                x = dropout(x)

        return torch.tanh(self.hidden_layers[-1](x))


class PolicyHead(nn.Module):
    """The policy head of the neural network."""

    def __init__(self):
        """Initialize the policy head of the network."""
        super(PolicyHead, self).__init__()
        # Convolutional filters
        self.conv = nn.Conv2d(
            cfg.model.number_of_filters, cfg.model.policy_head_filters, 1
        )
        self.batch_norm = nn.BatchNorm2d(cfg.model.policy_head_filters)

        # Dropout layers
        self.dropout_layers = nn.ModuleList(
            [
                nn.Dropout(p=cfg.model.dropout_rate)
                for _ in range(cfg.model.number_of_hidden_layers)
            ]
        )

        # Hidden layers
        hidden_layers = [
            nn.Linear(
                cfg.model.policy_head_filters * cfg.game.size * cfg.game.size,
                cfg.model.number_of_neurons,
            )
        ]
        for _ in range(cfg.model.number_of_hidden_layers - 2):
            hidden_layers.append(
                nn.Linear(cfg.model.number_of_neurons, cfg.model.number_of_neurons)
            )

        # NOTE: We also have an output for passing.
        hidden_layers.append(
            nn.Linear(cfg.model.number_of_neurons, cfg.game.size * cfg.game.size + 1)
        )

        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """Pass data through the policy head of the network."""
        x = torch.flatten(F.relu(self.batch_norm(self.conv(x))), start_dim=1)
        if training:
            x = self.dropout_layers[0](x)

        # Pass data through the hidden layers.
        for dropout, hidden in zip(self.dropout_layers[1:], self.hidden_layers[:-1]):
            x = F.relu(hidden(x))
            if training:
                x = dropout(x)

        if training:
            # If training is true return the logits
            return self.hidden_layers[-1](x)

        else:
            # If training is false return the policy vector
            return torch.exp(torch.log_softmax(self.hidden_layers[-1](x), dim=1))


class Model(nn.Module):
    """The actual model with both the policy head & value head."""

    def __init__(self):
        """Initialize the model."""
        super(Model, self).__init__()
        # NOTE: input axis = 0 has dimension: board (2) + board_mask (1) + liberties (1) + legal_moves (1) + move_history (unknown, but given in config)
        # so a totoal of 5 + move_given_to_model
        # Shared layers
        self.conv = nn.Conv2d(
            5 + cfg.game.moves_given_to_model,
            cfg.model.number_of_filters,
            3,
            padding=1,
        )
        self.batch_norm = nn.BatchNorm2d(cfg.model.number_of_filters)
        self.dropout = nn.Dropout(cfg.model.dropout_rate)

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock() for i in range(cfg.model.number_of_residual_blocks)]
        )

        # Policy & value head
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """Pass data through the network."""
        # Pass the data through the residual part of the network
        x = F.relu(self.batch_norm(self.conv(x)))
        if training:
            x = self.dropout(x)

        for res_block in self.residual_blocks:
            x = res_block(x, training=training)

        # Pass the data through value head and policy head, individually
        p = self.policy_head(x, training=training)
        v = self.value_head(x, training=training)

        # ([x, size*size], [x, 1]) where x is the first dimension of the input tensor
        return (p, v)

    def numberOfParameters(self) -> int:
        """Get the number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())


def save_model(model: Model):
    """Save the model, to the file system."""
    number_of_saved_models = len(os.listdir(path_to_cache))
    file_path = os.path.join(path_to_cache, f"{number_of_saved_models}.pt")
    torch.save(model, file_path)


def load_latest_model() -> Model:
    """Load the latest, model from the file system."""
    number_of_saved_models = len(os.listdir(path_to_cache))
    # If there is a model avalible.
    if number_of_saved_models != 0:
        file_path = os.path.join(path_to_cache, f"{number_of_saved_models - 1}.pt")
        return torch.load(file_path)
    # Create a new model if no model is avalible.
    else:
        model = Model()
        save_model(model)
        return model
