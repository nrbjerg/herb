#!/usr/bin/env python3
import yaml
from typing import Dict, Any
import os


def load_config():
    """Load the entiere config from the current working directory."""
    path = os.path.join(os.getcwd(), "conf.yaml")
    with open(path, "r") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

    return conf


config = load_config()
