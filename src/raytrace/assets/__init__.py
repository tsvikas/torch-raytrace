from importlib import resources

import torch

files = resources.files(__name__)


def load(name: str):
    return torch.load(files / f"{name}.pt")
