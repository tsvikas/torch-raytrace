"""Get assets for raytrace."""
from importlib import resources

import torch

files = resources.files(__name__)


def load(name: str):
    """Load a torch asset."""
    return torch.load(files / f"{name}.pt")
