"""Get assets for raytrace."""

from importlib import resources

import torch
from jaxtyping import Float

files = resources.files(__name__)


def load(name: str) -> Float[torch.Tensor, "triangles 3 xyz"]:
    """Load a torch asset."""
    with files.joinpath(f"{name}.pt").open("rb") as f:
        loaded = torch.load(f)
    assert isinstance(loaded, torch.Tensor)
    return loaded
