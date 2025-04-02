"""Get assets for raytrace."""

from importlib import resources

import torch
from jaxtyping import Float

files = resources.files(__name__)


def load(
    name: str, device: str | torch.device = "cuda"
) -> Float[torch.Tensor, "triangles 3 xyz"]:
    """Load a torch asset."""
    with files.joinpath(f"{name}.pt").open("rb") as f:
        loaded = torch.load(f, map_location=device)
    assert isinstance(loaded, torch.Tensor)
    return loaded
