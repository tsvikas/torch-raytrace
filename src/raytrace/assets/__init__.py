"""Get assets for raytrace."""

from importlib import resources

import torch
from jaxtyping import Float

files = resources.files(__name__)


def load(
    name: str, device: str | torch.device = "cuda"
) -> Float[torch.Tensor, "triangles 3 xyz"]:
    """Load a torch asset."""
    if files.joinpath(f"{name}.pt").is_file():
        with files.joinpath(f"{name}.pt").open("rb") as f:
            loaded = torch.load(f, map_location=device)
    elif files.joinpath(f"{name}.stl").is_file():
        from stl import mesh

        with files.joinpath(f"{name}.stl").open("rb") as f:
            loaded_mesh = mesh.Mesh.from_file(filename=f.name, fh=f)
        loaded = torch.tensor(loaded_mesh.vectors.copy(), device=device)
    else:
        raise FileNotFoundError
    assert isinstance(loaded, torch.Tensor)
    return loaded
