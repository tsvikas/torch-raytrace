"""Triangle Rendering Module.

This module implements a basic graphics renderer in PyTorch for ray tracing,
focusing on batched matrix operations.

Key Concepts:
- **Camera**: Emitting rays from (x0, 0, 0) to a screen at x=x1.
- **Screen**: A plane where ray intersections are visualized.
- **Objects**: Made of triangles, defined by three 3D points.
- **Rays**: Defined by an origin and a next-point, emitted from the camera.
"""

from typing import Literal

import einops
import torch
from jaxtyping import Bool, Float
from torch import linalg


def generate_rays_2d(
    num_pixels_y: int,
    num_pixels_z: int,
    y_limit: float,
    z_limit: float,
    x0: float,
    x1: float,
    axis: Literal["yz", "zx", "xy"] = "yz",
    device: str | torch.device = "cuda",
) -> Float[torch.Tensor, "{num_pixels_y} {num_pixels_z} 2 xyz"]:
    """Generate 2D Rays from the Origin.

    This function creates rays emitted from the origin (x0, 0, 0) in both x and y
    dimensions, forming a pyramid shape with the tip at the origin.

    Parameters:
    num_pixels_y: Number of pixels in the y dimension.
    num_pixels_z: Number of pixels in the z dimension.
    y_limit: At x=x1, rays extend from -y_limit to +y_limit.
    z_limit: At x=x1, rays extend from -z_limit to +z_limit.

    Returns: the origin and next-point of each ray.
    """
    axis1 = ord(axis[0]) - ord("x")
    axis2 = ord(axis[1]) - ord("x")
    axis0 = 3 - axis1 - axis2

    rays = torch.zeros(
        (2, 3, num_pixels_y, num_pixels_z), dtype=torch.float32, device=device
    )
    rays[0, axis0] = x0
    rays[1, axis0] = x1
    rays[1, axis1] = einops.repeat(
        torch.linspace(y_limit, -y_limit, num_pixels_y),
        "y -> y z",
        y=num_pixels_y,
        z=num_pixels_z,
    )
    rays[1, axis2] = einops.repeat(
        torch.linspace(z_limit, -z_limit, num_pixels_z),
        "z -> y z",
        y=num_pixels_y,
        z=num_pixels_z,
    )
    rays = einops.rearrange(
        rays, "p2 xyz y z -> y z p2 xyz", y=num_pixels_y, z=num_pixels_z, p2=2, xyz=3
    )
    return rays


def compute_mesh_intersections(
    triangles: Float[torch.Tensor, "triangles 3 xyz"],
    rays: Float[torch.Tensor, "*rays 2 xyz"],
) -> Float[torch.Tensor, "*rays"]:
    """Ray Tracing for Mesh Rendering.

    This function performs ray tracing to determine the closest intersection distance
    between rays and a mesh of triangles.

    Parameters:
    triangles: the vertices of each triangle.
    rays: the origin and next-point of each ray.

    Returns:
    the distance to the closest intersecting triangle or infinity.
    """
    rays, rays_packed_shape = einops.pack([rays], "* p2 xyz")

    n_triangles = triangles.shape[0]
    n_rays = rays.shape[0]

    # create (n_rays, xyz) tensors:
    As, Bs, Cs = einops.repeat(  # noqa: N806
        triangles, "triangles p3 xyz -> p3 rays triangles xyz", p3=3, xyz=3, rays=n_rays
    )
    Os, Ds = einops.repeat(  # noqa: N806
        rays, "rays p2 xyz -> p2 rays triangles xyz", p2=2, xyz=3, triangles=n_triangles
    )
    # solve
    left: Float[torch.Tensor, "rays triangles xyz suv"] = torch.stack(
        [Os - Ds, Bs - As, Cs - As], dim=-1
    )
    right: Float[torch.Tensor, "rays triangles xyz"] = Os - As

    threshold = 1e-5
    irreversible: Float[torch.Tensor, "rays triangles"] = (
        linalg.det(left).abs() < threshold
    )
    left[irreversible] = torch.eye(left.shape[-1], device=left.device)

    solve: Float[torch.Tensor, "rays triangles suv"] = linalg.solve(left, right)
    s, u, v = einops.rearrange(solve, "rays triangles suv -> suv rays triangles")

    seen: Bool[torch.Tensor, "rays triangles"] = (
        (u > 0) & (v > 0) & (u + v < 1) & (s > 0) & ~irreversible
    )
    seen_s: Float[torch.Tensor, "rays triangles"] = s.where(
        seen, torch.tensor(float("inf"), device=seen.device)
    )
    dist: Float[torch.Tensor, "rays"] = seen_s.amin(-1)
    (dist,) = einops.unpack(dist, rays_packed_shape, "*")
    return dist
