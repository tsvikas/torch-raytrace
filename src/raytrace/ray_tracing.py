"""Triangle Rendering Module.

This module implements a basic graphics renderer in PyTorch for ray tracing,
focusing on batched matrix operations.

Key Concepts:
- **Camera**: Emitting rays from (x0, 0, 0) to a screen at x=x1.
- **Screen**: A plane where ray intersections are visualized.
- **Objects**: Made of triangles, defined by three 3D points.
- **Rays**: Defined by an origin and a next-point, emitted from the camera.
"""

import einops
import torch as t
from jaxtyping import Bool, Float
from torch import linalg


def generate_rays_2d(  # noqa: PLR0913
    num_pixels_y: int,
    num_pixels_z: int,
    y_limit: float,
    z_limit: float,
    x0: float,
    x1: float,
    device: str | t.device = "cuda",
) -> Float[t.Tensor, "{num_pixels_y} {num_pixels_z} 2 xyz"]:
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
    rays = t.zeros((2, 3, num_pixels_y, num_pixels_z), dtype=t.float32, device=device)
    rays[0, 0] = x0
    rays[1, 0] = x1
    rays[1, 1] = einops.repeat(
        t.linspace(y_limit, -y_limit, num_pixels_y),
        "y -> y z",
        y=num_pixels_y,
        z=num_pixels_z,
    )
    rays[1, 2] = einops.repeat(
        t.linspace(z_limit, -z_limit, num_pixels_z),
        "z -> y z",
        y=num_pixels_y,
        z=num_pixels_z,
    )
    rays = einops.rearrange(
        rays, "p2 xyz y z -> y z p2 xyz", y=num_pixels_y, z=num_pixels_z, p2=2, xyz=3
    )
    return rays


def compute_mesh_intersections(
    triangles: Float[t.Tensor, "triangles 3 xyz"], rays: Float[t.Tensor, "*rays 2 xyz"]
) -> Float[t.Tensor, "*rays"]:
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
    left: Float[t.Tensor, "rays triangles xyz suv"] = t.stack(
        [Os - Ds, Bs - As, Cs - As], dim=-1
    )
    right: Float[t.Tensor, "rays triangles xyz"] = Os - As

    threshold = 1e-5
    irreversible: Float[t.Tensor, "rays triangles"] = linalg.det(left).abs() < threshold
    left[irreversible] = t.eye(left.shape[-1], device=left.device)

    solve: Float[t.Tensor, "rays triangles suv"] = linalg.solve(left, right)
    s, u, v = einops.rearrange(solve, "rays triangles suv -> suv rays triangles")

    seen: Bool[t.Tensor, "rays triangles"] = (
        (u > 0) & (v > 0) & (u + v < 1) & (s > 0) & ~irreversible
    )
    seen_s: Float[t.Tensor, "rays triangles"] = s.where(
        seen, t.tensor(float("inf"), device=seen.device)
    )
    dist: Float[t.Tensor, "rays"] = seen_s.amin(-1)
    (dist,) = einops.unpack(dist, rays_packed_shape, "*")
    return dist


def perform_ray_tracing(  # noqa: PLR0913
    triangles: Float[t.Tensor, "triangles 3 xyz"],
    num_pixels_y: int,
    num_pixels_z: int,
    y_limit: float,
    z_limit: float,
    x0: float = -1,
    x1: float = 0,
) -> Float[t.Tensor, "{num_pixels_y} {num_pixels_z}"]:
    """Perform Ray Tracing on a Mesh.

    This function executes ray tracing to render a 2D image of a mesh composed of
    triangles.

    Parameters:
    - triangles: Tensor containing the vertices of the triangles in the mesh.
    - num_pixels_y: Number of pixels along the y-axis.
    - num_pixels_z: Number of pixels along the z-axis.
    - y_limit: Maximum extent of rays along the y-axis at x=1.
    - z_limit: Maximum extent of rays along the z-axis at x=1.

    Process:
    1. Generates rays emitted from the origin spanning the specified y and z limits.
    2. Adjusts the ray origins for proper viewing of the mesh.
    3. Rotates the scene for perspective using a 90-degree rotation around the y-axis.
    4. Computes the intersection of rays with the mesh triangles.
    5. Reshapes and returns the resulting intersection distances in a 2D format.

    Returns:
    - A tensor representing the intersection distances as a pixel grid.
    """
    rays: Float[t.Tensor, "{num_pixels_y} {num_pixels_z} 2 xyz"] = generate_rays_2d(
        num_pixels_y, num_pixels_z, y_limit, z_limit, x0, x1, device=triangles.device
    )
    new_origin = t.zeros_like(rays)
    new_origin[..., 0, :] = t.tensor([-2, 0, 0], device=triangles.device)
    phi = t.tensor([0 * 3.1415 / 180], device=triangles.device)
    c, s = t.cos(phi), t.sin(phi)
    rot = t.tensor([[[[c, 0, s], [0, 1, 0], [-s, 0, c]]]], device=triangles.device)
    new_rays = (rays + new_origin) @ rot
    assert rays.shape == new_rays.shape
    screen: Float[t.Tensor, "{num_pixels_y} {num_pixels_z}"] = (
        compute_mesh_intersections(triangles, new_rays)
    )
    return screen


if __name__ == "__main__":
    import assets  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt

    triangles: Float[t.Tensor, "triangles 3 xyz"] = assets.load("pikachu")
    num_pixels_y = num_pixels_z = 200  # 120
    y_limit = z_limit = 1
    x0 = -1
    x1 = 0
    screen = perform_ray_tracing(
        triangles, num_pixels_y, num_pixels_z, y_limit, z_limit, x0, x1
    ).cpu()
    plt.imshow(screen, cmap="cividis_r")
    plt.show()
