"""Triangle Rendering Module

This module implements a basic graphics renderer in PyTorch for ray tracing,
focusing on batched matrix operations.

Key Concepts:
- **Camera**: Positioned at the origin, emitting rays towards a screen at x=1.
- **Screen**: A plane where ray intersections are visualized.
- **Objects**: Made of triangles, defined by three 3D points.
- **Rays**: Defined by an origin and a direction, emitted from the camera.

"""

import einops
import torch as t
from torch import linalg


def generate_rays_2d(
    num_pixels_y: int, num_pixels_z, y_limit: float, z_limit: float
) -> t.Tensor:
    """Generates 2D Rays from the Origin

    This function creates rays emitted from the origin (0, 0, 0) in both y and z dimensions,
    forming a pyramid shape with the tip at the origin.

    Parameters:
    - num_pixels_y (int): Number of pixels in the y dimension.
    - num_pixels_z (int): Number of pixels in the z dimension.
    - y_limit (float): At x=1, rays extend from -y_limit to +y_limit.
    - z_limit (float): At x=1, rays extend from -z_limit to +z_limit.

    Returns:
    - A tensor of shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3),
        representing the origin and direction of each ray.
    """
    source = einops.repeat(
        t.tensor([0, 0, 0], dtype=t.float),
        "xyz -> pixels xyz",
        pixels=num_pixels_y * num_pixels_z,
    )
    direction = t.cartesian_prod(
        t.tensor([1.0]),
        t.linspace(y_limit, -y_limit, num_pixels_y),
        t.linspace(z_limit, -z_limit, num_pixels_z),
    )
    return t.stack([source, direction], dim=1)


def compute_mesh_intersections(triangles: t.Tensor, rays: t.Tensor) -> t.Tensor:
    """
    Ray Tracing for Mesh Rendering

    This function performs ray tracing to determine the closest intersection distance between rays and a mesh of triangles.

    Parameters:
    - triangles (Tensor): Shape (n_triangles, points=3, dims=3) representing the vertices of each triangle.
    - rays (Tensor): Shape (n_pixels, points=2, dims=3) representing the origin and direction of each ray.

    Returns:
    - A tensor of shape (n_pixels,) indicating the distance to the closest intersecting triangle or infinity if no intersection occurs.

    Process:
    1. Calculate the intersection point using the triangle-ray intersection formula.
    2. Determine barycentric coordinates (u, v) to verify if the intersection is within triangle bounds.
    3. Handle multiple intersections by finding the minimum distance for each ray.
    4. Ensure numerical stability by checking for degeneracy in the intersection equation.
    5. Return intersection distances; use infinity for rays not intersecting any triangles.
    """
    device = t.device("cuda:0")
    triangles = triangles.to(device)
    rays = rays.to(device)

    n_triangles = triangles.shape[0]
    n_pixels = rays.shape[0]
    assert triangles.shape == (n_triangles, 3, 3)
    assert rays.shape == (n_pixels, 2, 3)
    # create (n_pixels, xyz) tensors:
    As, Bs, Cs = einops.repeat(  # noqa: N806
        triangles,
        "n_triangles p xyz -> p n_pixels n_triangles xyz",
        n_pixels=n_pixels,
        xyz=3,
    )
    Os, Ds = einops.repeat(  # noqa: N806
        rays, "n_pixels p xyz -> p n_pixels n_triangles xyz", n_triangles=n_triangles
    )
    # solve
    left = t.stack([-Ds, Bs - As, Cs - As], dim=-1)  # n_pixels n_triangles xyz suv
    right = Os - As  # n_pixels n_triangles xyz

    threshold = 1e-5
    irreversible = linalg.det(left).abs() < threshold  # n_pixels n_triangles
    left[irreversible] = t.eye(left.shape[-1], device=device)

    solve = linalg.solve(left, right)
    s, u, v = einops.rearrange(
        solve, "n_pixels n_triangles suv -> suv n_pixels n_triangles"
    )

    seen = (
        (u > 0) & (v > 0) & (u + v < 1) & (s > 0) & ~irreversible
    )  # n_pixels n_triangles
    seen_s = s.where(seen, t.tensor(float("inf"), device=device))
    dist_s = seen_s.amin(-1)  # n_pixels
    dist = dist_s / rays[:, 1].pow(2).sum(-1).sqrt()
    return dist.cpu()


def perform_ray_tracing(triangles, num_pixels_y, num_pixels_z, y_limit, z_limit):
    rays = generate_rays_2d(
        num_pixels_y, num_pixels_z, y_limit, z_limit
    )  # pixels point xyz
    new_origin = t.zeros_like(rays)
    new_origin[:, 0, :] = t.tensor([-2, 0, 0])
    phi = t.Tensor([90 * 3.1415 / 180])
    c, s = t.cos(phi), t.sin(phi)
    rot = t.tensor([[[[c, 0, s], [0, 1, 0], [-s, 0, c]]]])
    screen = compute_mesh_intersections(triangles, ((rays + new_origin) @ rot)[0])
    return screen.reshape((num_pixels_y, num_pixels_z))


if __name__ == "__main__":
    import assets  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt

    triangles = assets.load("pikachu")
    num_pixels_y = num_pixels_z = 200  # 120
    y_limit = z_limit = 1
    screen = perform_ray_tracing(
        triangles, num_pixels_y, num_pixels_z, y_limit, z_limit
    )
    plt.imshow(screen, cmap="cividis_r")
    plt.show()
