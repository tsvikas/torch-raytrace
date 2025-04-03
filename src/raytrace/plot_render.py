"""Plot renders."""

# ruff: noqa: T201

import einops
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from matplotlib.axes import Axes

from raytrace import assets
from raytrace.ray_tracing import compute_mesh_intersections, generate_rays_2d


def render_pikachu(
    num_pixels_yz: int = 200,
    yz_limit: float = 2,
    x0: float = -10,
    x1: float = 10,
    device: str | torch.device = "cuda",
) -> tuple[
    Float[torch.Tensor, "{num_pixels_y} {num_pixels_z} 2 xyz"],
    Float[torch.Tensor, "triangles 3 xyz"],
    Float[torch.Tensor, "{num_pixels_y} {num_pixels_z}"],
]:
    """Load and render pikachu."""
    return render_asset(
        asset="pikachu",
        num_pixels_y=num_pixels_yz,
        num_pixels_z=num_pixels_yz,
        y_limit=yz_limit,
        z_limit=-yz_limit,
        x0=x0,
        x1=x1,
        device=device,
    )


def render_asset(  # noqa: PLR0913
    asset: str,
    num_pixels_z: int,
    num_pixels_y: int,
    y_limit: float,
    z_limit: float,
    x0: float,
    x1: float,
    device: str | torch.device = "cuda",
) -> tuple[
    Float[torch.Tensor, "{num_pixels_y} {num_pixels_z} 2 xyz"],
    Float[torch.Tensor, "triangles 3 xyz"],
    Float[torch.Tensor, "{num_pixels_y} {num_pixels_z}"],
]:
    """Load and render an asset."""
    rays = generate_rays_2d(
        num_pixels_y, num_pixels_z, y_limit, z_limit, x0, x1, device=device
    )
    triangles = assets.load(asset, device=device)
    bounding_box = torch.stack(
        [triangles.amin(dim=(0, 1)), triangles.amax(dim=(0, 1))], dim=-1
    ).cpu()
    print(f"{bounding_box=}")
    screen = compute_mesh_intersections(triangles, rays)
    return rays, triangles, screen


def plot_3d(
    lines: Float[torch.Tensor, "lines p2 xyz"],
    points: Float[torch.Tensor, "points xyz"],
    ax: Axes | None = None,
) -> Axes:
    """Plot line segments and points in 3D."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

    lines = lines.cpu()
    for line in lines:
        ax.plot(line[:, 2], line[:, 0], line[:, 1], c="k")

    points = points.cpu()
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], c=points[:, 0])
    scale = torch.linspace(-10, 10, 100)
    ax.scatter(0, scale, 0, c=scale)

    ax.set(xlabel="z", ylabel="x", zlabel="y")
    return ax


def plot_2d(
    screen: Float[torch.Tensor, "z y"],
    points: Float[torch.Tensor, "points xyz"],
    ax: Axes | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> Axes:
    """Plot screen and points in 2D."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(screen.cpu(), cmap="cividis_r", extent=extent)
    points = points.cpu()
    ax.scatter(points[:, 2], points[:, 1], c=points[:, 0])

    ax.set(xlabel="z", ylabel="y")
    return ax


def plot_render(
    rays: Float[torch.Tensor, "{num_pixels_y} {num_pixels_z} 2 xyz"],
    triangles: Float[torch.Tensor, "triangles 3 xyz"],
    screen: Float[torch.Tensor, "{num_pixels_y} {num_pixels_z}"],
) -> None:
    """Plot rays, triangles, and screen."""
    num_pixels_y, num_pixels_z, _p2, _xyz = rays.shape
    _x1, y_limit, z_limit = rays[0, 0, 1].cpu()

    plot_3d(
        points=einops.rearrange(triangles, "triangles p3 xyz -> (triangles p3) xyz"),
        lines=einops.rearrange(
            rays[:: num_pixels_y // 5, :: num_pixels_z // 5],
            "y z p2 xyz -> (y z) p2 xyz",
        ),
    )
    plot_2d(
        screen=screen,
        points=einops.rearrange(triangles, "triangles p3 xyz -> (triangles p3) xyz"),
        extent=(z_limit, -z_limit, -y_limit, y_limit),
    )
    plt.show()


def main() -> None:
    """Load and plot pikachu."""
    plot_render(*render_pikachu())


if __name__ == "__main__":
    main()
