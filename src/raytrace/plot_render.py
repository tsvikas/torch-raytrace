"""Plot renders."""

from typing import Literal

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
    axis: Literal["yz", "zx", "xy"] = "yz",
    device: str | torch.device = "cuda",
) -> tuple[
    Literal["yz", "zx", "xy"],
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
        axis=axis,
        device=device,
    )


def render_snorlax(
    num_pixels_yz: int = 200,
    yz_limit: float = 100,
    x0: float = -400,
    x1: float = 100,
    axis: Literal["yz", "zx", "xy"] = "zx",
    device: str | torch.device = "cuda",
) -> tuple[
    Literal["yz", "zx", "xy"],
    Float[torch.Tensor, "{num_pixels_y} {num_pixels_z} 2 xyz"],
    Float[torch.Tensor, "triangles 3 xyz"],
    Float[torch.Tensor, "{num_pixels_y} {num_pixels_z}"],
]:
    """Load and render pikachu."""
    return render_asset(
        asset="snorlax",
        num_pixels_y=num_pixels_yz,
        num_pixels_z=num_pixels_yz,
        y_limit=yz_limit,
        z_limit=-yz_limit,
        x0=x0,
        x1=x1,
        axis=axis,
        device=device,
    )


def render_asset(
    asset: str,
    num_pixels_z: int,
    num_pixels_y: int,
    y_limit: float,
    z_limit: float,
    x0: float,
    x1: float,
    axis: Literal["yz", "zx", "xy"] = "yz",
    device: str | torch.device = "cuda",
) -> tuple[
    Literal["yz", "zx", "xy"],
    Float[torch.Tensor, "{num_pixels_y} {num_pixels_z} 2 xyz"],
    Float[torch.Tensor, "triangles 3 xyz"],
    Float[torch.Tensor, "{num_pixels_y} {num_pixels_z}"],
]:
    """Load and render an asset."""
    rays = generate_rays_2d(
        num_pixels_y, num_pixels_z, y_limit, z_limit, x0, x1, axis, device=device
    )
    triangles = assets.load(asset, device=device)
    bounding_box = torch.stack(
        [triangles.amin(dim=(0, 1)), triangles.amax(dim=(0, 1))], dim=-1
    ).cpu()
    print(f"{bounding_box=}")
    screen = compute_mesh_intersections(triangles, rays)
    return axis, rays, triangles, screen


def plot_3d(
    lines: Float[torch.Tensor, "lines p2 xyz"],
    points: Float[torch.Tensor, "points xyz"] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot line segments and points in 3D."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

    lines = lines.cpu()
    for line in lines:
        ax.plot(line[:, 2], line[:, 0], line[:, 1], c="k")

    if points is not None:
        points = points.cpu()
        ax.scatter(points[:, 2], points[:, 0], points[:, 1], c=points[:, 0])
    scale = torch.linspace(-10, 10, 100)
    ax.scatter(0, scale, 0, c=scale)

    ax.set(xlabel="z", ylabel="x", zlabel="y")
    return ax


def plot_2d(
    screen: Float[torch.Tensor, "z y"],
    points: Float[torch.Tensor, "points xyz"] | None = None,
    ax: Axes | None = None,
    extent: tuple[float, float, float, float] | None = None,
    axis: Literal["yz", "zx", "xy"] = "yz",
) -> Axes:
    """Plot screen and points in 2D."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(screen.cpu(), cmap="cividis_r", extent=extent)

    if points is not None:
        points = points.cpu()
        axis1 = ord(axis[0]) - ord("x")
        axis2 = ord(axis[1]) - ord("x")
        axis0 = 3 - axis1 - axis2
        ax.scatter(points[:, axis2], points[:, axis1], c=points[:, axis0])

    ax.set(xlabel=axis[1], ylabel=axis[0])
    return ax


def plot_render(
    axis: Literal["yz", "zx", "xy"],
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
        axis=axis,
    )
    plt.show()


def main() -> None:
    """Load and plot pikachu."""
    plot_render(*render_pikachu())


if __name__ == "__main__":
    main()
