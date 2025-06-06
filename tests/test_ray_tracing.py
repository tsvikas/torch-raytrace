import einops
import torch
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from torch.testing import assert_close

from raytrace import assets, ray_tracing

DEVICE = "cpu"


def test_rays_2d() -> None:
    n_y = 10
    n_z = 10
    y_limit = 0.3
    z_limit = 0.3
    x0 = 0
    x1 = 1
    rays_2d = ray_tracing.generate_rays_2d(
        n_y, n_z, y_limit, z_limit, x0, x1, device=DEVICE
    ).cpu()
    assert rays_2d.shape == (n_y, n_z, 2, 3)
    assert_close(rays_2d[:, :, 0, :], torch.zeros(n_y, n_z, 3))
    end_x, end_y, end_z = einops.rearrange(
        rays_2d[:, :, 1, :],
        "rays_y rays_z xyz -> xyz rays_y rays_z",
        rays_y=n_y,
        rays_z=n_z,
    )
    assert_close(end_x, torch.ones_like(end_z))
    assert_close(
        end_y,
        einops.repeat(
            torch.linspace(y_limit, -y_limit, n_y),
            "rays_y -> rays_y rays_z",
            rays_y=n_y,
            rays_z=n_z,
        ),
    )
    assert_close(
        end_z,
        einops.repeat(
            torch.linspace(z_limit, -z_limit, n_z),
            "rays_z -> rays_y rays_z",
            rays_y=n_y,
            rays_z=n_z,
        ),
    )


def test_raytrace(ndarrays_regression: NDArraysRegressionFixture) -> None:
    triangles = assets.load("pikachu", device=DEVICE)
    num_pixels_y = num_pixels_z = 100
    y_limit = z_limit = 2
    x0 = -10
    x1 = 10
    rays = ray_tracing.generate_rays_2d(
        num_pixels_y, num_pixels_z, y_limit, z_limit, x0, x1, device=triangles.device
    )
    screen = ray_tracing.compute_mesh_intersections(triangles, rays)
    ndarrays_regression.check({"screen": screen.cpu().numpy()})


def test_raytrace_translation_rotation() -> None:
    triangles = assets.load("pikachu", device=DEVICE)
    num_pixels_y = num_pixels_z = 30
    y_limit = z_limit = 2
    x0 = -10
    x1 = 10
    rays = ray_tracing.generate_rays_2d(
        num_pixels_y, num_pixels_z, y_limit, z_limit, x0, x1, device=triangles.device
    )
    screen = ray_tracing.compute_mesh_intersections(triangles, rays)
    # translation
    translation = torch.tensor([10, 20, 30], device=triangles.device)
    screen_translated = ray_tracing.compute_mesh_intersections(
        triangles + translation, rays + translation
    )
    assert_close(screen, screen_translated)
    # rotation
    sin_q = 1 / 2
    cos_q = (1 - sin_q**2) ** 0.5
    rotation = torch.tensor(
        [[1, 0, 0], [0, cos_q, sin_q], [0, -sin_q, cos_q]], device=triangles.device
    )
    screen_rotated = ray_tracing.compute_mesh_intersections(
        triangles @ rotation, rays @ rotation
    )
    assert_close(screen, screen_rotated)
