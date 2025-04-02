import einops
import torch
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from torch.testing import assert_close

from raytrace import assets, ray_tracing


def test_rays_2d() -> None:
    n_y = 10
    n_z = 10
    y_limit = 0.3
    z_limit = 0.3
    rays_2d = ray_tracing.generate_rays_2d(n_y, n_z, y_limit, z_limit).cpu()
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
    triangles = assets.load("pikachu")
    num_pixels_y = num_pixels_z = 200
    y_limit = z_limit = 1
    screen = ray_tracing.perform_ray_tracing(
        triangles, num_pixels_y, num_pixels_z, y_limit, z_limit
    )
    ndarrays_regression.check({"screen": screen.numpy()})
