import pytest

from raytrace import assets


@pytest.mark.parametrize("asset", ["pikachu", "snorlax"])
def test_load(asset: str) -> None:
    assets.load(asset, device="cpu")


def test_missing() -> None:
    with pytest.raises(FileNotFoundError):
        assets.load("missing", device="cpu")
