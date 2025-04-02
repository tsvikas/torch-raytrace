import importlib

import raytrace


def test_version() -> None:
    assert importlib.metadata.version("raytrace") == raytrace.__version__
