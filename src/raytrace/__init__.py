"""torch-raytrace: Ray tracing from scratch with PyTorch.

Copyright (c) 2025 Tsvika Shapira. All rights reserved.
"""

from . import plot_render, ray_tracing
from ._version import version as _version

__version__ = _version
__all__: list[str] = [
    "plot_render",
    "ray_tracing",
]
