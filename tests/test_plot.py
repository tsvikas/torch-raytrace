from typing import Literal

import pytest

from raytrace import plot_render


def test_render_pikachu() -> None:
    plot_render.render_pikachu(10)


def test_render_snorlax() -> None:
    plot_render.render_snorlax(10)


@pytest.mark.parametrize("axis", ["xy", "yz", "zx"])
def test_plot_render(axis: Literal["yz", "zx", "xy"]) -> None:
    plot_render.plot_render(*plot_render.render_pikachu(10, axis=axis))


def test_pikachu_side() -> None:
    plot_render.pikachu_side(10, None)
