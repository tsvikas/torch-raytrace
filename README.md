# torch-raytrace

[![Tests][tests-badge]][tests-link]
[![uv][uv-badge]][uv-link]
[![Ruff][ruff-badge]][ruff-link]
[![Black][black-badge]][black-link]
\
[![Made Using tsvikas/python-template][template-badge]][template-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![PRs Welcome][prs-welcome-badge]][prs-welcome-link]

## Overview

Ray tracing from scratch with PyTorch for educational and illustrative purposes.

## Usage

Install the package using pip, or with a dependency manager like uv:

```bash
pip install torch-raytrace
```

and import the package in your code:

```python
import raytrace
```

## Development

### Getting started

- install [git][install-git], [uv][install-uv].
- git clone this repo: `git clone https://github.com/tsvikas/torch-raytrace.git`
- run `uv run just prepare`

### Tests and code quality

- use `uv run just format` to format the code.
- use `uv run just lint` to see linting errors.
- use `uv run just test` to run tests.
- use `uv run just check` to run all the checks (format, lint, test, and pre-commit).
- Run a specific tool directly, with `uv run pytest`/`ruff`/`mypy`/`black`/...

<!---
Badges to add, when needed:
# TODO: enable after adding RTD documentation
[![Documentation Status][rtd-badge]][rtd-link]
# TODO: enable after writing tests, than follow the link to setup codecov
[![codecov][codecov-badge]][codecov-link]
\
# TODO: enable after uploading to PyPI
[![PyPI version][pypi-version-badge]][pypi-link]
[![PyPI platforms][pypi-platforms-badge]][pypi-link]
[![Total downloads][pepy-badge]][pepy-link]

[codecov-badge]: https://codecov.io/gh/tsvikas/torch-raytrace/graph/badge.svg
[codecov-link]: https://codecov.io/gh/tsvikas/torch-raytrace
[pepy-badge]: https://img.shields.io/pepy/dt/torch-raytrace
[pepy-link]: https://pepy.tech/project/torch-raytrace
[pypi-link]: https://pypi.org/project/torch-raytrace/
[pypi-platforms-badge]: https://img.shields.io/pypi/pyversions/torch-raytrace
[pypi-version-badge]: https://img.shields.io/pypi/v/torch-raytrace
[rtd-badge]: https://readthedocs.org/projects/torch-raytrace/badge/?version=latest
[rtd-link]: https://torch-raytrace.readthedocs.io/en/latest/?badge=latest
-->

[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]: https://github.com/tsvikas/torch-raytrace/discussions
[install-git]: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
[install-uv]: https://docs.astral.sh/uv/getting-started/installation/
[prs-welcome-badge]: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
[prs-welcome-link]: http://makeapullrequest.com
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-link]: https://github.com/astral-sh/ruff
[template-badge]: https://img.shields.io/badge/%F0%9F%9A%80_Made_Using-tsvikas%2Fpython--template-gold
[template-link]: https://github.com/tsvikas/python-template
[tests-badge]: https://github.com/tsvikas/torch-raytrace/actions/workflows/ci.yml/badge.svg
[tests-link]: https://github.com/tsvikas/torch-raytrace/actions/workflows/ci.yml
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[uv-link]: https://github.com/astral-sh/uv
