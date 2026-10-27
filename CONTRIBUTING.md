# Contributing to LongCache

Thanks for your interest in contributing to LongCache! Here is how to get started.

## Development Setup

We recommend using [uv](https://github.com/astral-sh/uv) for fast and deterministic local development:

```bash
git clone https://github.com/ENDEVSOLS/LongCache.git
cd LongCache

# With uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[all,dev,docs]"

# Or with standard pip
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all,dev,docs]"
```

## Running Tests

```bash
# Run all unit and integration tests
pytest

# Run tests with terminal coverage summary
pytest --cov=src/longcache --cov-report=term-missing

# Run a specific test suite
pytest tests/test_cache.py -v
pytest tests/test_backends.py -v
```

If you wish to test Redis integrations locally:
```bash
docker compose up -d redis
pytest tests/test_redis_backend.py -v
```

## Code Style & Formatting

We use [Ruff](https://github.com/astral-sh/ruff) for fast linting and code formatting:

```bash
# Lint checks
ruff check .

# Automated fixes
ruff check --fix .

# Code formatting check
ruff format --check .

# Format all files
ruff format .
```

Type checking with `mypy`:

```bash
mypy src/longcache/ --ignore-missing-imports
```

## Documentation

Documentation is built with Material for MkDocs:

```bash
# Preview docs locally with live reload
mkdocs serve

# Validate docs build with strict checking
mkdocs build --strict
```

## Pull Request Guidelines

1. **Keep it focused**: One bug fix or feature per pull request.
2. **Add tests**: Any bug fix or new feature must include tests in `tests/`.
3. **Run CI checks locally**: Ensure `make lint` and `make test` pass before pushing.
4. **Update docs**: Update the corresponding documentation in `docs/` and docstrings if public APIs are modified.
5. **Update CHANGELOG.md**: Add an entry under the `[Unreleased]` section following the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
