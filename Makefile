# Local CI gate — mirrors .github/workflows/ci.yml step-for-step so a green
# `make ci` is a green CI run (CI additionally runs the same steps on a
# 3.11/3.12 matrix). Run before every push.
.PHONY: ci venv lint manifest test

ci: venv lint manifest test

venv:
	uv venv --clear
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .
	uv run ruff format --check .

manifest:
	uv run auspexai-tenant experiment build pkg/ --exact-label

test:
	uv run pytest -q
