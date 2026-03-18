PYTHON ?= python3

.PHONY: install format lint test check

install:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	ruff format .

lint:
	ruff check .

test:
	pytest

check:
	ruff format --check .
	ruff check .
	pytest
