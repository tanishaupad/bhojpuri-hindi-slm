.PHONY: install test lint

install:
	python3 -m pip install -e ".[dev]"

test:
	PYTHONPATH=src pytest -q

lint:
	PYTHONPATH=src ruff check src tests
