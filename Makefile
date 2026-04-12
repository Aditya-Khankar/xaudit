.PHONY: demo test lint format install install-dev clean

demo:
	cognidrift demo

test:
	pytest tests/ -v

lint:
	ruff check cognidrift/

format:
	black cognidrift/

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -name "*.egg-info" -exec rm -rf {} +
