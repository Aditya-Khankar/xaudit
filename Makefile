.PHONY: demo test lint format install install-dev clean

demo:
	xaudit demo

test:
	pytest tests/ -v

lint:
	ruff check xaudit/

format:
	black xaudit/

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -name "*.egg-info" -exec rm -rf {} +
