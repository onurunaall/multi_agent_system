.PHONY: help install test lint format clean build docs

help:
	@echo "Available commands:"
	@echo "  install    Install dependencies using Poetry"
	@echo "  test       Run tests with coverage"
	@echo "  lint       Run linting checks"
	@echo "  format     Format code with black and isort"
	@echo "  clean      Clean up cache and build files"
	@echo "  build      Build distribution packages"
	@echo "  docs       Generate documentation"
	@echo "  run        Run the application"

install:
	poetry install

test:
	poetry run pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint:
	poetry run flake8 .
	poetry run mypy .
	poetry run black --check .
	poetry run isort --check-only .

format:
	poetry run black .
	poetry run isort .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

build:
	poetry build

docs:
	cd docs && poetry run make html

run:
	poetry run python main.py

# Development shortcuts
dev-install:
	poetry install --with dev,docs
	poetry run pre-commit install

update-deps:
	poetry update
	poetry export -f requirements.txt --output requirements.txt --without-hashes

security-check:
	poetry run pip-audit
