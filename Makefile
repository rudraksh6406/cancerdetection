.PHONY: help install test lint format clean build docker-build docker-run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

test: ## Run tests
	python -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Run linting checks
	flake8 src/ main.py tests/
	mypy src/ main.py --ignore-missing-imports

format: ## Format code with black
	black src/ main.py tests/

format-check: ## Check code formatting
	black --check --diff src/ main.py tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

build: ## Build the package
	python setup.py sdist bdist_wheel

docker-build: ## Build Docker image
	docker build -t cancer-prediction .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD)/data:/app/data cancer-prediction

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop docker-compose services
	docker-compose down

run: ## Run the main application
	python main.py

example: ## Run the example script
	python examples/basic_usage.py

all: install lint test ## Install, lint, and test 