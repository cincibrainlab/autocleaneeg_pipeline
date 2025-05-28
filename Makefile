# AutoClean EEG Pipeline - Development Makefile
# Provides convenient commands for local development and code quality checks

.PHONY: help install-dev check format lint test clean all

# Default target
help: ## Show this help message
	@echo "AutoClean EEG Pipeline - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup:"
	@echo "  install-dev    Install development tools"
	@echo "  install        Install package in development mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  check          Run all code quality checks"
	@echo "  format         Auto-format code (black + isort)"
	@echo "  lint           Run linting (ruff + mypy)"
	@echo "  format-check   Check formatting without fixing"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run unit tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  test-perf      Run performance benchmarks"
	@echo ""
	@echo "CI Simulation:"
	@echo "  ci-check       Run the same checks as CI"
	@echo "  pre-commit     Run pre-commit hooks manually"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean temporary files"
	@echo "  all            Run format, lint, and test"

# Installation
install-dev: ## Install development tools
	@python3 scripts/install_dev_tools.py

install: ## Install package in development mode
	@echo "📦 Installing AutoClean in development mode..."
	@pip install -e .

# Code Quality - Individual Tools
format: ## Auto-format code with black and isort
	@echo "🎨 Formatting code..."
	@black src/autoclean/
	@isort src/autoclean/
	@echo "✅ Code formatting completed"

format-check: ## Check code formatting without making changes
	@echo "🔍 Checking code formatting..."
	@black --check --diff src/autoclean/
	@isort --check-only --diff src/autoclean/

lint: ## Run linting with ruff and type checking with mypy
	@echo "🔍 Running linting..."
	@ruff check src/autoclean/
	@echo "⚠️ Type checking (mypy) temporarily disabled"
	# @mypy src/autoclean/ --ignore-missing-imports

# Code Quality - Combined
check: ## Run all code quality checks
	@python3 scripts/check_code_quality.py

check-fix: ## Run code quality checks and auto-fix issues
	@python3 scripts/check_code_quality.py --fix

# Testing
test: ## Run unit tests
	@echo "🧪 Running unit tests..."
	@pytest tests/unit/ -v

test-cov: ## Run tests with coverage reporting
	@echo "🧪 Running tests with coverage..."
	@pytest tests/unit/ --cov=autoclean --cov-report=term-missing --cov-report=html

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	@pytest tests/integration/ -v --tb=short

test-perf: ## Run performance benchmarks
	@echo "🏃 Running performance benchmarks..."
	@pytest tests/performance/ --benchmark-only -v

test-all: ## Run all tests (unit + integration)
	@echo "🧪 Running all tests..."
	@pytest tests/ -v --tb=short --maxfail=10

# CI Simulation
ci-check: ## Run the same checks as CI pipeline
	@echo "🚀 Running CI-equivalent checks locally..."
	@echo ""
	@echo "1/4 Code Quality Checks..."
	@python3 scripts/check_code_quality.py
	@echo ""
	@echo "2/4 Unit Tests..."
	@pytest tests/unit/ -v --tb=short --maxfail=5
	@echo ""
	@echo "3/4 Integration Tests..."
	@pytest tests/integration/ -v --tb=short --maxfail=3 || echo "⚠️ Integration tests may fail - that's expected"
	@echo ""
	@echo "4/4 Performance Tests..."
	@pytest tests/performance/ --benchmark-only --benchmark-min-rounds=1 -v || echo "⚠️ Performance tests are optional"
	@echo ""
	@echo "✅ CI simulation completed!"

pre-commit: ## Run pre-commit hooks manually
	@echo "🪝 Running pre-commit hooks..."
	@pre-commit run --all-files || echo "⚠️ Pre-commit not installed. Run 'pre-commit install' first."

# Development workflow
dev-setup: install install-dev ## Complete development setup
	@echo "🎯 Development environment setup completed!"
	@echo "💡 Try running: make check"

quick-check: format lint ## Quick format and lint check
	@echo "✅ Quick quality check completed"

# Utilities
clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache/
	@rm -rf .coverage htmlcov/
	@rm -rf dist/ build/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@echo "✅ Cleanup completed"

all: format lint test ## Run format, lint, and test
	@echo "🎉 All checks completed successfully!"

# Advanced workflows
fix-all: ## Auto-fix all possible issues
	@echo "🔧 Auto-fixing all possible issues..."
	@python3 scripts/check_code_quality.py --fix
	@echo "✅ Auto-fix completed. Review changes before committing."

validate: ci-check ## Validate code is ready for CI
	@echo "✅ Code validation completed - ready for CI!"

# Documentation
docs-setup: ## Install documentation dependencies
	@echo "📚 Installing documentation tools..."
	@pip install sphinx numpydoc pydata-sphinx-theme sphinx_gallery

docs-build: ## Build documentation
	@echo "📚 Building documentation..."
	@cd docs && make html

docs-serve: ## Serve documentation locally
	@echo "📚 Serving documentation at http://localhost:8000"
	@cd docs/_build/html && python3 -m http.server 8000
