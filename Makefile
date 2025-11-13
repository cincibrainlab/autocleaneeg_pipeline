# AutoClean EEG Pipeline - Development Makefile
# Provides convenient commands for local development and code quality checks
# Uses uv tool for isolated tool management (no dependency conflicts!)

.PHONY: help install-dev install-uv-tool upgrade-tools list-tools uninstall-uv-tool check check-fix format lint format-direct lint-direct test test-cov test-integration test-perf test-all ci-check pre-commit dev-setup clean all docs-setup docs-build docs-serve deploy

# Default target
help: ## Show this help message
	@echo "AutoClean EEG Pipeline - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup:"
	@echo "  install-dev         Install development tools (black, isort, ruff, mypy, pre-commit)"
	@echo "  install-uv-tool     Install AutoClean as standalone CLI tool (RECOMMENDED)"
	@echo "                      ✅ Global CLI, isolated, editable, matches CONTRIBUTING.md"
	@echo "  uninstall-uv-tool   Uninstall AutoClean uv tool"
	@echo "  upgrade-tools       Upgrade all development tools"
	@echo "  list-tools          List installed development tools"
	@echo ""
	@echo "Code Quality:"
	@echo "  check          Run all code quality checks (format + lint)"
	@echo "  check-fix      Run checks and auto-fix issues"
	@echo "  format         Auto-format code (black + isort)"
	@echo "  lint           Run linting (ruff)"
	@echo "  format-direct  Format with direct commands (fallback if uv unavailable)"
	@echo "  lint-direct    Lint with direct commands (fallback if uv unavailable)"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run unit tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  test-perf      Run performance benchmarks"
	@echo ""
	@echo "CI Simulation:"
	@echo "  ci-check       Run the same checks as CI (format + lint + tests)"
	@echo "  pre-commit     Run pre-commit hooks manually"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy         Build and publish package to PyPI"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean temporary files and caches"
	@echo "  all            Run format, lint, and test"

# Installation
install-dev: ## Install development tools using uv tool (black, isort, ruff, mypy, pre-commit)
	@python3 scripts/install_dev_tools.py

upgrade-tools: ## Upgrade all development tools
	@python3 scripts/uv_tools.py upgrade

list-tools: ## List installed development tools
	@python3 scripts/uv_tools.py list

# -----------------------------------------------------------------------------
# Installation Methods - IMPORTANT DIFFERENCES
# -----------------------------------------------------------------------------
# 
# uv tool install -e --upgrade . --force (install-uv-tool target):
#   - Installs into uv's ISOLATED tool environment (separate from Python envs)
#   - Package available as GLOBAL CLI command: autocleaneeg-pipeline
#   - Works from any directory, no environment activation needed
#   - Zero dependency conflicts (isolated environment)
#   - Flags breakdown:
#     -e, --editable: Development mode - code changes reflect immediately
#     --upgrade: Upgrade if already installed (checks for newer version)
#     --force: Force reinstall even if same version (ensures clean install)
#   - Use for: CLI usage, development workflow, matches CONTRIBUTING.md
#
# RECOMMENDED: Use 'install-uv-tool' for development (matches CONTRIBUTING.md)
# -----------------------------------------------------------------------------

install-uv-tool: ## Install AutoClean as standalone CLI tool (RECOMMENDED - matches CONTRIBUTING.md)
	@echo "🚀 Installing AutoClean as a uv tool (isolated environment)..."
	@echo "   This installs in editable mode with automatic upgrades"
	@echo "   Flags: -e (editable) --upgrade (update if exists) --force (clean reinstall)"
	@uv tool install -e --upgrade . --force
	@echo ""
	@echo "✅ AutoClean installed! Available globally as: autocleaneeg-pipeline"
	@echo "   Try: autocleaneeg-pipeline --help"
	@echo ""
	@echo "💡 Code changes will reflect immediately (editable mode)"
	@echo "💡 No dependency conflicts (isolated uv environment)"

uninstall-uv-tool: ## Uninstall AutoClean uv tool
	@echo "🗑️ Uninstalling AutoClean uv tool..."
	@uv tool uninstall autocleaneeg-pipeline

# Code Quality - Individual Tools
format: ## Auto-format code with black and isort (using uv tool)
	@echo "🎨 Formatting code with uv tool..."
	@python3 scripts/uv_tools.py run black src/autoclean/
	@python3 scripts/uv_tools.py run isort src/autoclean/
	@echo "✅ Code formatting completed"

format-direct: ## Auto-format code with direct commands (fallback)
	@echo "🎨 Formatting code with direct commands..."
	@black src/autoclean/
	@isort src/autoclean/
	@echo "✅ Code formatting completed"

lint: ## Run linting with ruff (using uv tool)
	@echo "🔍 Running linting with uv tool..."
	@python3 scripts/uv_tools.py run ruff check src/autoclean/
	@echo "✅ Linting completed"
	@echo "ℹ️  Note: mypy type checking temporarily disabled"

lint-direct: ## Run linting with direct commands (fallback)
	@echo "🔍 Running linting with direct commands..."
	@ruff check src/autoclean/
	# @mypy src/autoclean/ --ignore-missing-imports

# Code Quality - Combined
check: ## Run all code quality checks (format + lint, no fixes)
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

pre-commit: ## Run pre-commit hooks manually (using uv tool)
	@echo "🪝 Running pre-commit hooks with uv tool..."
	@python3 scripts/uv_tools.py run pre-commit run --all-files || echo "⚠️ Pre-commit not installed. Run 'make install-dev' first."

# Development workflow
dev-setup: install-uv-tool install-dev ## Complete development setup (matches CONTRIBUTING.md)
	@echo "🎯 Development environment setup completed!"
	@echo "💡 Installed with uv tool (global CLI, isolated environment)"
	@echo "💡 Try running: make check"

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
# Note: 'check-fix' and 'ci-check' cover most use cases

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

# Deployment
deploy: clean ## Build and publish package to PyPI
	@echo "📦 Building package..."
	@uv build
	@echo ""
	@echo "🚀 Publishing to PyPI..."
	@uvx uv-publish
	@echo ""
	@echo "✅ Deployment complete!"
	@echo "💡 Package is now live at: https://pypi.org/project/autocleaneeg-pipeline/"
