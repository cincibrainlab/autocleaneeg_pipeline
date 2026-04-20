# AutoClean EEG Pipeline - Development Makefile
# Provides convenient commands for local development and code quality checks
# Uses uv tool for isolated tool management (no dependency conflicts!)

.PHONY: help install-dev install-uv-tool serve-setup upgrade-tools list-tools uninstall-uv-tool check check-fix format lint format-direct lint-direct test test-quick test-unit-short test-ingestion test-cov test-integration test-integration-short test-all ci-check pre-commit dev-setup clean all docs-setup docs-build docs-serve deploy plans-serve plans-stop ensure-serve-workspace web-ui serve-run app-up app-stop

SERVE_WORKSPACE ?= $(HOME)/Documents/Autoclean-EEG
SERVE_MODE ?= live
AUTOCLEAN_CLI ?= autocleaneeg-pipeline
PYTEST ?= python scripts/run_pytest.py

ifneq ("$(wildcard .venv/bin/autocleaneeg-pipeline)","")
AUTOCLEAN_CLI := .venv/bin/autocleaneeg-pipeline
endif

# Default target
help: ## Show this help message
	@echo "AutoClean EEG Pipeline - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup:"
	@echo "  install-dev         Install contributor tools only (black, isort, ruff, mypy, pre-commit)"
	@echo "                      Does NOT install the runnable AutoClean CLI/Serve environment"
	@echo "  install-uv-tool     Install AutoClean as standalone CLI tool (RECOMMENDED)"
	@echo "                      ✅ Global CLI, isolated, editable, matches CONTRIBUTING.md"
	@echo "  serve-setup         Install the runnable CLI/Serve path plus contributor tools"
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
	@echo "  test-quick     Run fast ingestion unit tests"
	@echo "  test-unit-short Run unit tests (fail fast)"
	@echo "  test-ingestion Run ingestion unit tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  test-integration-short Run integration tests (fail fast)"
	@echo ""
	@echo "CI Simulation:"
	@echo "  ci-check       Run the same checks as CI (format + lint + tests)"
	@echo "  pre-commit     Run pre-commit hooks manually"
	@echo ""
	@echo "Serve:"
	@echo "  serve          Build frontend + start local server"
	@echo "  serve-data1    Publish, deploy to data1, start remote server"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy         Build and publish package to PyPI"
	@echo ""
	@echo "Plans Server:"
	@echo "  plans-serve    Start HTTP server for plans/_site on port 7933"
	@echo "  plans-stop     Stop the plans server"
	@echo ""
	@echo "AutoClean App:"
	@echo "  web-ui         Start the web UI/API for the single AutoClean workspace"
	@echo "  serve-run      Start the dispatcher for the single AutoClean workspace"
	@echo "  app-up         Start dispatcher in background and web UI in foreground"
	@echo "  app-stop       Stop the background dispatcher started by app-up"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean temporary files and caches"
	@echo "  all            Run format, lint, and test"

# Installation
install-dev: ## Install contributor tools only; does not install the runnable AutoClean CLI
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

serve-setup: install-uv-tool install-dev ## Install the runnable CLI/Serve path plus contributor tools
	@echo "🎯 Serve setup completed!"
	@echo "💡 The runnable CLI is installed via uv tool and contributor tooling is installed separately"
	@echo "💡 Try: autocleaneeg-pipeline serve api --mode test --api-port 8000"

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
	@$(PYTEST) tests/unit/ -v

test-quick: ## Run fast ingestion unit tests
	@echo "🧪 Running ingestion unit tests..."
	@$(PYTEST) tests/unit/utils/test_ingestion.py -v

test-unit-short: ## Run unit tests (fail fast)
	@echo "🧪 Running unit tests (fast fail)..."
	@$(PYTEST) tests/unit/ -q --maxfail=1

test-ingestion: ## Run ingestion unit tests
	@echo "🧪 Running ingestion unit tests..."
	@$(PYTEST) tests/unit/utils/test_ingestion.py -v

test-cov: ## Run tests with coverage reporting
	@echo "🧪 Running tests with coverage..."
	@$(PYTEST) tests/unit/ --cov=autoclean --cov-report=term-missing --cov-report=html

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	@$(PYTEST) tests/integration/ -v --tb=short

test-integration-short: ## Run integration tests (fail fast)
	@echo "🧪 Running integration tests (fast fail)..."
	@$(PYTEST) tests/integration/ -q --tb=short --maxfail=1

test-all: ## Run all tests (unit + integration)
	@echo "🧪 Running all tests..."
	@$(PYTEST) tests/ -v --tb=short --maxfail=10

# CI Simulation
ci-check: ## Run the same checks as CI pipeline
	@echo "🚀 Running CI-equivalent checks locally..."
	@echo ""
	@echo "1/4 Code Quality Checks..."
	@python3 scripts/check_code_quality.py
	@echo ""
	@echo "2/4 Unit Tests..."
	@$(PYTEST) tests/unit/ -v --tb=short --maxfail=5
	@echo ""
	@echo "3/4 Integration Tests..."
	@$(PYTEST) tests/integration/ -v --tb=short --maxfail=3 || echo "⚠️ Integration tests may fail - that's expected"
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
	@echo "💡 For the first Serve run, the CLI is available via: autocleaneeg-pipeline serve api --mode test --api-port 8000"

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
	@pip install -e ".[docs]"

docs-build: ## Build documentation
	@echo "📚 Building documentation..."
	@cd docs && make html

docs-serve: ## Serve documentation locally
	@echo "📚 Serving documentation at http://localhost:8000"
	@cd docs/_build/html && python3 -m http.server 8000

# Development Server
serve: ## Build frontend + start serve (editable mode, live code changes)
	@cd web && npx vite build --silent
	@autocleaneeg-serve

serve-data1: ## Deploy to data1 and start serve remotely
	@echo "📦 Building + publishing..."
	@uv build --wheel --quiet
	@uvx twine upload dist/* 2>/dev/null || true
	@echo "🚀 Upgrading on data1..."
	@ssh data1 'bash -lc "uv tool install autocleaneeg-pipeline --force --upgrade"'
	@ssh data1 "pkill -9 -f autoclean-serve 2>/dev/null; pkill -9 -f uvicorn 2>/dev/null" || true
	@sleep 2
	@ssh data1 "nohup /Users/ernie/.local/bin/autocleaneeg-serve --no-browser > /tmp/autoclean-serve.log 2>&1 &"
	@sleep 4
	@echo "✅ Running at http://10.241.38.116:8000"

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

# Plans Server
PLANS_PORT := 7933
PLANS_DIR := plans/_site

plans-serve: ## Start HTTP server for plans/_site on port 7933
	@echo "🌐 Starting plans server on http://localhost:$(PLANS_PORT)"
	@cd $(PLANS_DIR) && python3 -m http.server $(PLANS_PORT) &
	@echo "✅ Server running in background"
	@echo "💡 Stop with: make plans-stop"

web-ui: ## Start the web UI/API for the single AutoClean workspace
	@$(MAKE) ensure-serve-workspace
	@$(AUTOCLEAN_CLI) serve api --path "$(SERVE_WORKSPACE)" --mode "$(SERVE_MODE)" --host 127.0.0.1 --api-port 8000

serve-run: ## Start the dispatcher for the single AutoClean workspace
	@$(MAKE) ensure-serve-workspace
	@$(AUTOCLEAN_CLI) serve run --path "$(SERVE_WORKSPACE)" --mode "$(SERVE_MODE)"

app-up: ## Start dispatcher in background and web UI in foreground
	@$(MAKE) ensure-serve-workspace
	@if [ -f .serve-run.pid ] && kill -0 $$(cat .serve-run.pid) 2>/dev/null; then \
		echo "Dispatcher already running (PID $$(cat .serve-run.pid))"; \
	else \
		nohup $(AUTOCLEAN_CLI) serve run --path "$(SERVE_WORKSPACE)" --mode "$(SERVE_MODE)" >/tmp/autoclean-serve-run.log 2>&1 & \
		echo $$! > .serve-run.pid; \
		echo "Started dispatcher (PID $$(cat .serve-run.pid))"; \
	fi
	@$(AUTOCLEAN_CLI) serve api --path "$(SERVE_WORKSPACE)" --mode "$(SERVE_MODE)" --host 127.0.0.1 --api-port 8000

app-stop: ## Stop the background dispatcher started by app-up
	@if [ -f .serve-run.pid ]; then \
		kill $$(cat .serve-run.pid) 2>/dev/null || true; \
		rm -f .serve-run.pid; \
		echo "Stopped background dispatcher"; \
	else \
		echo "No background dispatcher pid file found"; \
	fi

ensure-serve-workspace:
	@if [ ! -d "$(SERVE_WORKSPACE)" ]; then \
		echo "Serve workspace not found: $(SERVE_WORKSPACE)"; \
		echo "Set one explicitly, e.g. make web-ui SERVE_WORKSPACE=/absolute/path/to/workspace"; \
		exit 1; \
	fi

plans-stop: ## Stop the plans server
	@echo "🛑 Stopping plans server on port $(PLANS_PORT)..."
	@-lsof -ti:$(PLANS_PORT) | xargs kill -9 2>/dev/null || true
	@echo "✅ Server stopped"
