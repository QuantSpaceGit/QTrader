# Project Configuration
# ---------------------
PROJECT_NAME := QTrader
PYTHON_VERSION := 3.13
VENV := .venv
BIN := $(VENV)/bin
SRC_DIR := src

# Terminal Colors
# ---------------
CYAN := \033[0;36m
GREEN := \033[0;32m
RED := \033[0;31m
BLUE := \033[0;34m
YELLOW := \033[1;33m
BOLD := \033[1m
END := \033[0m

# Default target
# --------------
.DEFAULT_GOAL := help

# Utility Functions
# -----------------
define log_info
echo "$(BLUE)ℹ️  $(1)$(END)"
endef

define log_success
echo "$(GREEN)✅ $(1)$(END)"
endef

define log_warning
echo "$(YELLOW)⚠️  $(1)$(END)"
endef

define log_error
echo "$(RED)❌ $(1)$(END)"
endef


################################################################################
# HELP
################################################################################
.PHONY: help
help: ## 📚 Show this help message
	@echo "$(BOLD)$(PROJECT_NAME) Development Makefile$(END)"
	@echo ""
	@echo "$(CYAN)📋 Available Commands:$(END)"
	@echo ""
	@echo "$(BOLD)🚀 Setup & Environment:$(END)"
	@grep -E '^(check-uv|sync|upgrade|install-hooks|setup|clean):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(END) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)🎨 Code Quality:$(END)"
	@grep -E '^(format|format-md|lint|lint-check|type-check|quality):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(END) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)🧪 Testing:$(END)"
	@grep -E '^(test[a-zA-Z_-]*|qa):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(END) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)📦 Build & Release:$(END)"
	@grep -E '^(build|release-check|version|release-prepare|release):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(END) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)📓 Development Tools:$(END)"
	@grep -E '^(setup-kernel|run-jupyter):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(END) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)🔧 Utilities:$(END)"
	@grep -E '^(help):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(END) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)💡 Quick Start:$(END)"
	@echo "  $(CYAN)make setup$(END)          - Complete development environment setup"
	@echo "  $(CYAN)make qa$(END)             - Run full quality assurance (format + lint + test)"
	@echo "  $(CYAN)make test$(END)           - Run all tests with coverage"
	@echo ""
	@echo "$(YELLOW)🚀 Release Workflow:$(END)"
	@echo "  $(CYAN)make version$(END)        - Show current version"
	@echo "  $(CYAN)make release-prepare$(END) - Check if ready for release (runs QA)"
	@echo "  $(CYAN)make release VERSION=x.y.z$(END) - Create and push GitHub release tag"
	@echo ""


################################################################################
# PROJECT SETUP
################################################################################
.PHONY: check-uv
check-uv: ## 🔧 Verify UV package manager is available
	@echo "$(BLUE)ℹ️  Checking UV package manager...$(END)"
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)❌ UV is not installed$(END)"; \
		echo "$(RED)Please install UV from: https://docs.astral.sh/uv/getting-started/installation/$(END)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ UV package manager is available$(END)"

.PHONY: sync
sync: check-uv ## 📦 Sync dependencies and create virtual environment
	@echo "$(BLUE)ℹ️  Syncing dependencies with UV...$(END)"
	@uv sync --all-packages --all-groups || { \
		echo "$(RED)❌ Failed to sync packages$(END)"; \
		exit 1; \
	}
	@echo "$(BLUE)ℹ️  Installing qtrader in editable mode...$(END)"
	@uv pip install -e . --quiet || { \
		echo "$(RED)❌ Failed to install package$(END)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Dependencies synced successfully$(END)"

.PHONY: upgrade
upgrade: check-uv ## 🔄 Upgrade all packages to latest versions
	@echo "$(BLUE)ℹ️  Upgrading all packages with UV...$(END)"
	@uv lock --upgrade || { \
		echo "$(RED)❌ Failed to upgrade packages$(END)"; \
		exit 1; \
	}
	@echo "$(BLUE)ℹ️  Syncing upgraded dependencies...$(END)"
	@uv sync --all-packages --all-groups || { \
		echo "$(RED)❌ Failed to sync upgraded packages$(END)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ All packages upgraded and synced successfully$(END)"

.PHONY: install-hooks
install-hooks: sync ## 🪝 Install pre-commit hooks
	@echo "$(BLUE)ℹ️  Installing pre-commit hooks...$(END)"
	@uv run pre-commit install || { \
		echo "$(RED)❌ Failed to install pre-commit hooks$(END)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Pre-commit hooks installed$(END)"

.PHONY: pre-commit
pre-commit: sync ## 🔍 Run pre-commit hooks manually
	@echo "$(BLUE)ℹ️  Running pre-commit hooks...$(END)"
	@uv run pre-commit run --all-files || { \
		echo "$(RED)❌ Pre-commit hooks failed$(END)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Pre-commit hooks passed$(END)"

.PHONY: setup
setup: sync install-hooks ## 🚀 Complete development environment setup
	@echo "$(GREEN)✅ Development environment setup complete!$(END)"
	@echo "$(BLUE)💡 Use 'uv run <command>' to run commands in the environment$(END)"
	@echo "$(BLUE)💡 Example: uv run python $(SRC_DIR)/main.py$(END)"

.PHONY: clean
clean: ## 🧹 Clean workspace (remove cache, temp files, and scaffolded project files)
	@echo "$(BLUE)ℹ️  Cleaning development environment...$(END)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ .mypy_cache/
	@rm -f .coverage coverage.xml
	@rm -rf htmlcov/ mypy-report/ .coverage.*
	@echo "$(BLUE)ℹ️  Removing scaffolded project files...$(END)"
	@rm -rf config/ library/ data/ output/ logs/ examples/ experiments/ QTRADER_README.md
	@echo "$(GREEN)✅ Workspace cleaned$(END)"


################################################################################
# CODE QUALITY
################################################################################

.PHONY: format
format: sync ## 🎨 Format code with ruff, isort, and markdown (matches pre-commit)
	@echo "$(BLUE)ℹ️  Formatting Python code with ruff (fix + format)...$(END)"
	@uv run ruff check --fix --target-version py313 $(SRC_DIR)/
	@uv run ruff format --target-version py313 $(SRC_DIR)/
	@echo "$(BLUE)ℹ️  Formatting imports with isort...$(END)"
	@uv run isort $(SRC_DIR)/
	@echo "$(BLUE)ℹ️  Formatting Markdown files...$(END)"
	@uv run mdformat . --wrap=no --end-of-line=lf || echo "$(YELLOW)⚠️  mdformat not installed, run 'uv add --dev mdformat mdformat-gfm mdformat-tables'$(END)"
	@echo "$(GREEN)✅ Code and markdown formatting completed$(END)"

.PHONY: lint
lint: sync ## 🔍 Lint code and fix auto-fixable issues (matches pre-commit)
	@echo "$(BLUE)ℹ️  Linting code...$(END)"
	@uv run ruff check --fix --target-version py313 $(SRC_DIR)/
	@echo "$(GREEN)✅ Code linting completed$(END)"

.PHONY: lint-check
lint-check: sync ## 📋 Check code without making changes (matches pre-commit)
	@echo "$(BLUE)ℹ️  Checking code quality...$(END)"
	@uv run ruff check --target-version py313 $(SRC_DIR)/
	@uv run ruff format --target-version py313 --check $(SRC_DIR)/
	@uv run isort --check-only $(SRC_DIR)/
	@echo "$(GREEN)✅ Code quality check passed$(END)"

.PHONY: format-md
format-md: sync ## 📝 Format Markdown files only
	@echo "$(BLUE)ℹ️  Formatting Markdown files...$(END)"
	@uv run mdformat . --wrap=no --end-of-line=lf || echo "$(YELLOW)⚠️  mdformat not installed, run 'uv add --dev mdformat mdformat-gfm mdformat-tables'$(END)"
	@echo "$(GREEN)✅ Markdown formatting completed$(END)"

.PHONY: type-check
type-check: sync ## 🔬 Run type checking with MyPy
	@echo "$(BLUE)ℹ️  Running type checks with MyPy...$(END)"
	@uv run mypy $(SRC_DIR)/ || { \
		echo "$(RED)❌ Type checking failed$(END)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Type checking completed$(END)"

.PHONY: quality
quality: format lint-check type-check ## 🏆 Run all code quality checks
	@echo "$(GREEN)✅ All code quality checks passed$(END)"

.PHONY: qa
qa: quality test ## 🔍 Full quality assurance (code quality + tests)
	@echo "$(GREEN)✅ Quality assurance complete - ready for production!$(END)"


################################################################################
# BUILD & RELEASE
################################################################################

.PHONY: build
build: clean qa ## 📦 Build package (clean + qa + uv build)
	@echo "$(BLUE)ℹ️  Building package with uv...$(END)"
	@uv build
	@echo "$(GREEN)✅ Package built successfully$(END)"
	@echo "$(CYAN)📦 Distribution files:$(END)"
	@ls -lh dist/

.PHONY: version
version: ## 📋 Show current version from pyproject.toml
	@grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'

.PHONY: release-prepare
release-prepare: qa ## 🚀 Prepare release (run QA, show version, prompt for confirmation)
	@echo "$(CYAN)════════════════════════════════════════════════════════════════$(END)"
	@echo "$(BOLD)📦 Release Preparation$(END)"
	@echo "$(CYAN)════════════════════════════════════════════════════════════════$(END)"
	@echo ""
	@echo "$(BLUE)Current version:$(END) $$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo ""
	@echo "$(YELLOW)⚠️  Before releasing:$(END)"
	@echo "  1. Update version in pyproject.toml if needed"
	@echo "  2. Update CHANGELOG.md with release notes"
	@echo "  3. Commit all changes: $(CYAN)git add -A && git commit -m 'chore: prepare release vX.Y.Z'$(END)"
	@echo "  4. Run: $(CYAN)make release VERSION=X.Y.Z$(END)"
	@echo ""
	@echo "$(GREEN)✅ QA checks passed - ready for release$(END)"

.PHONY: release
release: ## 🚀 Create GitHub release (usage: make release VERSION=x.y.z)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)❌ VERSION not specified$(END)"; \
		echo "$(YELLOW)Usage: make release VERSION=x.y.z$(END)"; \
		echo "$(YELLOW)Example: make release VERSION=0.2.0$(END)"; \
		exit 1; \
	fi
	@echo "$(CYAN)════════════════════════════════════════════════════════════════$(END)"
	@echo "$(BOLD)🚀 Creating GitHub Release v$(VERSION)$(END)"
	@echo "$(CYAN)════════════════════════════════════════════════════════════════$(END)"
	@echo ""
	@CURRENT_VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	if [ "$$CURRENT_VERSION" != "$(VERSION)" ]; then \
		echo "$(RED)❌ Version mismatch!$(END)"; \
		echo "$(YELLOW)pyproject.toml has: $$CURRENT_VERSION$(END)"; \
		echo "$(YELLOW)You specified: $(VERSION)$(END)"; \
		echo "$(YELLOW)Update pyproject.toml first or use VERSION=$$CURRENT_VERSION$(END)"; \
		exit 1; \
	fi
	@echo "$(BLUE)ℹ️  Checking git status...$(END)"
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "$(RED)❌ Working directory is not clean$(END)"; \
		echo "$(YELLOW)Commit or stash changes before releasing$(END)"; \
		git status --short; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Working directory is clean$(END)"
	@echo ""
	@echo "$(BLUE)ℹ️  Checking if tag v$(VERSION) already exists...$(END)"
	@if git rev-parse "v$(VERSION)" >/dev/null 2>&1; then \
		echo "$(RED)❌ Tag v$(VERSION) already exists$(END)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Tag is available$(END)"
	@echo ""
	@echo "$(BLUE)ℹ️  Building release artifacts...$(END)"
	@$(MAKE) build
	@echo ""
	@echo "$(BLUE)ℹ️  Creating git tag v$(VERSION)...$(END)"
	@git tag -a "v$(VERSION)" -m "Release version $(VERSION)"
	@echo "$(GREEN)✅ Tag created$(END)"
	@echo ""
	@echo "$(BLUE)ℹ️  Pushing tag to GitHub...$(END)"
	@git push origin "v$(VERSION)"
	@echo "$(GREEN)✅ Tag pushed to GitHub$(END)"
	@echo ""
	@echo "$(CYAN)════════════════════════════════════════════════════════════════$(END)"
	@echo "$(GREEN)✅ Release v$(VERSION) created successfully!$(END)"
	@echo "$(CYAN)════════════════════════════════════════════════════════════════$(END)"
	@echo ""
	@echo "$(YELLOW)📋 Next steps:$(END)"
	@echo "  1. Go to: https://github.com/QuantSpaceGit/QTrader/releases/new?tag=v$(VERSION)"
	@echo "  2. GitHub will auto-detect the tag"
	@echo "  3. Add release notes from CHANGELOG.md"
	@echo "  4. Attach files from dist/ directory:"
	@echo "     - dist/qtrader-$(VERSION)-py3-none-any.whl"
	@echo "     - dist/qtrader-$(VERSION).tar.gz"
	@echo "  5. Click 'Publish release'"
	@echo ""
	@echo "$(BLUE)💡 Or use GitHub CLI if installed:$(END)"
	@echo "  $(CYAN)gh release create v$(VERSION) dist/* --title 'Release v$(VERSION)' --notes-file CHANGELOG.md$(END)"
	@echo ""


################################################################################
# TESTING
################################################################################

.PHONY: test
test: sync ## 🧪 Run all tests with coverage
	@echo "$(BLUE)ℹ️  Running all tests with coverage...$(END)"
	@uv run pytest --cov --cov-report=term-missing --cov-report=html || { \
		echo "$(RED)❌ Tests failed$(END)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ All tests passed$(END)"

.PHONY: test-fast
test-fast: sync ## ⚡ Run tests without coverage (faster)
	@echo "$(BLUE)ℹ️  Running tests (fast mode)...$(END)"
	@uv run pytest -v || { \
		echo "$(RED)❌ Tests failed$(END)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ All tests passed$(END)"
