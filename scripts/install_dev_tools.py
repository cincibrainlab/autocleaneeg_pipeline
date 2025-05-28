#!/usr/bin/env python3
"""
Install development tools for AutoClean EEG Pipeline.

This script installs all the code quality tools needed for local development,
matching the exact versions and configurations used in CI.
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, check=True, shell=True)
        print(f"   ✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        return False


def main():
    """Install development tools."""
    print("🚀 Installing AutoClean Development Tools")
    print("=" * 50)

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        print("⚠️  Warning: Not in a virtual environment")
        print("   Consider activating a virtual environment first")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return 1

    # Core development tools
    tools = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install black", "Installing Black (code formatter)"),
        ("pip install isort", "Installing isort (import sorter)"),
        ("pip install ruff", "Installing Ruff (fast linter)"),
        ("pip install mypy", "Installing mypy (type checker)"),
        ("pip install pre-commit", "Installing pre-commit (optional git hooks)"),
    ]

    # Testing tools
    test_tools = [
        ("pip install pytest", "Installing pytest (test runner)"),
        ("pip install pytest-cov", "Installing pytest-cov (coverage)"),
        ("pip install pytest-benchmark", "Installing pytest-benchmark (performance)"),
        ("pip install pytest-mock", "Installing pytest-mock (mocking)"),
        ("pip install pytest-xdist", "Installing pytest-xdist (parallel testing)"),
    ]

    # Additional development tools
    dev_tools = [
        ("pip install jupyterlab", "Installing JupyterLab (optional)"),
        ("pip install ipython", "Installing IPython (better REPL)"),
        ("pip install memory-profiler", "Installing memory-profiler (performance)"),
        ("pip install psutil", "Installing psutil (system monitoring)"),
    ]

    all_tools = tools + test_tools + dev_tools
    failed_tools = []

    for cmd, description in all_tools:
        if not run_command(cmd, description):
            failed_tools.append(description)

    print("\n" + "=" * 50)
    print("📋 INSTALLATION SUMMARY")
    print("=" * 50)

    if failed_tools:
        print(f"❌ {len(failed_tools)} tools failed to install:")
        for tool in failed_tools:
            print(f"   - {tool}")
        print(f"✅ {len(all_tools) - len(failed_tools)} tools installed successfully")
    else:
        print("🎉 All development tools installed successfully!")

    print("\n💡 Next steps:")
    print("   1. Run: python scripts/check_code_quality.py")
    print("   2. (Optional) Set up pre-commit: pre-commit install")
    print("   3. Start developing with confidence!")

    return 0 if not failed_tools else 1


if __name__ == "__main__":
    sys.exit(main())
