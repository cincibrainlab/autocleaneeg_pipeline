#!/usr/bin/env python3
"""
Install development tools for AutoClean EEG Pipeline using uv tool.

This script installs all the code quality tools needed for local development
using uv tool for isolated tool management. Each tool runs in its own environment
to prevent dependency conflicts.
"""

import shutil
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


def check_uv_available():
    """Check if uv is available, suggest installation if not."""
    if not shutil.which('uv'):
        print("❌ uv is not installed or not in PATH")
        print("   Please install uv first:")
        print("   • Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   • Windows: powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'")
        print("   • Or with pip: pip install uv")
        return False
    return True

def main():
    """Install development tools using uv tool."""
    print("🚀 Installing AutoClean Contributor Tools with uv tool")
    print("=" * 50)
    
    # Check if uv is available
    if not check_uv_available():
        return 1
    
    print("✅ uv detected - using isolated tool environments")
    print("   No virtual environment conflicts possible!")
    print("ℹ️ This script installs contributor tooling only.")
    print("   It does NOT install the runnable AutoClean CLI/Serve environment.")
    print("   For the CLI and Serve commands, run: make install-uv-tool")

    # Core development tools using uv tool (isolated environments)
    tools = [
        ("uv tool install black", "Installing Black (code formatter)"),
        ("uv tool install isort", "Installing isort (import sorter)"),
        ("uv tool install ruff", "Installing Ruff (fast linter)"),
        ("uv tool install mypy", "Installing mypy (type checker)"),
        ("uv tool install pre-commit", "Installing pre-commit (git hooks)"),
    ]

    # Testing tools (installed in project environment since they're project dependencies)
    test_tools = [
        ("uv pip install pytest", "Installing pytest (test runner)"),
        ("uv pip install pytest-cov", "Installing pytest-cov (coverage)"),
        ("uv pip install pytest-benchmark", "Installing pytest-benchmark (performance)"),
        ("uv pip install pytest-mock", "Installing pytest-mock (mocking)"),
        ("uv pip install pytest-xdist", "Installing pytest-xdist (parallel testing)"),
    ]

    # Additional development tools using uv tool
    dev_tools = [
        ("uv tool install jupyterlab", "Installing JupyterLab (notebook environment)"),
        ("uv tool install ipython", "Installing IPython (better REPL)"),
        ("uv tool install memory-profiler", "Installing memory-profiler (performance)"),
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
        print("🎉 All contributor tools installed successfully!")

    print("\n💡 Next steps:")
    print("   1. Install the runnable CLI if needed: make install-uv-tool")
    print("   2. Run: python scripts/check_code_quality.py")
    print("   3. (Optional) Set up pre-commit: uv tool run pre-commit install")
    print("   4. List installed tools: uv tool list")
    print("   5. Start developing with confidence!")
    print("\n🎯 Benefit: All tools are isolated - no dependency conflicts!")

    return 0 if not failed_tools else 1


if __name__ == "__main__":
    sys.exit(main())
