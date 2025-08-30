#!/usr/bin/env python3
"""
UV tool management script for AutoClean EEG Pipeline.

This script provides easy management of development tools using uv tool,
including installation, upgrading, and listing of tools.
"""

import argparse
import shutil
import subprocess
import sys
from typing import List, Tuple


def run_uv_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a uv command and return success status and output."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            print(f"   ✅ {description} completed")
        else:
            print(f"   ❌ {description} failed")
            if output.strip():
                print(f"   Output: {output.strip()}")
        
        return success, output
    except FileNotFoundError:
        print(f"   ❌ uv command not found. Please install uv first.")
        return False, "uv not found"


def check_uv_available() -> bool:
    """Check if uv is available."""
    if not shutil.which('uv'):
        print("❌ uv is not installed or not in PATH")
        print("   Please install uv first:")
        print("   • Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   • Windows: powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'")
        print("   • Or with pip: pip install uv")
        return False
    return True


def install_dev_tools():
    """Install all development tools using uv tool."""
    if not check_uv_available():
        return False
    
    tools = [
        "black",
        "isort", 
        "ruff",
        "mypy",
        "pre-commit"
    ]
    
    print("🚀 Installing AutoClean development tools with uv tool")
    print("=" * 60)
    
    failed = []
    for tool in tools:
        success, _ = run_uv_command(["uv", "tool", "install", tool], f"Installing {tool}")
        if not success:
            failed.append(tool)
    
    print("\n" + "=" * 60)
    if failed:
        print(f"❌ Failed to install: {', '.join(failed)}")
        print(f"✅ Successfully installed: {len(tools) - len(failed)}/{len(tools)} tools")
        return False
    else:
        print("🎉 All development tools installed successfully!")
        return True


def upgrade_tools():
    """Upgrade all installed tools."""
    if not check_uv_available():
        return False
    
    print("🔄 Upgrading all uv tools...")
    success, output = run_uv_command(["uv", "tool", "upgrade", "--all"], "Upgrading all tools")
    if output.strip():
        print(f"Output:\n{output}")
    return success


def list_tools():
    """List all installed uv tools."""
    if not check_uv_available():
        return False
    
    print("📋 Installed uv tools:")
    success, output = run_uv_command(["uv", "tool", "list"], "Listing installed tools")
    if success and output.strip():
        print(output)
    return success


def run_tool(tool_name: str, args: List[str]):
    """Run a specific tool with arguments."""
    if not check_uv_available():
        return False
    
    cmd = ["uv", "tool", "run", tool_name] + args
    print(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        # Run with direct output (no capture for interactive tools)
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        print(f"❌ uv command not found")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="UV tool management for AutoClean EEG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/uv_tools.py install                      # Install all dev tools
  python scripts/uv_tools.py list                         # List installed tools
  python scripts/uv_tools.py upgrade                      # Upgrade all tools
  python scripts/uv_tools.py run black --help             # Run black with args
  python scripts/uv_tools.py run ruff check src/          # Run ruff on src/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Install command
    subparsers.add_parser('install', help='Install all development tools')
    
    # List command
    subparsers.add_parser('list', help='List installed tools')
    
    # Upgrade command  
    subparsers.add_parser('upgrade', help='Upgrade all installed tools')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a specific tool')
    run_parser.add_argument('tool', help='Tool name (e.g., black, ruff, mypy)')
    run_parser.add_argument('args', nargs='*', help='Arguments to pass to the tool')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute commands
    if args.command == 'install':
        success = install_dev_tools()
    elif args.command == 'list':
        success = list_tools()
    elif args.command == 'upgrade':
        success = upgrade_tools()
    elif args.command == 'run':
        success = run_tool(args.tool, args.args)
    else:
        parser.print_help()
        return 1
    
    return 0 if success else 1


def install_autoclean_as_tool():
    """Show instructions for installing AutoClean as a uv tool."""
    print("🚀 Installing AutoClean as a uv tool:")
    print("=" * 50)
    print()
    print("# Install from current directory (for development):")
    print("uv tool install .")
    print()
    print("# Install from PyPI (when published):")
    print("uv tool install autocleaneeg")
    print()
    print("# Use AutoClean CLI:")
    print("uv tool run autoclean-eeg --help")
    print("uv tool run autoclean-eeg process --task RestingEyesOpen --file data.raw")
    print()
    print("# List all installed tools:")
    print("uv tool list")


if __name__ == '__main__':
    # Show AutoClean uv tool instructions if no args provided
    if len(sys.argv) == 1:
        install_autoclean_as_tool()
        print()
        print("=" * 50)
        print("For development tool management, use:")
        print("python scripts/uv_tools.py --help")
        sys.exit(0)
    
    sys.exit(main())