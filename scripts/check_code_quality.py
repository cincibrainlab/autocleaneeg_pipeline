#!/usr/bin/env python3
"""
Local code quality checker for AutoClean EEG Pipeline using uv tool.

This script runs the same code quality checks that are performed in CI,
allowing developers to fix issues locally before committing. Uses uv tool run
for isolated tool execution without installation conflicts.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class CodeQualityChecker:
    """Run code quality checks locally using uv tool."""

    def __init__(self, src_dir: Path = None, fix: bool = False, verbose: bool = True, use_uv: bool = None):
        """Initialize code quality checker."""
        self.src_dir = src_dir or Path("src/autoclean")
        self.fix = fix
        self.verbose = verbose
        self.results = []
        
        # Auto-detect uv if not specified
        if use_uv is None:
            self.use_uv = shutil.which('uv') is not None
        else:
            self.use_uv = use_uv
            
        if self.verbose and self.use_uv:
            print("📦 Using uv tool for isolated command execution")

    def run_command(
        self, cmd: List[str], description: str, can_fix: bool = False
    ) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        if self.verbose:
            print(f"\n🔍 {description}...")
            if can_fix and self.fix:
                print("   (Running with --fix to automatically correct issues)")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=Path.cwd()
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            if self.verbose:
                if success:
                    print(f"   ✅ {description} passed")
                else:
                    print(f"   ❌ {description} failed")
                    if output.strip():
                        print(f"   Output:\n{output}")

            return success, output

        except FileNotFoundError:
            if self.use_uv:
                error_msg = f"Command not found: {cmd[0]}. Please install it with: uv tool install {cmd[0]}"
            else:
                error_msg = f"Command not found: {cmd[0]}. Please install it with: pip install {cmd[0]}"
            if self.verbose:
                print(f"   ❌ {error_msg}")
            return False, error_msg

    def check_black(self) -> Tuple[bool, str]:
        """Check code formatting with Black."""
        if self.use_uv:
            cmd = ["uv", "tool", "run", "black", "--check", "--diff", str(self.src_dir)]
            if self.fix:
                cmd = ["uv", "tool", "run", "black", str(self.src_dir)]
        else:
            cmd = ["black", "--check", "--diff", str(self.src_dir)]
            if self.fix:
                cmd = ["black", str(self.src_dir)]

        return self.run_command(cmd, "Black code formatting", can_fix=True)

    def check_isort(self) -> Tuple[bool, str]:
        """Check import sorting with isort."""
        if self.use_uv:
            cmd = ["uv", "tool", "run", "isort", "--check-only", "--diff", str(self.src_dir)]
            if self.fix:
                cmd = ["uv", "tool", "run", "isort", str(self.src_dir)]
        else:
            cmd = ["isort", "--check-only", "--diff", str(self.src_dir)]
            if self.fix:
                cmd = ["isort", str(self.src_dir)]

        return self.run_command(cmd, "isort import sorting", can_fix=True)

    def check_ruff(self) -> Tuple[bool, str]:
        """Check code with Ruff linter."""
        if self.use_uv:
            cmd = ["uv", "tool", "run", "ruff", "check", str(self.src_dir)]
            if self.fix:
                cmd = ["uv", "tool", "run", "ruff", "check", "--fix", str(self.src_dir)]
        else:
            cmd = ["ruff", "check", str(self.src_dir)]
            if self.fix:
                cmd = ["ruff", "check", "--fix", str(self.src_dir)]

        return self.run_command(cmd, "Ruff linting", can_fix=True)

    def check_mypy(self) -> Tuple[bool, str]:
        """Check types with mypy."""
        if self.use_uv:
            cmd = ["uv", "tool", "run", "mypy", str(self.src_dir), "--ignore-missing-imports"]
        else:
            cmd = ["mypy", str(self.src_dir), "--ignore-missing-imports"]

        return self.run_command(cmd, "mypy type checking", can_fix=False)

    def run_all_checks(self) -> bool:
        """Run all code quality checks."""
        checks = [
            ("Black Formatting", self.check_black),
            ("Import Sorting", self.check_isort),
            ("Ruff Linting", self.check_ruff),
            # ("Type Checking", self.check_mypy),  # Temporarily disabled
        ]

        if self.verbose:
            print("🚀 Running Local Code Quality Checks")
            print("=" * 50)
            print(f"Source directory: {self.src_dir}")
            print(f"Fix mode: {'ON' if self.fix else 'OFF'}")
            print(f"Tool execution: {'uv tool run' if self.use_uv else 'direct command'}")

        all_passed = True

        for check_name, check_func in checks:
            success, output = check_func()
            self.results.append((check_name, success, output))

            if not success:
                all_passed = False

        # Print summary
        if self.verbose:
            self.print_summary()

        return all_passed

    def print_summary(self):
        """Print summary of all checks."""
        print("\n" + "=" * 50)
        print("📋 CODE QUALITY SUMMARY")
        print("=" * 50)

        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)

        for check_name, success, _ in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {check_name}")

        print(f"\nOverall: {passed}/{total} checks passed")

        if passed == total:
            print("🎉 All code quality checks passed! Ready to commit.")
        else:
            print("⚠️  Some checks failed. Run with --fix to auto-correct issues.")
            tip_msg = "💡 Tip: Use 'python scripts/check_code_quality.py --fix' to automatically fix most issues"
            if self.use_uv:
                tip_msg += "\n💡 Using uv tool run for isolated execution (no dependency conflicts!)"
            print(tip_msg)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run local code quality checks for AutoClean EEG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_code_quality.py                    # Check code quality (with uv tool)
  python scripts/check_code_quality.py --no-uv            # Use direct commands (fallback)
  python scripts/check_code_quality.py --fix              # Fix issues automatically
  python scripts/check_code_quality.py --quiet            # Run silently
  python scripts/check_code_quality.py --src tests/       # Check tests directory
        """,
    )

    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix issues where possible"
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("src/autoclean"),
        help="Source directory to check (default: src/autoclean)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--no-uv", action="store_true", help="Use direct commands instead of 'uv tool run'"
    )
    parser.add_argument(
        "--check",
        choices=["black", "isort", "ruff", "mypy", "all"],
        default="all",
        help="Run specific check only",
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.src.exists():
        print(f"❌ Source directory does not exist: {args.src}")
        return 1

    # Initialize checker
    checker = CodeQualityChecker(
        src_dir=args.src, 
        fix=args.fix, 
        verbose=not args.quiet,
        use_uv=not args.no_uv
    )

    # Run checks
    if args.check == "all":
        success = checker.run_all_checks()
    else:
        # Run single check
        check_methods = {
            "black": checker.check_black,
            "isort": checker.check_isort,
            "ruff": checker.check_ruff,
            "mypy": checker.check_mypy,
        }

        success, output = check_methods[args.check]()
        if not args.quiet and not success:
            print(output)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
