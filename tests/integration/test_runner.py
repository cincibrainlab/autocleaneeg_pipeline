"""
Integration test runner and utilities.

This module provides utilities for running integration tests systematically
and collecting results for CI/CD pipeline validation.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict



class IntegrationTestRunner:
    """Runner for comprehensive integration testing."""

    def __init__(self, test_dir: Path = None):
        """Initialize test runner."""
        if test_dir is None:
            test_dir = Path(__file__).parent
        self.test_dir = test_dir
        self.results = {}

    def run_test_suite(
        self, test_pattern: str = "test_*.py", verbose: bool = True
    ) -> Dict:
        """Run complete integration test suite."""
        start_time = time.time()

        # Find all integration test files
        test_files = list(self.test_dir.glob(test_pattern))
        test_files = [f for f in test_files if f.name != __file__.name]  # Exclude self

        if not test_files:
            return {"error": "No integration test files found"}

        # Run each test file
        results = {
            "test_files": [],
            "summary": {
                "total_files": len(test_files),
                "passed_files": 0,
                "failed_files": 0,
                "skipped_files": 0,
                "total_time": 0,
            },
        }

        for test_file in test_files:
            if verbose:
                print(f"\n🧪 Running integration tests in {test_file.name}")

            file_result = self._run_single_test_file(test_file, verbose)
            results["test_files"].append(file_result)

            # Update summary
            if file_result["status"] == "passed":
                results["summary"]["passed_files"] += 1
            elif file_result["status"] == "failed":
                results["summary"]["failed_files"] += 1
            else:
                results["summary"]["skipped_files"] += 1

        results["summary"]["total_time"] = time.time() - start_time

        # Print summary
        if verbose:
            self._print_summary(results)

        return results

    def _run_single_test_file(self, test_file: Path, verbose: bool = True) -> Dict:
        """Run a single test file and collect results."""
        start_time = time.time()

        # Run pytest on the specific file
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--maxfail=10",
            "-x",  # Stop on first failure for integration tests
            "--json-report",
            f"--json-report-file={test_file.stem}_report.json",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_dir.parent.parent,  # Run from project root
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per file
            )

            # Parse JSON report if available
            report_file = self.test_dir / f"{test_file.stem}_report.json"
            test_details = []

            if report_file.exists():
                try:
                    with open(report_file, "r") as f:
                        json_report = json.load(f)

                    # Extract test details
                    for test in json_report.get("tests", []):
                        test_details.append(
                            {
                                "name": test.get("nodeid", "unknown"),
                                "outcome": test.get("outcome", "unknown"),
                                "duration": test.get("duration", 0),
                            }
                        )

                    # Clean up report file
                    report_file.unlink()

                except (json.JSONDecodeError, KeyError):
                    pass

            return {
                "file": test_file.name,
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": time.time() - start_time,
                "return_code": result.returncode,
                "stdout": (
                    result.stdout[-1000:] if result.stdout else ""
                ),  # Last 1000 chars
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "test_details": test_details,
            }

        except subprocess.TimeoutExpired:
            return {
                "file": test_file.name,
                "status": "timeout",
                "duration": time.time() - start_time,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test execution timed out",
                "test_details": [],
            }
        except Exception as e:
            return {
                "file": test_file.name,
                "status": "error",
                "duration": time.time() - start_time,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "test_details": [],
            }

    def _print_summary(self, results: Dict):
        """Print test results summary."""
        summary = results["summary"]

        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)

        print(f"Total test files: {summary['total_files']}")
        print(f"✅ Passed: {summary['passed_files']}")
        print(f"❌ Failed: {summary['failed_files']}")
        print(f"⏭️  Skipped: {summary['skipped_files']}")
        print(f"⏱️  Total time: {summary['total_time']:.1f}s")

        # Details for failed files
        failed_files = [f for f in results["test_files"] if f["status"] == "failed"]
        if failed_files:
            print("\nFAILED FILES:")
            for file_result in failed_files:
                print(f"  - {file_result['file']}: {file_result['stderr'][:200]}...")

        print("\nINDIVIDUAL FILE RESULTS:")
        for file_result in results["test_files"]:
            status_emoji = {
                "passed": "✅",
                "failed": "❌",
                "timeout": "⏱️",
                "error": "💥",
            }
            emoji = status_emoji.get(file_result["status"], "❓")
            print(f"  {emoji} {file_result['file']} ({file_result['duration']:.1f}s)")


def run_quick_integration_tests():
    """Run a quick subset of integration tests for CI."""
    runner = IntegrationTestRunner()

    # Run only fast integration tests
    quick_tests = ["test_pipeline_workflows.py", "test_output_validation.py"]

    results = {"files": [], "summary": {"passed": 0, "failed": 0}}

    for test_name in quick_tests:
        test_file = runner.test_dir / test_name
        if test_file.exists():
            result = runner._run_single_test_file(test_file, verbose=False)
            results["files"].append(result)

            if result["status"] == "passed":
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1

    return results


def main():
    """Main entry point for running integration tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoClean integration tests")
    parser.add_argument(
        "--quick", action="store_true", help="Run only quick integration tests"
    )
    parser.add_argument(
        "--pattern", default="test_*.py", help="Test file pattern to match"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    if args.quick:
        results = run_quick_integration_tests()
        passed = results["summary"]["passed"]
        failed = results["summary"]["failed"]
        print(f"Quick integration tests: {passed} passed, {failed} failed")
        return 0 if failed == 0 else 1
    else:
        runner = IntegrationTestRunner()
        results = runner.run_test_suite(
            test_pattern=args.pattern, verbose=not args.quiet
        )

        failed = results["summary"]["failed_files"]
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
