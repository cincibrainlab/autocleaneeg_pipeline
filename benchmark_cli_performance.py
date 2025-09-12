#!/usr/bin/env python3
"""
Benchmark script to measure CLI startup performance.

This script tests various CLI commands to establish baseline performance
before and after optimizations.
"""

import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import List, Tuple


def run_command_timed(command: List[str], runs: int = 5) -> Tuple[float, float, List[float]]:
    """Run a command multiple times and return timing statistics.
    
    Args:
        command: Command to run as list of strings
        runs: Number of times to run the command
        
    Returns:
        Tuple of (mean_time, std_dev, all_times)
    """
    times = []
    
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"{elapsed:.3f}s")
            
            # Check if command failed
            if result.returncode != 0:
                print(f"    Warning: Command returned {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            print("TIMEOUT (30s)")
            times.append(30.0)  # Use timeout value as worst case
        except Exception as e:
            print(f"ERROR: {e}")
            times.append(30.0)  # Use timeout value as worst case
    
    return mean(times), stdev(times) if len(times) > 1 else 0.0, times


def find_autoclean_command() -> str:
    """Find the correct autoclean command to use."""
    # Try different possible command names/paths
    candidates = [
        "autoclean-eeg",
        "autocleaneeg-pipeline", 
        "python -m autoclean.cli",
        str(Path.cwd() / ".venv" / "Scripts" / "autoclean-eeg.exe"),
        str(Path.cwd() / ".venv" / "bin" / "autoclean-eeg"),
    ]
    
    for cmd in candidates:
        try:
            if cmd.startswith("python"):
                test_cmd = cmd.split() + ["--help"]
            else:
                test_cmd = [cmd, "--help"]
                
            result = subprocess.run(
                test_cmd, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            # Accept both success (0) and help exit codes (some CLIs exit with 1 for help)
            if result.returncode in [0, 1, 2] and "autoclean" in result.stdout.lower():
                print(f"Found working command: {cmd}")
                return cmd
        except Exception:
            continue
    
    raise RuntimeError("Could not find a working autoclean command")


def main():
    """Run CLI performance benchmarks."""
    print("AutoClean CLI Performance Benchmark")
    print("=" * 50)
    
    # Find the correct command
    try:
        base_cmd = find_autoclean_command()
        if base_cmd.startswith("python"):
            cmd_parts = base_cmd.split()
        else:
            cmd_parts = [base_cmd]
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please ensure autoclean is installed and available in PATH")
        return 1
    
    # Test cases: (name, command_args, description)
    test_cases = [
        ("help", ["--help"], "Root help screen"),
        ("no_command", [], "Main dashboard (no arguments)"),
        ("task_help", ["task", "--help"], "Task help screen"),
        ("workspace_help", ["workspace", "--help"], "Workspace help screen"),
        ("input_help", ["input", "--help"], "Input help screen"),
        ("unknown_command", ["nonexistent"], "Unknown command error"),
        ("task_no_subcommand", ["task"], "Task command with no subcommand"),
        ("workspace_no_subcommand", ["workspace"], "Workspace command with no subcommand"),
        ("input_no_subcommand", ["input"], "Input command with no subcommand"),
    ]
    
    results = {}
    total_start = time.perf_counter()
    
    print(f"\nUsing command: {' '.join(cmd_parts)}")
    print(f"Running {len(test_cases)} test cases with 5 runs each...\n")
    
    for name, args, description in test_cases:
        print(f"Testing: {name} - {description}")
        command = cmd_parts + args
        
        try:
            mean_time, std_dev, all_times = run_command_timed(command, runs=5)
            results[name] = {
                'mean': mean_time,
                'std_dev': std_dev,
                'times': all_times,
                'description': description
            }
            print(f"  Result: {mean_time:.3f}s ± {std_dev:.3f}s\n")
        except Exception as e:
            print(f"  Failed: {e}\n")
            results[name] = {
                'mean': 999.0,
                'std_dev': 0.0,
                'times': [],
                'description': description,
                'error': str(e)
            }
    
    total_time = time.perf_counter() - total_start
    
    # Print summary
    print("=" * 50)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Test Case':<25} {'Mean Time':<12} {'Std Dev':<10} {'Description'}")
    print("-" * 80)
    
    for name, data in results.items():
        if 'error' in data:
            print(f"{name:<25} {'ERROR':<12} {'':<10} {data['description']}")
        else:
            print(f"{name:<25} {data['mean']:.3f}s{'':<6} ±{data['std_dev']:.3f}s{'':<3} {data['description']}")
    
    print("-" * 80)
    print(f"Total benchmark time: {total_time:.1f}s")
    
    # Identify slowest operations
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        slowest = max(valid_results.items(), key=lambda x: x[1]['mean'])
        fastest = min(valid_results.items(), key=lambda x: x[1]['mean'])
        
        print(f"\nSlowest operation: {slowest[0]} ({slowest[1]['mean']:.3f}s)")
        print(f"Fastest operation: {fastest[0]} ({fastest[1]['mean']:.3f}s)")
        print(f"Speed difference: {slowest[1]['mean'] / fastest[1]['mean']:.1f}x")
    
    # Save detailed results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("AutoClean CLI Performance Benchmark Results\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(cmd_parts)}\n")
        f.write("=" * 50 + "\n\n")
        
        for name, data in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Description: {data['description']}\n")
            if 'error' in data:
                f.write(f"  Error: {data['error']}\n")
            else:
                f.write(f"  Mean time: {data['mean']:.3f}s\n")
                f.write(f"  Std dev: {data['std_dev']:.3f}s\n")
                f.write(f"  All times: {[f'{t:.3f}' for t in data['times']]}\n")
            f.write("\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
