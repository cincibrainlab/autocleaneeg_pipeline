#!/usr/bin/env python3
"""
Simple benchmark script to measure CLI startup performance.
"""

import subprocess
import time
from statistics import mean


def time_command(command_args, runs=3):
    """Time a command multiple times and return average."""
    times = []
    cmd = ["autoclean-eeg"] + command_args
    
    print(f"Testing: autoclean-eeg {' '.join(command_args)}")
    
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        
        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"{elapsed:.3f}s")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            times.append(15.0)
        except Exception as e:
            print(f"ERROR: {e}")
            times.append(15.0)
    
    avg_time = mean(times)
    print(f"  Average: {avg_time:.3f}s\n")
    return avg_time


def main():
    """Run performance benchmarks."""
    print("AutoClean CLI Performance Benchmark")
    print("=" * 40)
    print()
    
    # Test cases
    test_cases = [
        ([], "Main dashboard (no args)"),
        (["--help"], "Root help"),
        (["task"], "Task command (no subcommand)"),
        (["task", "--help"], "Task help"),
        (["workspace"], "Workspace command (no subcommand)"),
        (["workspace", "--help"], "Workspace help"),
        (["input"], "Input command (no subcommand)"),
        (["input", "--help"], "Input help"),
        (["unknowncommand"], "Unknown command"),
    ]
    
    results = {}
    total_start = time.perf_counter()
    
    for args, description in test_cases:
        print(f"=== {description} ===")
        try:
            avg_time = time_command(args, runs=3)
            results[description] = avg_time
        except Exception as e:
            print(f"Failed: {e}\n")
            results[description] = 999.0
    
    total_time = time.perf_counter() - total_start
    
    # Summary
    print("=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    
    for desc, time_val in results.items():
        if time_val < 900:
            print(f"{desc:<35} {time_val:.3f}s")
        else:
            print(f"{desc:<35} FAILED")
    
    print(f"\nTotal benchmark time: {total_time:.1f}s")
    
    # Find fastest/slowest
    valid_results = {k: v for k, v in results.items() if v < 900}
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1])
        slowest = max(valid_results.items(), key=lambda x: x[1])
        
        print(f"Fastest: {fastest[0]} ({fastest[1]:.3f}s)")
        print(f"Slowest: {slowest[0]} ({slowest[1]:.3f}s)")
        
        if fastest[1] > 0:
            ratio = slowest[1] / fastest[1]
            print(f"Speed difference: {ratio:.1f}x")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"benchmark_{timestamp}.txt", "w") as f:
        f.write("AutoClean CLI Benchmark Results\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for desc, time_val in results.items():
            f.write(f"{desc}: {time_val:.3f}s\n")
    
    print(f"\nResults saved to benchmark_{timestamp}.txt")


if __name__ == "__main__":
    main()
