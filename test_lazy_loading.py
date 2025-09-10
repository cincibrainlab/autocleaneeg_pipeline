#!/usr/bin/env python3
"""
Test script to verify lazy task discovery performance improvements.

This script tests the performance difference between the original task discovery
and the new lazy loading system.
"""

import time
import sys
from pathlib import Path

# Add src to path so we can import autoclean modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_original_discovery():
    """Test performance of original task discovery system."""
    print("Testing original task discovery...")
    start_time = time.time()
    
    try:
        from autoclean.utils.task_discovery import safe_discover_tasks, get_task_by_name
        
        # Full task discovery
        tasks, invalid, skipped = safe_discover_tasks()
        print(f"Found {len(tasks)} tasks, {len(invalid)} invalid, {len(skipped)} skipped")
        
        # Single task lookup
        if tasks:
            task_name = tasks[0].name
            task_class = get_task_by_name(task_name)
            print(f"Successfully retrieved task: {task_name}")
        
    except Exception as e:
        print(f"Error in original discovery: {e}")
    
    end_time = time.time()
    return end_time - start_time

def test_lazy_discovery():
    """Test performance of lazy task discovery system."""
    print("\nTesting lazy task discovery...")
    start_time = time.time()
    
    try:
        from autoclean.utils.lazy_task_discovery import (
            safe_discover_tasks_lazy, 
            get_task_by_name_lazy,
            get_lazy_cache
        )
        
        # Clear any existing cache to simulate fresh start
        cache = get_lazy_cache()
        cache.invalidate_cache()
        
        # Full task discovery (should cache results)
        tasks, invalid, skipped = safe_discover_tasks_lazy()
        print(f"Found {len(tasks)} tasks, {len(invalid)} invalid, {len(skipped)} skipped")
        
        # Single task lookup (should use cache)
        if tasks:
            task_name = tasks[0].name
            task_class = get_task_by_name_lazy(task_name)
            print(f"Successfully retrieved task: {task_name}")
        
    except Exception as e:
        print(f"Error in lazy discovery: {e}")
    
    end_time = time.time()
    return end_time - start_time

def test_cached_performance():
    """Test performance of cached lazy discovery."""
    print("\nTesting cached lazy discovery performance...")
    start_time = time.time()
    
    try:
        from autoclean.utils.lazy_task_discovery import (
            safe_discover_tasks_lazy, 
            get_task_by_name_lazy
        )
        
        # This should use cached data
        tasks, invalid, skipped = safe_discover_tasks_lazy()
        print(f"Found {len(tasks)} cached tasks")
        
        # Multiple single task lookups
        task_names = [task.name for task in tasks[:3]]  # Test first 3 tasks
        for task_name in task_names:
            task_class = get_task_by_name_lazy(task_name)
            print(f"Retrieved cached task: {task_name}")
        
    except Exception as e:
        print(f"Error in cached discovery: {e}")
    
    end_time = time.time()
    return end_time - start_time

def test_single_task_performance():
    """Test performance of single task lookup without full discovery."""
    print("\nTesting efficient single task lookup...")
    start_time = time.time()
    
    try:
        from autoclean.utils.lazy_task_discovery import get_task_by_name_lazy, get_lazy_cache
        
        # Clear cache to test efficient single task lookup
        cache = get_lazy_cache()
        cache.invalidate_cache()
        
        # Try to find a common built-in task without full discovery
        common_task_names = ["RestingEyesOpen", "RestingEyesClosed", "ChirpDefault"]
        
        for task_name in common_task_names:
            task_class = get_task_by_name_lazy(task_name)
            if task_class:
                print(f"Efficiently found task: {task_name}")
                break
        else:
            print("No common tasks found")
        
    except Exception as e:
        print(f"Error in efficient lookup: {e}")
    
    end_time = time.time()
    return end_time - start_time

def main():
    """Run all performance tests."""
    print("=" * 60)
    print("LAZY TASK DISCOVERY PERFORMANCE TEST")
    print("=" * 60)
    
    # Test original system
    original_time = test_original_discovery()
    print(f"Original discovery time: {original_time:.3f}s")
    
    # Test lazy system (first run)
    lazy_time = test_lazy_discovery()
    print(f"Lazy discovery time (first run): {lazy_time:.3f}s")
    
    # Test cached performance
    cached_time = test_cached_performance()
    print(f"Cached discovery time: {cached_time:.3f}s")
    
    # Test efficient single lookup
    single_time = test_single_task_performance()
    print(f"Efficient single lookup time: {single_time:.3f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Original system:           {original_time:.3f}s")
    print(f"Lazy system (first run):   {lazy_time:.3f}s")
    print(f"Lazy system (cached):      {cached_time:.3f}s")
    print(f"Efficient single lookup:   {single_time:.3f}s")
    
    if original_time > 0:
        cached_improvement = ((original_time - cached_time) / original_time) * 100
        single_improvement = ((original_time - single_time) / original_time) * 100
        
        print(f"\nPerformance improvements:")
        print(f"Cached access:   {cached_improvement:.1f}% faster")
        print(f"Single lookup:   {single_improvement:.1f}% faster")

if __name__ == "__main__":
    main()
