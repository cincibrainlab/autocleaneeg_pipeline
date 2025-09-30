#!/usr/bin/env python3
"""Test script for external plugin block loading."""

import sys
sys.path.insert(0, '/Volumes/braindata/cincineuro_github/autocleaneeg_pipeline/src')

print("=" * 80)
print("Testing External Plugin Block Discovery")
print("=" * 80)
print()

# Import the mixins module - this triggers discovery
from autoclean.mixins import DISCOVERED_MIXINS

print()
print("Discovery Results:")
print(f"  Total mixin classes discovered: {len(DISCOVERED_MIXINS)}")
print()

# Get the combined mixin class
mixin_class = DISCOVERED_MIXINS[0]
print(f"Combined mixin class: {mixin_class.__name__}")
print()

# Check for external block methods
methods = [m for m in dir(mixin_class) if not m.startswith('_')]
print(f"Total methods available: {len(methods)}")
print()

# Check for source localization method
has_source_loc = hasattr(mixin_class, 'apply_source_localization')
print(f"Has 'apply_source_localization' method: {has_source_loc}")

if has_source_loc:
    method = getattr(mixin_class, 'apply_source_localization')
    print(f"  Method: {method}")
    print(f"  Defined in: {method.__module__ if hasattr(method, '__module__') else 'unknown'}")
    print()
    print("✅ External plugin block loaded successfully!")
else:
    print()
    print("⚠️  External plugin block was not loaded.")
    print("   Checking ~/.autoclean/blocks/ for plugin files...")

print()
print("=" * 80)