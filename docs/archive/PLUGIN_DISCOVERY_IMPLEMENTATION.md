# External Plugin Block Discovery - Implementation Details

**Date:** 2025-09-30
**Status:** Phase 1 Complete - Discovery System Implemented
**File Modified:** `src/autoclean/mixins/__init__.py`

---

## What Was Implemented

The pipeline now supports **external plugin blocks** through automatic discovery. This allows blocks to be distributed as standalone Python files that users can drop into designated directories.

### Key Features

1. **Three Search Locations** (checked in order):
   - `~/.autoclean/blocks/` - User-specific blocks
   - `./blocks/` - Project-local blocks
   - Task-registry path (via `AUTOCLEAN_TASK_REGISTRY_PATH` env var)

2. **Automatic Discovery**:
   - Scans for `*.py` files (skips `_*.py` and `__init__.py`)
   - Loads modules directly from file paths
   - Extracts classes ending with "Mixin"
   - Adds to Task inheritance chain

3. **Zero Configuration**:
   - No registration required
   - Drop file → pipeline finds it
   - Works identically to internal mixins

4. **Graceful Failure**:
   - External block errors print warnings but don't crash
   - Internal mixins continue to work normally

---

## Code Changes

### Location
`src/autoclean/mixins/__init__.py` (lines 251-304)

### Before
```python
# --- Assemble the Final Tuple of Mixins for Task Inheritance ---
_discovered_other_mixins.sort(key=lambda cls: cls.__name__)
```

### After
```python
# --- External Block Discovery ---
_discovered_external_mixins: List[Type[Any]] = []

_EXTERNAL_BLOCK_PATHS = [
    Path.home() / ".autoclean" / "blocks",
    Path.cwd() / "blocks",
]

# Add task-registry if env var set
if os.getenv("AUTOCLEAN_TASK_REGISTRY_PATH"):
    _registry_path = Path(os.getenv("AUTOCLEAN_TASK_REGISTRY_PATH")) / "blocks"
    if _registry_path.exists():
        _EXTERNAL_BLOCK_PATHS.append(_registry_path)

for external_path in _EXTERNAL_BLOCK_PATHS:
    if not external_path.exists():
        continue

    for block_file in external_path.rglob("*.py"):
        if block_file.name.startswith("_"):
            continue

        relative_path = block_file.relative_to(external_path)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        module_name = f"autoclean_external_blocks.{'.'.join(module_parts)}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, block_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        class_obj.__module__ == module_name
                        and class_name.endswith("Mixin")
                        and class_obj is not _BASE_MIXIN_CLASS
                    ):
                        if class_obj not in _discovered_external_mixins:
                            _discovered_external_mixins.append(class_obj)
                            print(f"✓ Loaded external block: {class_name} from {block_file.name}")
        except Exception as e:
            print(f"Warning: Could not load external block from {block_file}: {e}")
            continue

# --- Assemble the Final Tuple of Mixins for Task Inheritance ---
_discovered_other_mixins.sort(key=lambda cls: cls.__name__)
_discovered_external_mixins.sort(key=lambda cls: cls.__name__)

_final_mixins_list: List[Type[Any]] = []
if _base_mixin_found:
    _final_mixins_list.append(_BASE_MIXIN_CLASS)

# Add internal mixins
for mixin_cls in _discovered_other_mixins:
    if mixin_cls not in _final_mixins_list:
        _final_mixins_list.append(mixin_cls)

# Add external mixins
for mixin_cls in _discovered_external_mixins:
    if mixin_cls not in _final_mixins_list:
        _final_mixins_list.append(mixin_cls)
```

---

## How It Works

### 1. Module Discovery
```python
for block_file in external_path.rglob("*.py"):
    # Finds: source_localization_plugin.py, my_custom_block.py, etc.
```

### 2. Dynamic Loading
```python
spec = importlib.util.spec_from_file_location(module_name, block_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

### 3. Class Extraction
```python
for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
    if class_name.endswith("Mixin"):
        _discovered_external_mixins.append(class_obj)
```

### 4. Inheritance Integration
```python
class Task(ABC, *DISCOVERED_MIXINS):
    # DISCOVERED_MIXINS now includes external blocks!
```

---

## Usage Example

### 1. Create a Plugin Block

**File:** `~/.autoclean/blocks/my_custom_analysis.py`

```python
"""My custom analysis plugin block."""

from autoclean.calc.fooof_analysis import fit_fooof_models

__block_metadata__ = {
    "name": "my_custom_analysis",
    "version": "1.0.0",
    "category": "analysis",
}

class MyCustomAnalysisMixin:
    def apply_my_custom_analysis(self, data=None):
        """Apply my custom analysis."""
        # Use algorithms from pipeline
        results = fit_fooof_models(...)
        return results
```

### 2. Use in Task

**File:** `workspace/tasks/my_task.py`

```python
from autoclean.core.task import Task

config = {
    "apply_my_custom_analysis": {
        "enabled": True,
        "value": {...}
    }
}

class MyTask(Task):
    def run(self):
        self.import_raw()
        self.apply_my_custom_analysis()  # Method from plugin!
```

### 3. Run

```bash
autocleaneeg-pipeline process --task MyTask --file data.raw
```

**Output:**
```
✓ Loaded external block: MyCustomAnalysisMixin from my_custom_analysis.py
Processing with MyTask...
```

---

## Environment Variable

To load blocks from the task-registry:

```bash
export AUTOCLEAN_TASK_REGISTRY_PATH=/path/to/autocleaneeg-task-registry
```

This adds `/path/to/autocleaneeg-task-registry/blocks/` to the search path.

---

## Testing

### Test Setup
1. Copy prototype to test location:
   ```bash
   mkdir -p ~/.autoclean/blocks
   cp /path/to/task-registry/blocks/source_localization_plugin.py \
      ~/.autoclean/blocks/
   ```

2. Create test script:
   ```python
   from autoclean.core.task import Task

   # source_localization_plugin.py provides:
   # - apply_source_localization()
   # - convert_stc_to_eeg()
   # - Methods available to ALL tasks

   print(hasattr(Task, 'apply_source_localization'))  # True
   ```

### Expected Behavior
- On import of `autoclean.mixins`, should print:
  ```
  ✓ Loaded external block: SourceLocalizationMixin from source_localization_plugin.py
  ```
- All Task instances have `apply_source_localization()` method
- Method calls algorithms from `autoclean.calc.source` (no duplication)

---

## Error Handling

### Graceful Degradation
If an external block fails to load:
```python
try:
    # Load external block
except Exception as e:
    print(f"Warning: Could not load external block from {block_file}: {e}")
    continue  # Don't crash - skip this block
```

### Internal Mixins Protected
- External block errors don't affect internal mixins
- Pipeline continues to function normally
- Only the problematic external block is skipped

---

## Benefits Achieved

| Feature | Status |
|---------|--------|
| Zero Duplication | ✅ Blocks import from pipeline |
| Auto-Discovery | ✅ Pipeline finds blocks automatically |
| Drop-in Installation | ✅ Copy file → works |
| Backward Compatible | ✅ Existing code unaffected |
| Error Isolation | ✅ Bad external blocks don't crash pipeline |

---

## Next Steps (Phase 1 Remaining)

1. **Migration Script**: Tool to convert multi-file blocks → single-file plugins
2. **Testing Framework**: Automated tests for block loading and execution
3. **Documentation**: User guide for creating custom blocks

---

## Technical Notes

### Import Mechanism
Uses `importlib.util.spec_from_file_location()` to load modules directly from file paths without requiring them to be in `sys.path`.

### Module Naming
External blocks get synthetic module names:
```python
# File: ~/.autoclean/blocks/source_localization_plugin.py
# Module name: autoclean_external_blocks.source_localization_plugin
```

This avoids conflicts with internal modules.

### MRO (Method Resolution Order)
External mixins are added **after** internal mixins in the inheritance chain:
```python
Task → ABC → BaseMixin → Internal Mixins → External Mixins
```

If methods conflict, internal mixins take precedence.

---

## Files Modified

1. **src/autoclean/mixins/__init__.py** (lines 251-327)
   - Added external block discovery
   - Added external mixins to final inheritance list

---

## Related Documents

- **Architecture:** `/Volumes/braindata/cbl_github/autocleaneeg-task-registry/PLUGIN_BLOCK_ARCHITECTURE.md`
- **Summary:** `/Volumes/braindata/cbl_github/autocleaneeg-task-registry/PLUGIN_ARCHITECTURE_SUMMARY.md`
- **Prototype:** `/Volumes/braindata/cbl_github/autocleaneeg-task-registry/blocks/source_localization_plugin.py`
- **Plan:** `PLUGIN_BLOCKS_PLAN.md`