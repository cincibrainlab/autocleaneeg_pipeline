# AutoClean Python Task File Refactor - Implementation Plan

## Overview
Transform AutoClean from YAML+Task separation to unified Python task files where users write a single .py file containing both configuration and processing logic. Replace stage_files configuration with optional `export=True` parameters on mixin methods.

## Current State Analysis
- Pipeline loads YAML config + instantiates Task classes from task registry
- Export functions require stage definitions in YAML `stage_files` section
- Tasks call `save_raw_to_set()` and `save_epochs_to_set()` explicitly with hardcoded stage names
- Task registry auto-discovers classes in `src/autoclean/tasks/`

## Target State
```python
# User writes: my_resting_task.py
from autoclean import Task

class MyRestingTask(Task):
    def __init__(self, config):
        # Set user configuration
        self.settings = {
            'resample': {'enabled': True, 'value': 250},
            'filtering': {'enabled': True, 'l_freq': 1, 'h_freq': 40},
            'reference_step': {'enabled': True, 'value': 'average'},
            'epochs': {'enabled': True, 'length': 4, 'overlap': 0.5},
            # ... other settings
        }
        super().__init__(config)  # Call parent initialization
    
    def run(self):
        self.import_raw()
        self.run_basic_steps(export=True)  # Optional export
        self.run_ica(export=False)         # Skip export
        self.create_regular_epochs(export=True)

# Usage
pipeline = Pipeline(autoclean_dir="output/", task="my_resting_task.py")
pipeline.process_file(file_path="data.set", task='MyRestingTask')

or 
pipeline = Pipeline(autoclean_dir="output/")
pipeline.add_task("my_resting_task.py") #Where this would add it to the package and not just the python session
pipeline.process_file(file_path="data.set", task='MyRestingTask')

```

---

## Phase 1: Core Infrastructure Setup ✅ **COMPLETED**
**Goal**: Establish foundation for loading Python task files and dynamic stage generation

### Step 1.1: Modify Pipeline Class ✅
- [x] Add `add_task(task_file_path)` method to register Python task files
- [x] Keep backward compatibility with existing YAML-based approach
- [x] Add `_load_python_task()` method to dynamically import and register task classes
- [x] Add `_generate_default_stage_config()` method to create stage_files dict automatically
- [x] Update `_validate_task()` to handle both built-in tasks and user-registered tasks
- [x] Extend task registry to include user-registered tasks alongside built-in ones

### Step 1.2: Dynamic Stage Configuration Generator ✅
- [x] Create method to generate default stage_files configuration
- [x] Include standard stages: `post_import`, `post_basic_steps`, `post_clean_raw`, `post_epochs`, `post_comp`
- [x] Auto-assign stage numbers and suffixes based on processing order
- [x] Ensure generated config matches current export function expectations

### Step 1.3: Python Task File Loader ✅
- [x] Implement dynamic import mechanism using `importlib`
- [x] Scan imported module for Task subclasses
- [x] Handle multiple Task classes in one file (use first found or require single class)
- [x] Add proper error handling for import failures and missing Task classes
- [x] Maintain security by validating file paths and preventing code injection

### Step 1.4: Task Configuration Integration ✅
- [x] Update Task base class to support `self.settings` attribute from user's `__init__` method
- [x] Update `_check_step_enabled()` in BaseMixin to check `self.settings` first, fallback to YAML config
- [x] Ensure backward compatibility with existing YAML-based task configurations
- [x] Add validation for `self.settings` format and structure
- [x] Handle cases where `self.settings` is not defined (fallback to YAML behavior)

---

## Phase 2: Export Parameter Integration ✅ **COMPLETED**
**Goal**: Add `export=True` functionality to all mixin methods

### Step 2.1: Update BaseMixin Export Methods ✅
- [x] Modify `_save_raw_result()` to accept optional stage name parameter
- [x] Modify `_save_epochs_result()` to accept optional stage name parameter  
- [x] Add `_auto_export_if_enabled()` helper method for conditional exporting
- [x] Update error handling to gracefully handle missing stage configurations

### Step 2.2: Add Export Parameters to High-Level Mixin Methods ✅
**Files modified:**
- [x] `basic_steps.py` - `run_basic_steps()` method
- [x] `ica.py` - `run_ica()` method  
- [x] `regular_epochs.py` - `create_regular_epochs()` method
- [x] `gfp_clean_epochs.py` - `gfp_clean_epochs()` method

**For each high-level method:**
- [x] Add `export: bool = False` parameter
- [x] Add conditional export call at end: `self._auto_export_if_enabled(data, stage_name, export)`
- [x] Keep `stage_name` parameter for optional configuration
- [x] Update docstrings to document export parameter

**Note**: Individual low-level methods (like `filter_data()`, `resample()`) get their parameters from `self.settings` configured in `__init__` method, not as direct parameters.

### Step 2.3: Smart Stage Name Generation ✅
- [x] Create `_generate_stage_name()` method in BaseMixin
- [x] Map method names to stage names (e.g., `run_basic_steps` → `post_basic_steps`)
- [x] Add `_ensure_stage_exists()` for automatic stage config generation
- [x] Handle missing stage configurations gracefully

---

## Phase 3: Export Function Modernization ✅ **COMPLETED**
**Goal**: Make export functions work seamlessly with new dynamic approach

### Step 3.1: Update Export Function Interface ✅
- [x] Modify `save_raw_to_set()` to handle missing stage configurations gracefully
- [x] Modify `save_epochs_to_set()` to handle missing stage configurations gracefully
- [x] Modify `save_stc_to_file()` to handle missing stage configurations gracefully
- [x] Add automatic stage configuration generation when stage not in config
- [x] Maintain backward compatibility with existing YAML-based calls

### Step 3.2: Dynamic Stage Directory Creation ✅  
- [x] Update `_get_stage_number()` to handle dynamically generated stages
- [x] Ensure stage directories are created consistently with `mkdir(parents=True, exist_ok=True)`
- [x] Handle stage numbering when stages are called out of order
- [x] Add robust fallback numbering for edge cases
- [x] Maintain deterministic stage numbering for reproducibility

### Step 3.3: Enhanced Error Handling ✅
- [x] Add descriptive error messages for missing configurations
- [x] Provide helpful suggestions for auto-generated stages
- [x] Log stage configuration generation for debugging
- [x] Handle edge cases in file path generation and directory creation
- [x] Improved error messages with context about dynamic vs configured stages

---

## Phase 4: Backward Compatibility & Testing
**Goal**: Ensure existing workflows continue to work while new approach is available

### Step 4.1: Maintain YAML Support ✅
- [ ] Ensure existing Pipeline usage continues to work unchanged
- [ ] Add clear detection logic for YAML vs Python task files
- [ ] Test all existing task implementations work as before
- [ ] Update documentation to show both approaches

### Step 4.2: Create Test Suite ✅
- [ ] Unit tests for Python task file loading
- [ ] Unit tests for export parameter functionality  
- [ ] Integration tests comparing YAML vs Python approaches
- [ ] Test edge cases: malformed task files, missing classes, etc.
- [ ] Performance tests to ensure no regression

### Step 4.3: Example Task Files ✅
- [ ] Create example Python task files for common use cases
- [ ] Convert existing YAML+Task combinations to Python format
- [ ] Add comprehensive docstrings and comments
- [ ] Create migration guide for existing users

---

## Phase 5: Documentation & Migration Tools
**Goal**: Support user transition to new system

### Step 5.1: Update Documentation ✅
- [ ] Update Pipeline class docstrings with new usage patterns
- [ ] Add examples to README showing Python task file approach
- [ ] Update CLAUDE.md with new development patterns
- [ ] Create migration guide from YAML to Python approach

### Step 5.2: Migration Utilities (Optional) ✅
- [ ] Create script to convert YAML+Task to Python task file
- [ ] Add validation tool for Python task files
- [ ] Create template generator for common task patterns
- [ ] Add linting rules for task file best practices

---

## Implementation Notes

### Key Design Decisions
1. **Backward Compatibility**: Keep YAML approach working to avoid breaking existing workflows
2. **Auto-Configuration**: Generate stage_files config automatically to remove user burden
3. **Progressive Enhancement**: `export=False` by default, users opt-in to exporting
4. **Error Resilience**: Graceful fallbacks when stage configurations are missing

### Testing Strategy  
1. **Unit Tests**: Each component in isolation
2. **Integration Tests**: Full pipeline workflows with both approaches
3. **Regression Tests**: Ensure existing functionality unchanged
4. **User Acceptance Tests**: Validate simplified user experience

### Rollout Plan
1. **Phase 1-3**: Core implementation with feature flags
2. **Phase 4**: Testing and validation
3. **Phase 5**: Documentation and examples
4. **Post-Release**: Monitor usage, gather feedback, iterate

### Risk Mitigation
- **Import Security**: Validate Python file paths, use restricted import mechanisms
- **Configuration Conflicts**: Clear precedence rules when both YAML and Python configs exist  
- **Performance Impact**: Minimize overhead of dynamic configuration generation
- **User Confusion**: Clear documentation distinguishing old vs new approaches

---

## Success Criteria
- [ ] Users can create single Python file with task definition and configuration
- [ ] `export=True` parameter works on all mixin methods
- [ ] No breaking changes to existing YAML-based workflows  
- [ ] Performance equivalent to current implementation
- [ ] Clear migration path with examples and documentation
- [ ] Test coverage >90% for new functionality