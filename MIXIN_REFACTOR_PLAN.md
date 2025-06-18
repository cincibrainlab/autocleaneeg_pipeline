# AutoClean Mixin Refactor Implementation Plan

## Project Overview
Extract mixin methods into standalone functions while maintaining pipeline compatibility. Simple translation of existing mixin code into pure functions with parameter-based configuration.

## Success Criteria
- [ ] All processing functions available as standalone imports
- [ ] Existing task files work identically 
- [ ] Mixin methods accept optional parameter overrides
- [ ] Comprehensive documentation for standalone functions

---

## Phase 1: Infrastructure Setup
**Goal:** Create basic structure for standalone functions

### Tasks:
- [x] Create `src/autoclean/functions/` directory structure
  - [x] `preprocessing/` (filtering, resampling, referencing)
  - [x] `epoching/` (regular, eventid, statistical learning)
  - [x] `artifacts/` (channels, ICA, detection)
  - [x] `visualization/` (plotting, reports)
- [x] Create `__init__.py` files with exports
- [x] Update main `src/autoclean/__init__.py` for function imports
- [x] Set up test structure in `tests/functions/`

---

## Phase 2: Core Preprocessing Functions
**Goal:** Extract basic signal processing functions

### Extract These Functions:
- [x] `filter_data()` from `BasicStepsMixin.filter_data()`
- [x] `resample_data()` from `BasicStepsMixin.resample_data()`
- [x] `rereference_data()` from `BasicStepsMixin.rereference_data()`
- [x] `drop_channels()` from `BasicStepsMixin.drop_outer_layer()`
- [x] `crop_data()` from `BasicStepsMixin.crop_duration()`
- [x] `trim_edges()` from `BasicStepsMixin.trim_edges()`

### For Each Function:
1. Copy logic from mixin method
2. Remove `self` dependencies 
3. Use direct parameters instead of config
4. Add comprehensive docstring
5. Create standalone function in appropriate module
6. Refactor mixin to call standalone function + handle config/overrides
7. Add to exports
8. Write basic tests

---

## Phase 3: Epoching Functions
**Goal:** Extract epoching functionality

### Extract These Functions:
- [x] `create_regular_epochs()` from `RegularEpochsMixin`
- [x] `create_eventid_epochs()` from `EventIdEpochsMixin` 
- [x] `create_sl_epochs()` from `StatisticalLearningEpochsMixin`
- [x] `detect_outlier_epochs()` from outlier detection mixins
- [x] `gfp_clean_epochs()` from GFP cleaning mixin

### ✅ PHASE 3 COMPLETE:
- ✅ **create_regular_epochs()**: Fixed-length epoch creation with metadata
- ✅ **create_eventid_epochs()**: Event-based epochs with flexible event_id formats  
- ✅ **create_sl_epochs()**: Statistical learning epochs with syllable validation
- ✅ **detect_outlier_epochs()**: FASTER algorithm-based outlier detection
- ✅ **gfp_clean_epochs()**: Global Field Power-based epoch cleaning
- ✅ **Mixin Integration**: All mixins refactored to use standalone functions
- ✅ **Tests**: Comprehensive test suite for all epoching functions
- ✅ **Exports**: All functions enabled in main module with lazy imports

**All epoching functions now available as standalone imports while maintaining full backward compatibility**

---

## Phase 4: Channel Operations ✅ COMPLETED
**Goal:** Extract channel detection and cleaning

### Extract These Functions:
- [x] `detect_bad_channels()` from `ChannelsMixin.clean_bad_channels()`
  - ✅ **Function**: PyPREP NoisyChannels integration with correlation, deviation, and RANSAC detection
  - ✅ **Tests**: Comprehensive test suite with 5 test methods covering basic functionality, method breakdown, exclusion handling, validation, and error conditions
  - ✅ **Documentation**: Extensive docstring with parameter guidelines, examples, and performance notes
  - ✅ **Integration**: Added to all module exports with lazy imports enabled
- [x] `interpolate_bad_channels()` from channel cleaning logic  
  - ✅ **Function**: Spherical spline interpolation for Raw and Epochs data
  - ✅ **Tests**: 8 test methods covering basic usage, info['bads'] handling, parameter validation, montage requirements, and epochs support
  - ✅ **Documentation**: Complete docstring with best practices, requirements, and examples
  - ✅ **Integration**: Added to all module exports with lazy imports enabled
- [x] `assign_channel_types()` from EOG assignment logic *(completed in Phase 2)*

### Status:
- ✅ **Module Structure**: Created `src/autoclean/functions/artifacts/` with channels module
- ✅ **Implementation**: Both functions fully implemented with comprehensive error handling and validation
- ✅ **Testing**: 13 test methods total, all passing with proper synthetic data setup and montage handling
- ✅ **Exports**: All functions enabled in main module with lazy imports

**All channel operation functions now available as standalone imports with full PyPREP integration**

---

## Phase 5: Artifact Detection & ICA
**Goal:** Extract artifact processing

### Extract These Functions:
- [ ] `fit_ica()` from `IcaMixin.run_ica()` 
- [ ] `classify_ica_components()` from ICA classification
- [ ] `apply_ica_rejection()` from component rejection
- [ ] `detect_muscle_artifacts()` from artifact detection
- [ ] `annotate_noisy_segments()` from annotation logic

---

## Phase 6: Advanced Processing ✅ COMPLETED
**Goal:** Extract specialized functions

### Extract These Functions:
- [x] `autoreject_epochs()` from autoreject integration
  - ✅ **Function**: Complete AutoReject implementation with cross-validation parameter optimization
  - ✅ **Tests**: Comprehensive test suite with 4 test methods covering basic functionality, custom parameters, input validation, and channel selection
  - ✅ **Documentation**: Extensive docstring with algorithm overview, parameter guidelines, examples, and performance notes
  - ✅ **Integration**: Added to all module exports with lazy imports enabled
- [x] `annotate_noisy_segments()` from segment-based rejection
  - ✅ **Function**: IQR-based statistical outlier detection for identifying noisy data segments
  - ✅ **Tests**: 5 test methods covering basic usage, parameter customization, input validation, edge cases, and channel selection
  - ✅ **Documentation**: Complete docstring with detection algorithm, parameter guidelines, and examples
  - ✅ **Integration**: Added to all module exports with lazy imports enabled
- [x] `annotate_uncorrelated_segments()` from spatial correlation analysis
  - ✅ **Function**: Spatial neighbor correlation analysis with multiple aggregation methods (max, mean, trimmean)
  - ✅ **Tests**: 6 test methods covering basic usage, correlation methods, input validation, montage requirements, and edge cases  
  - ✅ **Documentation**: Complete docstring with algorithm overview, correlation methods, and examples
  - ✅ **Integration**: Added to all module exports with lazy imports enabled

### Status:
- ✅ **Module Structure**: Created `src/autoclean/functions/advanced/` with autoreject and segment_rejection modules
- ✅ **Implementation**: All functions fully implemented with comprehensive error handling, input validation, and helper functions
- ✅ **Testing**: 18 test methods total, all passing with proper type checking and synthetic data compatibility
- ✅ **Dependencies**: Proper handling of autoreject dependency with mocking for tests and clear error messages
- ✅ **Exports**: All functions enabled in main module with lazy imports

**All advanced processing functions now available as standalone imports with full AutoReject integration and pylossless-style segment rejection**

---

## Phase 7: Visualization & Reports
**Goal:** Extract plotting and reporting

### Extract These Functions:
- [ ] `plot_raw_comparison()` from visualization mixins
- [ ] `plot_ica_components()` from ICA plotting
- [ ] `plot_psd_topography()` from PSD visualization
- [ ] `generate_processing_report()` from report generation

---

## Implementation Pattern

### Standalone Function Template:
```python
def function_name(
    data: Union[mne.io.base.BaseRaw, mne.Epochs],
    param1: Type = default,
    param2: Type = default
) -> Union[mne.io.base.BaseRaw, mne.Epochs]:
    """Detailed docstring with examples"""
    # Pure function logic extracted from mixin
    result = data.copy()
    # ... processing logic
    return result
```

### Mixin Wrapper Template:
```python
def mixin_method(self, data=None, use_epochs=False, param1=None, param2=None):
    """Brief docstring referencing standalone function"""
    data = self._get_data_object(data, use_epochs)
    
    # Get config values
    is_enabled, config_value = self._check_step_enabled("step_name")
    if not is_enabled:
        return data
    
    # Apply overrides or use config
    final_param1 = param1 if param1 is not None else config_value.get("param1")
    final_param2 = param2 if param2 is not None else config_value.get("param2")
    
    # Call standalone function
    result = standalone_function(data, final_param1, final_param2)
    
    # Pipeline integration
    self._update_instance_data(data, result, use_epochs)
    self._save_raw_result(result, "stage_name")
    self._update_metadata("operation", metadata_dict)
    
    return result
```

## Testing Strategy
- [ ] Test standalone functions in isolation
- [ ] Test mixin wrappers preserve existing behavior
- [ ] Test parameter overrides work correctly
- [ ] Verify all existing tasks still pass

This is a straightforward code extraction and reorganization project focused on separating pure signal processing logic from pipeline infrastructure.