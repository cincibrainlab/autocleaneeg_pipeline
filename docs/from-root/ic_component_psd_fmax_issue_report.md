# IC Component PSD FMAX Issue Analysis

## Problem Statement

The IC component view is not respecting the `psd_fmax` configuration setting. Despite having `psd_fmax: 45` in the configuration, the PSD plots are showing frequencies up to 80Hz instead of the configured 45Hz limit.

## Root Cause Analysis

### 🔍 **Investigation Findings**

1. **Configuration Loading**: ✅ `psd_fmax` is properly loaded from config in `classify_ica_components()`
2. **ICVision Implementation**: ✅ `psd_fmax` is properly implemented in ICVision-specific functions
3. **ICA Component Plotting**: ❌ `psd_fmax` is **NOT** passed to the plotting function

### 📋 **Code Flow Analysis**

#### Step 1: Configuration Loading (✅ Working)
**File**: `src/autoclean/mixins/signal_processing/ica.py`
**Lines**: 250-255

```python
if psd_fmax is None:
    psd_fmax = config_params_nested.get("psd_fmax")
    if psd_fmax is None and "psd_fmax" in step_config_main_dict:
        psd_fmax = step_config_main_dict.get("psd_fmax")
    if psd_fmax is not None:
        message("info", f"Using psd_fmax={psd_fmax} Hz from config")
```

**Status**: ✅ **WORKING** - Configuration is properly loaded and logged.

#### Step 2: Parameter Passing to Classification (✅ Working)
**File**: `src/autoclean/mixins/signal_processing/ica.py`
**Lines**: 276-277

```python
if psd_fmax is not None:
    extra_kwargs["psd_fmax"] = psd_fmax
```

**Status**: ✅ **WORKING** - Parameter is passed to the classification function.

#### Step 3: ICVision Layout Implementation (✅ Working)
**File**: `src/autoclean/functions/visualization/icvision_layouts.py`
**Lines**: 391-394

```python
if psd_fmax is not None:
    fmax_psd = min(psd_fmax, nyquist - 0.51)
else:
    fmax_psd = min(80.0, nyquist - 0.51)
```

**Status**: ✅ **WORKING** - ICVision functions properly respect `psd_fmax`.

#### Step 4: ICA Component Report Generation (❌ BROKEN)
**File**: `src/autoclean/mixins/viz/ica.py`
**Lines**: 503-515

```python
fig = plot_component_for_classification(
    ica,
    raw_fast,
    idx,
    output_dir=pdf_path.parent,
    return_fig_object=True,
    classification_label=classification_label,
    classification_confidence=classification_confidence,
    classification_reason=classification_reason,
    classification_method=classification_method,
    raw_full=raw,
    source_filename=source_name,
    # ❌ MISSING: psd_fmax parameter
)
```

**Status**: ❌ **BROKEN** - `psd_fmax` parameter is not passed to the plotting function.

### 🎯 **The Core Issue**

The `_plot_ica_components()` method in the ICA visualization mixin does **NOT** retrieve the `psd_fmax` value from the configuration and pass it to the `plot_component_for_classification()` function.

## Detailed Technical Analysis

### 1. **Missing Configuration Retrieval**

The `_plot_ica_components()` method needs to:
1. Retrieve the `psd_fmax` value from the configuration
2. Pass it to the `plot_component_for_classification()` function

### 2. **Configuration Storage Gap**

The `classify_ica_components()` method loads `psd_fmax` from config but doesn't store it in the metadata or as an instance variable, making it unavailable to the plotting functions.

### 3. **Function Signature Compatibility**

The `plot_component_for_classification()` function **DOES** support `psd_fmax`:
- **File**: `src/autoclean/functions/visualization/icvision_layouts.py`
- **Line**: 94
- **Parameter**: `psd_fmax: Optional[float] = None`

### 4. **ICVision Classification Path (✅ Working)**

- **File**: `src/autoclean/mixins/signal_processing/ica.py`
- **Lines**: 240-287 (`classify_ica_components`)

`psd_fmax` pulled from the pipeline config is threaded straight into the `label_components` entry point that ships with the `autoclean-icvision` package:

```python
extra_kwargs: Dict[str, object] = {}
if psd_fmax is not None:
    extra_kwargs["psd_fmax"] = psd_fmax
...
label_components(raw, ica, **kwargs)
```

The ICVision adapter renders the exact same component layout (`plot_component_for_classification`) when preparing the `.webp` frames used for the OpenAI Vision inference call. Because we hand `psd_fmax` to that adapter, those images already clamp the PSD to the configured ceiling. In other words, **ICVision “sees” the correct PSD-limited plots today**; the breakage is isolated to the PDF reporting mixin that was never given the same value.

## Ground Truth for IC Component Pictures

### 🎯 **Expected Behavior**

When `psd_fmax: 45` is configured:
1. **PSD plots should show**: 1Hz to 45Hz frequency range
2. **Plot titles should show**: "IC{X} Power Spectrum (1-45Hz)"
3. **Frequency axis should be limited**: to 45Hz maximum

### 🚨 **Current Behavior**

When `psd_fmax: 45` is configured:
1. **PSD plots actually show**: 1Hz to 80Hz frequency range (default fallback)
2. **Plot titles show**: "IC{X} Power Spectrum (1-80Hz)"
3. **Frequency axis goes to**: 80Hz (ignoring configuration)

### 📊 **Evidence of the Issue**

**Configuration**:
```python
"component_rejection": {
    "enabled": True,
    "method": "icvision", 
    "value": {
        "psd_fmax": 45.0,  # ← This should limit PSD to 45Hz
    },
}
```

**Actual Plot Behavior**:
- PSD plots show 0-80Hz range
- Plot titles say "Power Spectrum (1-80Hz)"
- Configuration is ignored

## Solution Requirements

### 🔧 **Immediate Fix Needed**

1. **Modify `_plot_ica_components()` method**:
   - Store the `psd_fmax` value during classification (e.g., attach to `self._ica_plot_psd_fmax` or metadata)
   - Fall back to config lookup when a report is generated standalone
   - Pass the resolved value to `plot_component_for_classification()`

2. **Configuration Retrieval Logic** (used both at classification time and as a fallback in `_plot_ica_components()`):
   ```python
   psd_fmax = getattr(self, "_ica_plot_psd_fmax", None)
   if psd_fmax is None:
       is_enabled, step_config_main_dict = self._check_step_enabled("component_rejection")
       if is_enabled and step_config_main_dict:
           config_params_nested = step_config_main_dict.get("value", {})
           psd_fmax = config_params_nested.get("psd_fmax")
           if psd_fmax is None:
               psd_fmax = step_config_main_dict.get("psd_fmax")
   ```

3. **Parameter Passing**:
   ```python
   fig = plot_component_for_classification(
       ica,
       raw_fast,
       idx,
       # ... other parameters ...
       psd_fmax=psd_fmax,
   )
   ```

### 🏗️ **Architectural Improvements**

1. **Store `psd_fmax` in Metadata**:
   - Modify `classify_ica_components()` to store `psd_fmax` in metadata
   - Make it available to all subsequent plotting functions

2. **Centralized Configuration Access**:
   - Create a helper method to retrieve plotting parameters
   - Ensure consistency across all visualization functions

## Testing Strategy

### 🧪 **Verification Steps**

1. **Configuration Test**:
   - Set `psd_fmax: 45` in task configuration
   - Run ICA classification and report generation
   - Verify PSD plots show 1-45Hz range

2. **Visual Verification**:
   - Check plot titles show correct frequency range
   - Verify frequency axis limits
   - Confirm PSD data is truncated at configured limit

3. **Edge Case Testing**:
   - Test with different `psd_fmax` values (20Hz, 60Hz, 100Hz)
   - Test with `psd_fmax: null` (should use default 80Hz)
   - Test with sampling rates that affect Nyquist frequency

## Impact Assessment

### 📈 **User Impact**

- **Current**: Users see inconsistent frequency ranges in IC component plots
- **After Fix**: Users will see plots that respect their `psd_fmax` configuration
- **Benefit**: Consistent behavior across all plotting functions

### 🔧 **Implementation Effort**

- **Complexity**: **LOW** - Simple parameter addition
- **Risk**: **LOW** - Additive change, won't break existing functionality
- **Time**: **MINIMAL** - Single method modification

## Conclusion

The issue is a **simple missing parameter** in the `_plot_ica_components()` method. The `psd_fmax` configuration is properly loaded and the plotting function supports it, but the connection between them is missing.

**Priority**: **HIGH** - This affects core functionality and user experience
**Solution**: **SIMPLE** - Add one parameter to one function call
**Risk**: **LOW** - No breaking changes, only additive functionality

The fix requires modifying the `_plot_ica_components()` method in `src/autoclean/mixins/viz/ica.py` to retrieve and pass the `psd_fmax` parameter to the plotting function.

## Tentative Surgical Plan

1. **Persist the ceiling during classification**
   - Capture `psd_fmax` inside `classify_ica_components()` (e.g., `self._ica_plot_psd_fmax = psd_fmax` and store in pipeline metadata for audit).
2. **Teach `_plot_ica_components()` to reuse it**
   - First consult the persisted value; if absent, fall back to the configuration lookup shown above.
   - Thread the result through to `plot_component_for_classification()` and any batch helpers.
3. **Regression guards**
   - Add a unit test in `tests/functions/test_visualization.py` asserting the PSD axis limit changes when `psd_fmax` is supplied.
   - Add an integration-style test (or fixture) confirming `_plot_ica_components()` writes a figure whose title reflects the configured band.
4. **Documentation + changelog**
   - Update docs/tutorial snippets mentioning `psd_fmax` to clarify it now governs both ICVision and report outputs.

Once those steps land, both the classification path and generated PDFs will honor the same `psd_fmax` ceiling.
