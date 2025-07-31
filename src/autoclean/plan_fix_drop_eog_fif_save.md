I'm not entirely certain if the proposed change is the absolute minimum fix, as there could be deeper pipeline configurations or task-specific overrides influencing the output behavior that I haven't fully explored yet—perhaps something in the task's config or a global export setting that's bypassing folder creation only for FIF saves. However, based on the code I've reviewed, the custom saving in drop_eog_channels does appear to be the outlier, directly writing a single file without subfolder, while other steps use standardized methods that enforce directories. This makes me tentatively think it's minimal, but I should question if switching to _save_raw_result might introduce unintended format changes, like from FIF to SET, which could affect downstream tools if they're FIF-dependent.

Delving deeper, the minimalism holds up because the change targets only the save operation, preserving the rest of the method's logic for dropping channels and handling epochs/raw types—it's not overhauling the entire mixin or pipeline. I'm gaining more confidence here as I consider that the standard savers already handle BIDS compliance and folder creation, so aligning with them should resolve the inconsistency without side effects, assuming no hard dependencies on FIF for this step. Still, I wonder if a even smaller tweak, like just adding folder creation around the existing save, could work, but that might duplicate code and violate DRY principles.

Ultimately, yes, this seems like the minimum viable fix after cross-referencing the export.py module and other mixins; it leverages existing pipeline infrastructure for consistency, requires the fewest lines of change, and avoids introducing new bugs by not reinventing saving logic. If testing reveals issues, we could refine it, but as a senior dev approach, this is efficient and maintainable.

### Detailed Implementation Plan for Fixing Inconsistent Output in `drop_eog_channels`

#### 1. **Problem Description**
The `drop_eog_channels` method in `src/autoclean/mixins/signal_processing/channels.py` produces a single FIF file directly in the stage directory (e.g., `07_drop_eog_channels_raw.fif`) instead of creating a subfolder with basenamed intermediates, which is the expected behavior for other pipeline steps. This inconsistency disrupts the standardized output structure, potentially complicating downstream analysis, BIDS compliance, and user expectations.

#### 2. **Root Cause Analysis**
- The method uses custom saving logic (lines 427-440) that manually constructs a filename and calls `result_data.save()` directly to the stage directory.
- Other steps in the pipeline use standardized methods like `_save_raw_result` or `_save_epochs_result` (inherited from base mixins), which create stage-specific subfolders (e.g., `07_drop_eog_channels/`) and save files inside them (typically in SET format for raw data).
- This custom approach bypasses the pipeline's directory creation in `utils/file_system.py` and export handling in `io/export.py`, leading to the observed single-file output.

#### 3. **Proposed Solution**
Replace the custom saving block with calls to the existing `_save_raw_result` or `_save_epochs_result` methods. This ensures:
- A subfolder is created (e.g., `07_drop_eog_channels/`).
- The file is saved with the basename inside (e.g., `basename_drop_eog_channels_raw.set`).
- Consistency with other steps, switching to SET format (which is standard for intermediates and BIDS-compliant).
- Minimal code changes: Remove ~14 lines of custom logic and add 4-5 lines invoking the standard methods.

**Potential Benefits:**
- Aligns with pipeline conventions.
- Improves maintainability by reducing duplicated code.
- Ensures future-proofing if export logic changes globally.

**Potential Risks:**
- Format switch from FIF to SET might affect tools expecting FIF; verify in testing.
- If the step is used in non-raw contexts (e.g., epochs-only tasks), ensure the conditional handles it correctly.

#### 4. **Implementation Steps**
1. **Preparation (Dev Lead/Team) - 15-30 minutes**
   - Review the current code in `src/autoclean/mixins/signal_processing/channels.py` (focus on lines 403-450).
   - Confirm the base mixin (likely `src/autoclean/mixins/base.py`) defines `_save_raw_result` and `_save_epochs_result`. If not, trace inheritance to ensure availability.
   - Create a feature branch: `git checkout -b fix/drop-eog-channels-output-consistency`.

2. **Code Changes (Developer) - 30-45 minutes**
   - Open `src/autoclean/mixins/signal_processing/channels.py`.
   - Locate the try block starting at line ~403.
   - Replace the custom export block (lines ~427-441) with the following:
     ```
     # Export the result using standard pipeline saving
     if use_epochs:
         self._save_epochs_result(result_data, stage_name)
     else:
         self._save_raw_result(result_data, stage_name)

     message("info", f"Exported {stage_name} data using standard pipeline method")
     ```
   - Remove any redundant variables (e.g., `stage_number`, `exported_filename`, `save_path`) if they're no longer used.
   - Ensure the method still returns `result_data` and updates instance data correctly (lines ~444-446 remain unchanged).

3. **Update Dependencies/Configs (If Needed) - 15 minutes**
   - Check if any task configs (e.g., in `tasks/` or `configs/`) explicitly set `export=True` or FIF format for this step. If so, adjust to align with SET.
   - No changes expected to `io/export.py` or `utils/file_system.py`, as we're leveraging existing logic.

4. **Commit and PR (Developer) - 10 minutes**
   - Commit changes: `git commit -m "Fix inconsistent output in drop_eog_channels by using standard save methods"`.
   - Push and create a PR: Target `main` branch, add description linking to this issue.
