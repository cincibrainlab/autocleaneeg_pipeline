# Migration Guide: Source Localization v2.0.0

This guide explains how to upgrade from the legacy v1.x block to version 2.0.0, which now relies on the `autocleaneeg-eeg2source` PyPI package and always produces 68 Desikan–Killiany (DK) ROI channels saved to the derivatives folder.

## Summary of Changes

| Feature | v1.x | v2.0.0 |
|---------|------|--------|
| Core implementation | Custom, duplicated from pipeline | Delegates to `autocleaneeg-eeg2source` |
| Default output | MNE `SourceEstimate` (STC) | 68-channel ROI EEG (`Raw`/`Epochs`) |
| Optional conversion | `convert_to_eeg=True` | Always on |
| Derivative files | Optional, under `source_localization_eeg/` | Mandatory, under `derivatives/source_localization/` |
| Memory handling | None | `MemoryManager` (configurable cap) |
| EOG handling | Manual | Automatic removal |

## Prerequisites

1. Install the PyPI package (if not already bundled):
   ```bash
   pip install autocleaneeg-eeg2source
   ```
2. Ensure MNE can locate the `fsaverage` dataset (download occurs automatically on first run).

## Updating the Block Files

Pull the latest version of the block from the registry or copy the files in this directory into your deployment. Version 2.0.0 consists of:
- `manifest.json`
- `mixin.py`
- `algorithm.py`
- `README.md`
- `MIGRATION_GUIDE.md`

The v1.0.0 files are preserved in git history (commit `004da4e` and earlier) if rollback is needed.

## Configuration Changes

Remove the legacy options (`convert_to_eeg`, `pick_ori`, `n_jobs`) from your task configs. The new block exposes the following optional parameters:

```python
"apply_source_localization": {
    "enabled": True,
    "value": {
        "method": "MNE",           # currently fixed
        "lambda2": 0.111,
        "montage": "GSN-HydroCel-129",
        "resample_freq": None,
        "max_memory_gb": 8.0
    }
}
```

If you omit the `value` dictionary, all defaults apply.

## Downstream Code Adjustments

Because `self.stc` / `self.stc_list` are no longer produced, downstream blocks must read ROI data via `self.source_eeg`:

- **PSD / Connectivity / FOOOF**: update mixins to accept ROI-level `Raw` or `Epochs` objects. In most cases this means swapping STC-specific utilities for channel-based calculations.
- **Custom scripts**: replace any direct access to `SourceEstimate` attributes with ROI channel operations (e.g., `data.get_data()` on the returned `Raw`/`Epochs`).

During the transition the mixin sets `self.stc` and `self.stc_list` to `None` and emits a warning to catch missed updates.

## Output Location

Every run writes the following files:

```
derivatives/source_localization/
├── {subject}_dk_regions.set
├── {subject}_dk_regions.fdt
└── {subject}_region_info.csv
```

The block determines `{subject}` from the task configuration (preferring `unprocessed_file`, `subject_id`, `base_fname`, `original_fname`) and falls back to the task’s file stem. If no `derivatives_dir` is specified, the mixin uses `<task_dir>/derivatives/source_localization/`.

## Validation Checklist

1. Run a representative task (e.g., `SourceLocalization_Epochs`) and verify that the 68-channel file appears in the derivatives folder.
2. Inspect the log or `self.source_eeg.info` to confirm `nchan == 68` and DK ROI names.
3. Execute downstream PSD/connectivity pipelines to ensure they accept the ROI data.
4. Monitor memory usage on large files; adjust `max_memory_gb` as needed.

## Rollback Strategy

If you need to revert to v1.0.0 temporarily, restore the files from git history:

```bash
# Restore v1.0.0 files from git history (commit 004da4e)
git checkout 004da4e -- blocks/analysis/source_localization/algorithm.py
git checkout 004da4e -- blocks/analysis/source_localization/mixin.py
git checkout 004da4e -- blocks/analysis/source_localization/manifest.json
```

Remember to restore any configuration changes (e.g., re-enable `convert_to_eeg`) that the legacy block expects.

## Support

For questions or to report issues, contact the AutoCleanEEG maintainers or open a ticket in the task registry repository.
