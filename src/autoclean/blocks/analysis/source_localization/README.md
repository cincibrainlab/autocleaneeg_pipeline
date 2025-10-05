# Source Localization Block (v2.0.0)

**Category:** Analysis  \
**Status:** Stable  \
**Package dependency:** [`autocleaneeg-eeg2source`](https://pypi.org/project/autocleaneeg-eeg2source/)

## Overview

This block performs EEG source localization by delegating to the `autocleaneeg-eeg2source` PyPI package. It applies MNE minimum-norm estimation using the `fsaverage` template brain, converts the results to 68 Desikan–Killiany (DK) atlas regions, and saves the derived EEG file directly to the task’s derivatives directory.

Unlike the legacy v1 block, the new implementation does **not** expose raw `SourceEstimate` (STC) objects. Every run produces ROI-level EEG data that is ready for downstream power, connectivity, and spectral parameterisation analyses.

## Key Features

- Guarantees 68 ROI channels (DK atlas)
- Handles both continuous (`Raw`) and epoched (`Epochs`) inputs
- Writes `{subject}_dk_regions.set/.fdt` and `{subject}_region_info.csv` under `derivatives/source_localization/`
- Uses `MemoryManager` to cap RAM usage during processing
- Automatically strips EOG channels prior to localization
- Compatible with downstream ROI-based PSD, connectivity, and FOOOF blocks

## Requirements

- `autocleaneeg-eeg2source >= 0.3.7` (installed automatically when using the bundled pipeline)
- Access to `fsaverage` FreeSurfer data (downloaded on first use by MNE)
- At least 19 EEG channels with a recognised montage (e.g., `GSN-HydroCel-129`)
- Artifact-cleaned data (post-ICA/AutoReject recommended)

## Configuration

```python
config = {
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "MNE",          # Currently only MNE is supported
            "lambda2": 0.111,          # Regularisation (1 / SNR^2)
            "montage": "GSN-HydroCel-129",
            "resample_freq": None,     # Optional downsample target (Hz)
            "max_memory_gb": 8.0       # Memory cap for processing
        }
    }
}
```

### Parameter Reference

| Parameter        | Type    | Default               | Description |
|------------------|---------|-----------------------|-------------|
| `method`         | string  | `"MNE"`              | Inverse solution; only MNE supported by the package |
| `lambda2`        | float   | `0.111`               | Regularisation (1 / SNR²) |
| `montage`        | string  | `"GSN-HydroCel-129"` | Input montage name passed to the package |
| `resample_freq`  | float   | `None`                | Optional resampling frequency before localization |
| `max_memory_gb`  | float   | `8.0`                 | Memory cap enforced by `MemoryManager` |

All parameters are optional—omitting the `value` block will use the defaults.

## Outputs

When the block completes, the following files are written:

- `derivatives/source_localization/{subject}_dk_regions.set`
- `derivatives/source_localization/{subject}_dk_regions.fdt` (binary companion file)
- `derivatives/source_localization/{subject}_region_info.csv`

In-memory, the mixin stores the ROI data in `self.source_eeg` and the file path in `self.source_eeg_file`. Legacy attributes (`self.stc`, `self.stc_list`) are set to `None` to flag the absence of raw STCs.

## Usage Example

```python
from autoclean.core.task import Task

config = {
    "schema_version": "2025.09",
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "montage": "GSN-HydroCel-129"
        }
    }
}

class RestingStateSource(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.rereference_data()
        self.run_ica()
        self.create_regular_epochs()

        roi_epochs = self.apply_source_localization()
        print(f"ROI channels: {roi_epochs.info['nchan']}")  # -> 68
        print(f"Saved file: {self.source_eeg_file}")
```

## Migration Notes

- Replace any usage of `self.stc`/`self.stc_list` in downstream code with `self.source_eeg` (ROI `Raw`/`Epochs`).
- PSD, connectivity, and FOOOF blocks must consume ROI data after adopting this version.
- For workflows that require high-density vertex data, call the `autocleaneeg-eeg2source` package directly outside the block.

See `MIGRATION_GUIDE.md` for detailed upgrade instructions.

## Troubleshooting

| Issue | Remedy |
|-------|--------|
| `ImportError: autocleaneeg-eeg2source not found` | Install with `pip install autocleaneeg-eeg2source` |
| ROI file missing in derivatives | Ensure the task config exposes `derivatives_dir` or the task has a valid `file_path`; the block defaults to `<cwd>/derivatives/source_localization/` otherwise |
| Output shape unexpected | Confirm the input data have a valid montage and minimum channel count |

For additional support, contact the AutoCleanEEG maintainers or open an issue in the registry.
