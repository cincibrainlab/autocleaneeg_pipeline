# MATLAB FOOOF Block

MATLAB-backed FOOOF analysis for EEGLAB `.set` files using `eeg_htpCalcFooof`.

## Purpose

This block packages the workflow from `temp/run_fooof_batch.m` into a route-safe,
single-subject analysis step that runs inside the AutoClean task system.

It is intended for AutoClean outputs that already exist as EEGLAB `.set` files,
especially epoched resting-state exports such as `*_rest_comp_epo.set`.

## What It Writes

Under the task derivatives directory:

```text
derivatives/
  matlab/fooof/{subject}/
    {subject}_fooof_summary.csv
    {subject}_fooof_aperiodic.csv
    {subject}_fooof_manifest.json
    eeg_htpCalcFooof/
      ... MATLAB-generated fit and summary CSVs ...
```

## Config

```python
"apply_matlab_fooof": {
    "enabled": True,
    "value": {
        "vhtp_path": "/path/to/vhtp",
        "eeglab_path": "/path/to/eeglab2024.2.1",
        "spect_freqs": [1, 55],
        "save_fooof_img": False,
        "parallel": False,
        "startup_options": "-nodesktop",
        "startup_timeout_seconds": 60.0,
        "license_file": None,
        "artifacts_subdir": "matlab/fooof"
    }
}
```

## Usage

In a task:

```python
class MyTask(Task):
    def run(self):
        self.import_epochs()
        self.apply_matlab_fooof()
```

## Operational Notes

- This block requires a MATLAB-capable AutoClean environment.
- `vhtp_path` and `eeglab_path` are intentionally explicit. They are external
  runtime requirements and should not be hidden behind hard-coded machine paths.
- The block uses a bundled MATLAB wrapper function to keep the Python side thin
  and to preserve predictable output artifacts for route automation.
