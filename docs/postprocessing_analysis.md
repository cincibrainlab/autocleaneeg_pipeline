# Postprocessing analysis

`postprocessing_analysis` runs optional analysis blocks after the main cleaning
task. A failure is logged with the enabled block context and traceback, but does
not discard or prevent saving the successfully cleaned output.

```python
"postprocessing_analysis": {
    "enabled": True,
    "value": {
        "sensor_psd": {
            "enabled": True,
            "input": "clean_epochs",
            "output": "sensor_spectra",
            "freq_bands": "default",
        },
        "fooof": {
            "enabled": True,
            "input": "sensor_spectra",
            "aperiodic_mode": "fixed",
            "freq_range": [1, 45],
        },
    },
}
```

Blocks execute in this order: `sensor_psd`, `source_localization`, `source_psd`,
then `fooof`. The `input` field may name a built-in input or an earlier block's
`output`/`aliases` value.

The orchestrated `fooof` block is a lightweight tabular PSD parameterization,
not the full standalone specparam implementation. It accepts a DataFrame (or a
mapping containing one under `spectra`, `psd`, or `data`) produced by
`sensor_psd` or `source_psd`. It estimates a fixed log-log aperiodic model and
rejects `aperiodic_mode: "knee"` rather than mislabeling a fixed fit. Use the
standalone `apply_fooof_aperiodic` and `apply_fooof_periodic` APIs when the full
source-estimate/specparam workflow is required.

For Sensor PSD, `baseline` applies only to Epochs. It is ignored with a warning
for continuous Raw input. Time windows are cropped after Epochs baseline
correction.
