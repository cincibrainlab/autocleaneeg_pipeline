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
`fooof`, `power_coupling`, `pac`, `hfo`, `connectivity`. The `input` field may
name a built-in input or an earlier block's `output`/`aliases` value.

If a block raises, the error still propagates. Before it does, the runner
records the completed blocks and the failed block (`status: "failed"` with the
error text) in `step_postprocessing_analysis` metadata and in
`postprocessing_analysis/resolved_settings.json`, so CSVs from earlier blocks
stay accounted for.

## fooof

The orchestrated `fooof` block fits each spectrum with
`specparam.SpectralModel`. It accepts a DataFrame (or a mapping containing one
under `spectra`, `psd`, or `data`) produced by `sensor_psd` or `source_psd`.
Spectra are grouped by `subject`, `channel`, `roi`, and `time_window` when those
columns exist.

Settings: `freq_range` (default `[1, 45]`), `aperiodic_mode` (`fixed` or
`knee`), `run_periodic`, `peak_width_limits` (default `[1, 12]`), `max_n_peaks`
(default 6), `min_peak_height` and `peak_threshold` (default 2.0).
`peak_residual_threshold` is accepted as a legacy alias for `min_peak_height`
(default 0.1). Invalid settings, such as `peak_width_limits: [12, 1]`, raise
`ValueError` before any spectrum is fit.

Outputs: `<subject>_postprocessing_fooof_aperiodic.csv` and
`<subject>_postprocessing_fooof_periodic.csv`. Aperiodic columns include
`offset`, `knee` (always present, empty in fixed mode), `exponent`, `r_squared`, `error` (mean
absolute error), `status`, and `fit_message`. Peak rows add `center_frequency`,
`power` (log10 units above the aperiodic fit), and `bandwidth`. The periodic
file has its headers even when no peaks are found.

`status` is one of `SUCCESS`, `INSUFFICIENT_DATA`, `FIT_FAILED`, or
`INVALID_PSD`. `FIT_FAILED` records the exception text in `fit_message`.
`INVALID_PSD` means a PSD value in range is zero, negative, or non-finite.
Bins at 0 Hz (or below) are dropped before fitting, so a `freq_range` that
starts at 0 still works and the spacing stays even. If no spectrum fits
successfully, a warning logs the first `fit_message`.

## Oscillation and connectivity blocks

These four blocks default to `input: "clean_raw"` and also accept epochs.
Each writes `<subject>_postprocessing_<block>.csv`; `hfo` writes
`_events.csv` and `_summary.csv` instead.

| Block | Purpose | Settings (defaults) |
| --- | --- | --- |
| `power_coupling` | Correlation of log band-power envelopes between channel/band pairs | `pairs`, `bands`, `picks`, `window_s` (1.0) |
| `pac` | Phase-amplitude coupling (Tort modulation index, with surrogates) | **required:** `phase_bands`, `amp_bands`. Optional: `channel_pairs`, `n_bins` (18), `n_surrogates` (200), `min_shift_s` (1.0), `seed` (0), `picks` |
| `hfo` | High-frequency oscillation detection by robust RMS threshold | `band` (120-250 Hz), `rms_window_s` (0.003), `threshold_sd` (3.0), `min_duration_s` (0.015), `max_duration_s` (0.3), `min_cycles` (4), `merge_gap_s` (0.01), `picks` |
| `connectivity` | Channel-level connectivity | **required:** `bands`. Optional: `methods` (all of `wpli`, `wpli2_debiased`, `plv`, `coh`, `pli`, `aec`), `epoch_length` (2.0 s), `n_epochs`, `picks` |

For connectivity, `wpli2_debiased` is the recommended primary measure. It is
debiased, so unconnected pairs have a noise floor near 0. Spectral methods use
multitaper estimates across epochs; `aec` uses a continuous band-pass plus
Hilbert envelope.

`regions` maps a region name to a channel-name prefix, for example
`{"cortex": "CTX", "hippocampus": "HPC"}`. It adds `region`, `region_a` /
`region_b`, or `phase_region` / `amp_region` columns to the output. Channels
that match no prefix get an empty region. A region whose prefix matches no
channel raises `ValueError`.

Data quality: spans annotated `BAD_*` are excluded, and channels in
`info["bads"]` are excluded even if named in `picks`. For Raw data, the good
spans are handled as follows:

- `power_coupling`, `pac`: each good segment is filtered and edge-trimmed on
  its own, and their values are pooled across segments.
- `hfo`: each good segment is filtered on its own but is not edge-trimmed, so
  events at span edges are counted. It uses one robust threshold per channel,
  pooled across segments. Spans shorter than `2 * min_duration_s` are dropped.
- `connectivity`: fixed-length epochs (`epoch_length`) are cut so that they
  never cross a `BAD_*` span.
- Segments shorter than an analysis-specific minimum are dropped. The count is
  stored in `n_segments_dropped` (DataFrame `attrs`) for `power_coupling`,
  `pac`, and `hfo`. `power_coupling` and `pac` also store
  `n_segments_dropped_by_band` (dict, key `"low-high"` in Hz) with the spans
  that are too short for that band.
- `power_coupling` and `pac` apply the minimum length per band, not for the
  lowest requested band. The filter edge trimmed at each end is at least one
  cycle of the band's low edge (a pair of different bands uses the larger
  edge). `power_coupling` needs one full window after trimming. `pac` needs at
  least 3 cycles of the phase band after trimming, and at least 10 cycles in
  total. For example, 2 s epochs cannot support a 1-4 Hz band (the 1 s edges
  leave nothing). Adding a delta band never changes gamma results.
- A band or pair without enough data does not fail the block. Its row has
  `status` = `insufficient_data`, NaN statistics, and the usable
  `n_windows` (`power_coupling`). Other rows have `status` = `ok`. The block
  raises only if no band or pair can be computed. The error names the minimum
  usable length per band.

A block refuses to run when a requested band reaches the Nyquist frequency
(for example after `resample_step`) or exceeds the data low-pass (the
`filtering` `h_freq`). The error names the band and the setting to change.

Further checks:

- `pac`: each amplitude band must be at least 2 x the phase band's high edge
  wide.
- `connectivity`: bands must be at least 2 Hz wide, and a band's low edge must
  be at least `5 / epoch_length` Hz (2.5 Hz at the default). Violations raise
  `ValueError`; a failing method or band is not skipped.

Example for a mouse recording with cortex and hippocampus channels:

```python
"postprocessing_analysis": {
    "enabled": True,
    "value": {
        "power_coupling": {
            "enabled": True,
            "regions": {"cortex": "CTX", "hippocampus": "HPC"},
            "bands": {"theta": [4, 8], "gamma": [30, 80]},
        },
        "pac": {
            "enabled": True,
            "regions": {"cortex": "CTX", "hippocampus": "HPC"},
            "phase_bands": [[4, 8]],
            "amp_bands": [[30, 80]],
        },
        "hfo": {
            "enabled": True,
            "regions": {"hippocampus": "HPC"},
            "band": [120, 250],
        },
        "connectivity": {
            "enabled": True,
            "regions": {"cortex": "CTX", "hippocampus": "HPC"},
            "bands": {"theta": [4, 8]},
            "methods": ["wpli", "coh"],
        },
    },
}
```

For Sensor PSD, `baseline` applies only to Epochs. It is ignored with a warning
for continuous Raw input. Time windows are cropped after Epochs baseline
correction.

## Behavior changes

For existing users:

- `fooof` now uses specparam. In periodic rows, `power` is log10 above the
  aperiodic fit (it was absolute PSD). The `residual` column is removed.
  `error` is mean absolute error (it was RMSE). New columns: `knee`,
  `fit_message`. The method string is `specparam_tabular` (it was
  `tabular_psd_parameterization`). Knee mode is allowed. Invalid settings
  raise.
- `sensor_psd` raises for a band with no frequency bins (it gave 0.0) and for
  `fmax` above Nyquist.
- Source connectivity now estimates across epochs
  (`spectral_connectivity_epochs`, consecutive epochs), so values differ from
  before. It raises on a failing method or band instead of skipping it. Graph
  metrics are now computed; the `bctpy` import was broken before.
  `char_path_length` is the mean shortest-path length on connection lengths
  `1 / weight` (bct `distance_wei`; unreachable pairs are excluded), so a
  stronger network has a shorter path. `small_worldness` is
  `clustering * char_path_length`, a simplified index carried over from the
  original code; it is not the standard small-world sigma (no random-graph
  normalization), so do not compare it across studies. Modularity uses Louvain with a fixed seed.
- `power_coupling` and `pac` output has a new `status` column
  (`ok` / `insufficient_data`). Trimming and minimum segment length are now
  decided per band (see above).
