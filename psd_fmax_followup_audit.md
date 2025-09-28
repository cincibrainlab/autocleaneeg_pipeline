# PSD FMAX Follow-Up Audit (2024-11)

This note captures the remaining places in the AutoClean pipeline where
`psd_fmax` is either ignored or handled inconsistently after the ICA report
remediation.

## Confirmed Compliance
- **ICA classification + reports** (`src/autoclean/mixins/signal_processing/ica.py`,
  `src/autoclean/mixins/viz/ica.py`): the ceiling is persisted from
  `component_rejection` and propagated into `plot_component_for_classification`.
- **ICVision plotting helpers** (`src/autoclean/functions/visualization/icvision_layouts.py`):
  clamp PSD axes to the provided ceiling, including the `.webp` outputs used for
  OpenAI Vision.
- **Wavelet QA metrics/reports** (`src/autoclean/functions/preprocessing/wavelet_thresholding.py`):
  the PSD computation now honours the optional `psd_fmax` value threaded from
  configuration and surfaces it in the generated summaries.

## Outstanding Gaps
- **Topography helper ignores ceiling** –
  `src/autoclean/functions/visualization/plotting.py:361` calls
  `raw.compute_psd(fmax=50)` unconditionally. Any task that relies on
  `plot_psd_topography` (directly or via `step_psd_topo_figure`) will still show
  energy up to 50 Hz even if the config narrows the band.
- **Report mixin hardcodes 80 Hz window** –
  `src/autoclean/mixins/viz/visualization.py:822` sets `fmax = 80` before
  building PSD comparisons and topographies. This affects every report
  generated through `step_psd_topo_figure`.

## Suggested Next Steps
1. Thread `psd_fmax` through the PSD/topography mixin helpers so the derivative
   figures match the ICA output.
2. Add regression tests that inspect the PSD axis limits/titles for
   `plot_psd_topography` and `step_psd_topo_figure` to ensure the ceiling is
   respected once the plumbing exists.

## Testing
No automated runs were executed for this audit. The findings are based on static
code review with `rg` against commit `a56c4f7`.
