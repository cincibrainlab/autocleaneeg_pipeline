# RestingState Basic Wavelet Task Migration Plan

This note captures the work required to publish the `RestingState_BasicWavelet`
pipeline as an external registry task while keeping the core AutoClean codebase
maintainable.

## Current Status (post-psd_fmax update)
- Wavelet thresholding now accepts optional `psd_fmax`, `threshold_scale`, and
  `picks` values via config and forwards them into the QA helpers so reports
  mirror preprocessing choices.
- `RestingState_BasicWavelet` sets `wavelet_threshold.value.psd_fmax = 45.0` so
  the new plumbing is exercised.

## Migration Steps
1. **Extract task config**
   - Copy the updated `config` block from
     `src/autoclean/tasks/RestingState_BasicWavelet.py` into a new registry entry
     (e.g., `tasks/resting/RestingState_BasicWavelet.py` in the
     `autocleaneeg-task-registry` repo).
   - Preserve the wavelet settings, including the new `psd_fmax` field.

2. **Registry plumbing**
   - Update the registry’s generator (`src/utils/pythonGenerator.ts`) so exported
     Python stubs include the `wavelet_threshold` block with `psd_fmax` when
     present.
   - Add the new task to `registry.json` with an explicit name/path mapping.

3. **Documentation**
   - Add a README note in the registry describing the wavelet variant and its
     expected use cases (e.g., quick denoising before ICA).
   - Reference this plan (or summarize it) in `docs/task_registry_plan.md` to
     keep contributor guidance centralized.

4. **Validation**
   - Run the pipeline locally using the registry-exported task to confirm the
     configuration round-trips and the new PSD ceiling is honoured in the
     generated reports.
   - Capture before/after figures for QA (optional) to document the expected
     visuals.

## Open Questions
- Should additional wavelet-specific metrics (e.g., band definitions above
  45 Hz) ship with the registry task? If so, extend `FREQUENCY_BANDS` or allow a
  task-level override.
- Do we deprecate the in-repo task after migration, or keep it as a thin stub
  that imports from the registry snapshot?

## Next Actions
- [ ] Create the new registry task file with the updated config.
- [ ] Raise a PR in the registry repo with generator + listing updates.
- [ ] Update AutoClean docs once the registry task ships (linking to the new
      location).
