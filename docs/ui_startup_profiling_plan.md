# UI Startup Profiling Test Plan

## Goal
Verify and document the source of the 20+ second apparent “freeze” when launching the AutoClean Exclusion GUI on macOS, and outline regression checks once mitigations (alternate backends or fallbacks) are in place.

## Preconditions
- macOS host with a working desktop session (no headless display)
- `autocleaneeg-pipeline` installed via `uv` (as in this repo)
- Dataset present at `~/Documents/Autoclean-EEG/output/Chirp_Default/exports/128_SteadyState_D3158_comp_epo.set`
- Optional: `snakeviz` (or similar) installed for visualizing `.prof` files

## Test Matrix
| Scenario | Environment variables | Expected UI behaviour | Purpose |
| --- | --- | --- | --- |
| Baseline Qt browser | *(none)* | Long freeze (~20 s) before data renders | Reproduce current issue |
| Matplotlib fallback | `MNE_BROWSER_BACKEND=matplotlib` | Window responsive immediately | Confirm Qt/OpenGL dependency |
| Qt on alternate display server | `QT_QPA_PLATFORM=xcb` or `wayland` | Should match Matplotlib responsiveness | Validate platform-specific fix |
| Headless guard | Force CLI fallback (future change) | CLI logs warning and skips Qt viewer | Regression once guard implemented |

## Test Procedure

### 1. Baseline reproduction
1. Ensure no profiling artefacts from previous runs: `rm ~/Documents/Autoclean-EEG/profiling/exclude-*.{prof,txt}` *(optional)*
2. Launch exclusion GUI:
   ```bash
   uv run --python 3.11 autocleaneeg-pipeline exclude \
     --exports ~/Documents/Autoclean-EEG/output/Chirp_Default/exports \
     --task-root ~/Documents/Autoclean-EEG/output/Chirp_Default
   ```
3. Observe timestamps; confirm ~20 s delay between “Starting exclusion GUI…” and first redraw (PDF loads).
4. Verify warnings in console: repeated `QPainter::... Painter not active`.
5. Confirm profiler artefacts produced: `exclude-*.prof` and `.txt` in `~/Documents/Autoclean-EEG/profiling/`.

### 2. Profile baseline interaction
1. Start profiler right before launching (already built into CLI).
2. After window appears, perform a single interaction (e.g., scroll, toggle ICA component) within 5 s.
3. Close window promptly.
4. Inspect new `.prof` file with:
   ```bash
   uv run --python 3.11 python -m pstats ~/Documents/Autoclean-EEG/profiling/exclude-YYYYMMDD-HHMMSS.prof
   ```
5. Record total time and whether interaction functions (`_auto_plot_current`, pyqtgraph paint handlers) dominate; if `exec` still holds >90%, issue persists.

### 3. Matplotlib backend validation
1. Set backend and relaunch:
   ```bash
   MNE_BROWSER_BACKEND=matplotlib uv run --python 3.11 autocleaneeg-pipeline exclude ...
   ```
2. Confirm window appears without long freeze.
3. Interact with epochs; note responsiveness.
4. Review profiler artefacts—expect `exec` cumulative time to drop dramatically, with Python painting callbacks showing and no QPainter warnings.

### 4. Alternative Qt display check (optional)
1. If available, run under XQuartz or Wayland session; export `QT_QPA_PLATFORM` accordingly.
2. Repeat launch/profiling.
3. Compare startup delay to baseline; responsiveness similar to Matplotlib indicates OpenGL backend compatibility issue.

### 5. Regression checklist once mitigations land
- Launch with default settings on macOS; window should render in under 2 s.
- Confirm profiler shows <5 s spent in `exec` before first interaction.
- Verify CLI automatically switches to Matplotlib backend in headless/unsupported contexts (future change).
- Ensure the guard logs a clear message and doesn’t crash or hang.

## Data Collection
- Archive profiler outputs with scenario labels (e.g., `exclude-20251003-qt.prof`, `exclude-20251003-mpl.prof`).
- Capture console logs highlighting timestamps and warnings.
- Document qualitative responsiveness (scroll lag, button response) in the test report.

## Follow-up Actions
- If Matplotlib backend resolves the freeze, create issue to add automatic fallback when `QOpenGLWidget` initialization fails.
- If delays persist even with Matplotlib or alternate displays, escalate with MNE Qt browser maintainers; attach `.prof` and console logs.
- Once guard is implemented, integrate this plan into CI smoke tests (headless mode should exit gracefully without hang).

## References
- Profiling artefact: `~/Documents/Autoclean-EEG/profiling/exclude-20251003-102212.prof`
- Profile summary: `~/Documents/Autoclean-EEG/profiling/exclude-20251003-102212.txt`
- Relevant modules: `src/autoclean/tools/autoclean_exclude.py` (plot pipeline), `mne.viz._figure`, `mne_qt_browser._pg_figure`
