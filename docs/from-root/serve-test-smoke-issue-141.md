# Serve Test-Mode E2E Smoke (Issue #141)

This runbook adds a **one-command**, de-identified smoke workflow for `serve` test mode.

## What it validates

The workflow performs these checks in order:

1. Validate `serve-test.yaml` structure and routes.
2. Deploy config to `deploy/serve-test.yaml` (read-only copy).
3. Verify runtime CLI is resolvable for test mode.
4. Enqueue one de-identified sample file into `queue-test.json`.
5. Run one dispatch pass and wait for queue status transition.
6. Verify output artifacts exist.

By default it uses a **mock runner** so no real EEG processing is required.

---

## One-command operator usage

From repo root:

```bash
docs/from-root/smoke_tools/serve_test_smoke.sh /path/to/serve-workspace
```

Expected success signals:

- JSON output includes:
  - `"statuses_seen": ["pending", "processed"]`
  - `"final_status": "processed"`
- A report is written under:
  - `/path/to/serve-workspace/smoke-reports/serve-test-smoke-*.json`

---

## Script details

Main script: `docs/from-root/smoke_tools/serve_test_smoke.py`

Helpful flags:

- `--workspace <path>` (required)
- `--route-id <id>` (optional, defaults to first enabled route)
- `--bootstrap-runtime` (optional: run `uv` install if runtime CLI missing)
- `--real-runner` (optional: executes real process command instead of mock)
- `--report-json <path>` (optional: write report to explicit path)

Example real processing attempt:

```bash
python3 docs/from-root/smoke_tools/serve_test_smoke.py \
  --workspace /path/to/serve-workspace \
  --mode test \
  --real-runner \
  --report-json /tmp/serve-smoke-real.json
```

---

## Expected artifacts (mock mode)

For the selected route, artifacts are written under:

- `<automation_root>/<workspace_name>/_smoke/dispatch-summary.json`
- `<automation_root>/<workspace_name>/_smoke/runner.log`

These prove dispatch execution and command construction end-to-end without PHI.

---

## Troubleshooting: stuck `pending`

If queue status does not move beyond `pending`, check in this order:

1. **Sentinel missing**
   - If route uses `sentinel_ext` (e.g. `.ready`), ensure `<file><sentinel_ext>` exists.
2. **Route mismatch**
   - Confirm sample filename matches `file_globs` and file is under `ingestion_folders`.
3. **Route disabled / wrong route**
   - Ensure route is enabled and selected route ID is correct.
4. **Queue contamination**
   - Legacy entries without `route_id` can block dispatch. Remove/migrate old queue entries.
5. **Runtime CLI missing**
   - Ensure test runtime has `.venv/bin/autocleaneeg-pipeline`, or run with `--bootstrap-runtime`.
6. **Permission issues**
   - Verify write access to `deploy/`, queue file, ingestion folder, and automation output root.

Quick debug commands:

```bash
cat /path/to/workspace/queue-test.json
ls -la /path/to/workspace/deploy
ls -la /path/to/workspace/automations
```

---

## Evidence captured

- `docs/from-root/evidence/issue-141/run-output.txt`
- `docs/from-root/evidence/issue-141/run-report.json`

These were produced from a de-identified fixture workspace using mock mode.
