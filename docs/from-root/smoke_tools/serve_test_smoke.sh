#!/usr/bin/env bash
set -euo pipefail

# One-command entrypoint for issue #141 serve test smoke workflow.
# Usage:
#   scripts/serve_test_smoke.sh /path/to/serve-workspace

WORKSPACE="${1:-$(pwd)}"
REPORT_DIR="${REPORT_DIR:-${WORKSPACE}/smoke-reports}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_PATH="${REPORT_DIR}/serve-test-smoke-${STAMP}.json"

python3 "$(dirname "$0")/serve_test_smoke.py" \
  --workspace "${WORKSPACE}" \
  --mode test \
  --mock-runner \
  --report-json "${REPORT_PATH}" \
  --reset-existing

echo "[serve-smoke] report: ${REPORT_PATH}"
