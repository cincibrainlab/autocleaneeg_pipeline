# RFC 021 Reasoning Plan: Serve Multi-route Implementation

## Intent
Implement the RFC 019 multi-route serve model and the RFC 020 review adjustments across config parsing, routing, queue and ledger metadata, CLI validation and reporting, and tests, and capture open questions for operator alignment.

## Ordered Steps
1. Build a normalized ServeConfig parser with defaults, legacy migration, and linting.
2. Add route-aware matching and dispatch with priority and glob specificity tie handling.
3. Extend queue, ledger, and receipt metadata to persist route IDs and ingestion roots.
4. Update CLI validation and serve run output to reflect route-aware behavior.
5. Add unit tests and run targeted validation.
6. Add plain-English questions and assumptions that require stakeholder decisions.

## Rationale
This sequence aligns with the RFC 019 sprints by establishing deterministic configuration semantics before routing and dispatch, then validating operator-facing behavior through CLI output and tests.
