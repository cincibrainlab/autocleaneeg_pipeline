# RFC 015 Reasoning Plan: Ingestion Integration Phase

## Intent

Implement the next RFC 008 phase by wiring readiness scanning to dispatch execution using a single integration helper.

## Ordered Steps

1. Add an integration helper that loads serve config, scans readiness, and dispatches ready files.
2. Enforce ingestion folder alignment with serve config.
3. Extend unit tests for integrated readiness + dispatch flow.
4. Run targeted pytest with coverage override and document warnings.

## Rationale

This phase provides a minimal, testable bridge between readiness detection and runtime dispatch without introducing long‑running services.
