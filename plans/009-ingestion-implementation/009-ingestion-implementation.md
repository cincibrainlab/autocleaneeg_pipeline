# RFC 009 Reasoning Plan: Ingestion Implementation (RFC 008)

## Intent

Implement the ingestion foundations defined in RFC 008, focusing on readiness evaluation, hashing, receipts, and deduplication utilities with unit tests.

## Ordered Steps

1. Implement ingestion utilities for hashing, receipts, and readiness evaluation.
2. Add a JSON ledger for duplicate detection.
3. Create unit tests for determinism, receipt lifecycle, and readiness checks.
4. Execute targeted tests with coverage override to satisfy existing pytest settings.
5. Document outcomes, warnings, and follow‑up gaps for watcher integration.

## Rationale

This sequence delivers the core deterministic ingestion primitives required by RFC 008 while keeping watcher integration and dispatch logic as a subsequent phase.
