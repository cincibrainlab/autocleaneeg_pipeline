# RFC 010 Reasoning Plan: Ingestion Provenance Phase

## Intent

Implement the RFC 008 provenance layout phase by adding deterministic folder resolution and staged receipts with ledger integration.

## Ordered Steps

1. Add provenance folder resolution tied to hash inputs.
2. Stage receipts into deterministic folders and record ledger entries.
3. Extend unit tests for folder resolution and ledger‑backed staging.
4. Run targeted pytest with coverage override and document warnings.

## Rationale

This phase locks in deterministic folder mapping and receipt writing so ingestion can proceed to watcher integration with a stable provenance model.
