# RFC 012 Reasoning Plan: Ingestion Dispatch Phase

## Intent

Implement the dispatch‑planning phase from RFC 008 by adding serve config loading, workspace naming, and dispatch plan construction.

## Ordered Steps

1. Add serve config loader and path resolution helpers.
2. Implement workspace naming from template tokens.
3. Build a dispatch plan structure from config + files.
4. Extend unit tests for workspace naming and dispatch planning.
5. Run targeted pytest with coverage override and document warnings.

## Rationale

Dispatch planning ties ingestion readiness to downstream execution without launching the pipeline, allowing validation of config logic before integrating the serve runtime.
