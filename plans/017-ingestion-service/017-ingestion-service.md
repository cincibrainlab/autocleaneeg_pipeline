# RFC 017 Reasoning Plan: Ingestion Service Phase

## Intent

Implement the next RFC 008 phase by adding a bounded ingestion service loop that repeats ingestion cycles until idle.

## Ordered Steps

1. Add a service result model capturing cycles, idle count, and loop outputs.
2. Implement a service helper that reuses `run_ingestion_loop` until idle limit.
3. Extend unit tests for idle exit behavior.
4. Run targeted pytest with coverage override and document warnings.

## Rationale

This phase establishes a minimal scheduler loop for automation without introducing a daemon, enabling safe iteration and deterministic tests.
