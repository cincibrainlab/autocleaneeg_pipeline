# RFC 016 Reasoning Plan: Ingestion Loop Phase

## Intent

Implement the next RFC 008 phase by adding a bounded ingestion loop that iterates across ingestion roots and dispatches ready files until idle.

## Ordered Steps

1. Add an ingestion loop result model capturing iterations and pending roots.
2. Implement a loop helper that cycles through configured ingestion roots.
3. Extend unit tests for loop behavior and idle exits.
4. Run targeted pytest with coverage override and document warnings.

## Rationale

A bounded loop completes the scheduling bridge without introducing a daemon, providing a safe, testable iteration mechanism for readiness → dispatch.
