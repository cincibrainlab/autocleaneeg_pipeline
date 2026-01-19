# RFC 022 Reasoning Plan: Serve Multi-route Enforcement

## Intent
Implement the RFC 021 decisions in code by enforcing strict route metadata in queues, requiring ingestion roots to exist in strict mode, and adding tests that verify the new guardrails.

## Ordered Steps
1. Enforce strict ingestion root existence during serve config parsing.
2. Reject queue entries missing route identifiers before dispatch.
3. Extend unit coverage for strict parsing and queue enforcement.
4. Run targeted ingestion tests to validate behavior.

## Rationale
The sequence applies the agreed guardrails at the parser and dispatch boundaries, then proves the behavior with focused unit tests to keep the enforcement deterministic and auditable.
