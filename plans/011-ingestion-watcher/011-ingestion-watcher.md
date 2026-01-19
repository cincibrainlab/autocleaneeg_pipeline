# RFC 011 Reasoning Plan: Ingestion Watcher Phase

## Intent

Implement the watcher integration phase from RFC 008 by adding scan/poll helpers and a watchfiles-backed watcher with a polling fallback.

## Ordered Steps

1. Add scan and polling helpers for ready file detection.
2. Implement watchfiles-based watcher with a polling fallback.
3. Extend unit tests to cover scan, poll, and fallback behavior.
4. Run targeted pytest with coverage override and document warnings.

## Rationale

This phase introduces readiness monitoring without a full daemon, enabling iterative testing while keeping watcher integration deterministic and testable.
