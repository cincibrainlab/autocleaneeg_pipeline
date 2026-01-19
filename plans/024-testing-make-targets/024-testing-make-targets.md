# RFC 024 Reasoning Plan: Testing Make Targets

## Intent
Add simple Makefile targets that make common test paths easier to run during local development.

## Ordered Steps
1. Define short, easy targets for fast unit checks and ingestion-specific tests.
2. Add fail-fast variants for unit and integration suites.
3. Update the Makefile help text to surface the new targets.

## Rationale
Short, consistent targets reduce friction for routine validation, while fail-fast variants provide quick feedback during iterative development.
