# RFC 004 Reasoning Plan: Automation Mode Validation

## Intent

Validate automation mode behavior by comparing batch processing with two single-file runs using the `--automation` flag, confirming equivalent outputs and additive logs.

## Ordered Steps

1. Run batch processing on the sample directory using `--automation` and record outputs.
2. Run two single-file processing commands with `--automation` into a separate output root.
3. Compare exports and logs between batch and single-run outputs.
4. Document warnings, results, and any discrepancies.

## Rationale

This validation ensures automation mode behaves consistently across batch and single-run workflows, which is critical for idempotent automation and operator trust.
