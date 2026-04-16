# Pending Approval Tasks

This directory contains task implementations that are present in-tree but are
not part of the stable public task surface.

## Status

- these tasks are under maintainer review
- they may change shape, move, or be removed without normal stability guarantees
- they are not the recommended first choice for public-facing examples or docs

## Policy

- if a task becomes supported for normal users, it should move into
  `src/autoclean/tasks/`
- if a task is retained only for historical context, it should move to an
  archive or example-oriented location instead of staying here indefinitely
- docs and release notes should avoid presenting this directory as equivalent to
  curated built-in tasks

## Note For Contributors

If you are evaluating whether a task belongs here, use the lifecycle terms
documented in the repo root docs and contributor guidance.
