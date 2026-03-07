# Branching Policy

## Default workflow

- `main` is the only long-lived branch.
- Do not use a persistent `dev` branch.
- Create short-lived branches from the latest `main`.
- Merge completed work back into `main`.
- Delete branches after merge.

## Branch types

Use clear, short branch names:

- `feature/<name>` for new functionality
- `fix/<name>` for bug fixes
- `chore/<name>` for maintenance or tooling changes
- `docs/<name>` for documentation-only changes

Examples:

- `feature/epoch-indices-export`
- `fix/bdf-file-discovery`
- `chore/disable-dependabot`

## Standard flow

1. Update local `main`.
2. Create a new branch from `main`.
3. Make the change on that branch.
4. Open a pull request targeting `main`.
5. Merge the pull request.
6. Delete the branch locally and remotely.

## Rules

- Do not branch from old feature branches.
- Do not reopen a shared integration branch like `dev`.
- If a branch becomes stale, recreate it from current `main` instead of stacking more work onto an outdated base.
- If a pull request is superseded, close it and delete the branch.

## Dependency updates

- Dependabot version-update PRs are disabled for this repository.
- Dependency changes should be made intentionally from fresh branches off `main`.
- Prefer small, isolated dependency PRs over large bundled updates.

## Cleanup expectations

- Delete merged branches promptly.
- Review stale open branches regularly.
- Keep the remote branch list short and current.
