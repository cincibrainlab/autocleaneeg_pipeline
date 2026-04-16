# Plans Directory

This directory is the structured home for larger planning artifacts.

## Layout

- numbered folders like `001-*` through `024-*` hold plan-specific source files
- `main-plan.md` is the higher-level planning log
- `matlab-integration-plan.md` is a currently retained standalone plan
- `archive/` contains retired plan material
- `_site/` is local generated Quarto output and should not be tracked

## Conventions

- keep long-running or multi-phase engineering plans here
- keep plans out of `docs/`
- archive one-time or superseded plans instead of leaving them loose in the active top level
- keep the numbered RFC `.md` + `.qmd` pairs as the current active plan format unless we make an explicit future format change
- use `archive/` for retired standalone plan notes and old plan sets
- use `archive/imported-docs/` for former doc-tree planning notes that belong with planning history instead
- standalone plans that are superseded should be moved out of the active top
  level rather than left beside current planning sources

## Generated Content

- `_site/` is generated output and should be treated as such
- `_site/` should stay ignored and be regenerated locally when needed
- edit the plan source files, not the generated HTML
