# Docs Index

This directory contains the main project documentation tree.

## Canonical Contributor Docs

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [README.md](../README.md)
- [BRANCHING.md](../BRANCHING.md)

## Documentation Areas

- `docs/tutorials/`: user and workflow tutorials
- `docs/api_reference/`: API reference sources
- `docs/development/`: development-oriented published docs
- `docs/archive/`: archived documentation and older implementation notes
- tracked `serve-*.html` pages at the top level are intentionally versioned
  operational guides that are part of the current Serve documentation surface

## Notes

- Prefer the root [CONTRIBUTING.md](../CONTRIBUTING.md) as the canonical contributor guide.
- Use `plans/` for larger, structured planning artifacts.
- Keep published user/developer docs in `docs/`, active engineering plans in
  `plans/`, and historical material in the matching `archive/` trees.
- Treat `docs/_build/` as generated output, not canonical source.
- GitHub Pages publishing is driven by [`.github/workflows/docs.yml`](../.github/workflows/docs.yml),
  not by a manually maintained `gh-pages` branch.
- Treat tracked `src/autoclean/api/static/` files as the current shipped runtime
  web bundle, regenerated from `web/` rather than edited by hand.
- For Serve docs, treat `autocleaneeg-serve` as the normal launcher and
  `autocleaneeg-pipeline serve ...` as the lower-level operator/control surface.
