---
title: Task Wizard ↔ Task Library Integration Plan
description: Project plan to connect the online Task Wizard with the public GitHub Task Library/Registry so scientists can create, preview, and publish templates end‑to‑end.
---

## Executive Summary

We will connect the online Task Wizard directly to the public Task Library (GitHub registry) so a scientist can:

- Start from an official template or a fresh form in the wizard
- Configure steps with clear, non‑technical options
- Preview the generated Python Task file (readable, copy‑paste friendly)
- Publish the template to the public Task Library via a guided “Create Pull Request” flow
- See immediate instructions to install it locally via `autocleaneeg-pipeline task library install <Name>`

The integration uses a small backend (Cloudflare Pages Functions/Workers) and a GitHub App to open safe, consistent PRs against the registry, with automated validation and human review.

## Current State (as of 2025‑09)

Task Wizard (this repo)
- Vite + React app deployed to Cloudflare Pages (`wrangler.toml`, `Dockerfile` present)
- Produces downloadable Python Task files from a step‑by‑step UI
- Audience: scientists and analysts; zero install; outputs a single `.py` file

Task Library/Registry (`cincibrainlab/autocleaneeg-task-registry`)
- Public GitHub repo with `registry.json` and `tasks/*/*.py`
- CI validates entries and stamps a `commit` field on `registry.json`
- Pipeline CLI fetches the index and files via `task library update | list | install`

Pipeline (autocleaneeg_pipeline)
- Adds “Task Library” commands and sync badges for templates
- Supports offline fallback to a bundled snapshot

Gaps
- Wizard does not currently read from or publish to the public registry
- Users must download the file and manually submit via GitHub (friction)

## Integration Goals

1) Discoverability: The wizard shows official templates sourced from the Task Library so users start from best‑practice defaults.
2) One‑click Publishing: “Create Pull Request” from the wizard produces a clean PR in the registry (no local Git required).
3) Safety & Consistency: Enforce naming rules, Task subclassing, and file layout; CI in the registry stays authoritative.
4) Scientist‑friendly UX: Plain language, previews, and copy‑and‑paste CLI instructions after publish.

## High‑Level Architecture

```
User (browser)
   |  (create/edit template)
   v
Task Wizard (React)  —— calls ——>  Wizard Backend (Cloudflare Functions)
                                         |  GitHub App install
                                         |  (App private key in secrets)
                                         v
                                GitHub API (Task Library repo)
```

Components
- Frontend: Vite/React wizard
  - Reads the live `registry.json` to surface official templates (read‑only)
  - Generates `.py` via a Jinja‑style template on the client
  - Posts “Publish” requests to the backend with draft content + metadata
- Backend: Cloudflare Pages Functions/Workers
  - GitHub App OAuth handshake (user signs in with GitHub)
  - Creates a fork (if needed) or a feature branch in the registry
  - Writes `tasks/<category>/<Name>.py`, updates `registry.json`, opens a PR
  - Runs lightweight server‑side validations (syntax, Task subclass, filename/class match)
  - Returns PR URL and next steps to the user
- Registry CI
  - Validates that every `tasks/*/*.py` path appears in `registry.json`
  - Optional: re‑stamp `commit` on merge

## UX Flow (Scientist)

1. Start
   - Choose “Build from a template” (list comes from live registry) or “Start from scratch”
2. Configure
   - Set task name (with live naming checks), montage, filtering, ICA choice, epoching, etc.
3. Preview
   - See the generated Python file and a short “what this task does” summary
4. Publish or Download
   - Download: save `.py` for local use
   - Publish: “Sign in with GitHub” → confirm details → “Create Pull Request”
5. After Publish
   - Show PR link + instructions:
     - “Once merged, run: `autocleaneeg-pipeline task library update && task library install <Name>`”
   - Optionally trigger a CLI copy command for power users

## Data & Naming Rules (enforced by the wizard)

- Name: `PascalCase`, letters/digits only, must match class name and filename
- Path: `tasks/<category>/<Name>.py` with known categories (`resting`, `auditory`, …)
- Class: must subclass `autoclean.core.task.Task`
- Config: validated against a JSON schema (wizard‑side) for safe defaults

## Validation & Safety

Wizard‑side
- Required fields present; category valid; name/class/file consistent
- Python code compiles (AST parse) and contains no forbidden imports (small allowlist)

Backend‑side (Cloudflare Function)
- Re‑validate payload and run AST checks server‑side
- Compute SHA256 for logging; attach as PR metadata

Registry CI
- Ensure `registry.json` references the new file exactly once
- Optional: lint style; auto‑stamp the `commit` on merge

## API Endpoints (Wizard Backend)

- `GET /library/index` → Proxy to `registry.json` (with cache‑busting and short TTL)
- `POST /publish` → Accepts `{ name, category, python, summary }`
  - Auth: GitHub OAuth (user scope `repo` or limited via GitHub App)
  - Action: create branch → write file → patch `registry.json` → open PR
- `GET /status/:pr_number` → Return PR state for UI polling (optional)

## Implementation Plan & Timeline

Phase 0 (1–2 days)
- Read‑only integration
  - Add “Start from Template” drawer populated from live `registry.json`
  - Add link to template source (registry path) and short descriptions

Phase 1 (1 week)
- Backend scaffolding
  - Cloudflare Functions project for `/library/index` and `/publish`
  - Configure GitHub App (App ID, private key, installation on registry repo)
  - Implement OAuth (device code or OAuth web flow) and store token in session/cookie

Phase 2 (1–2 weeks)
- Publish flow MVP
  - Add review screen (diff‑like view) before “Create Pull Request”
  - Server: file write + `registry.json` patch + PR creation
  - Return PR URL + “after publish” instructions
  - Add server + client validations (class/filename/category)

Phase 3 (1 week)
- Moderation & safety
  - CI job comments on common issues (naming, subclassing, missing docstring)
  - Optional auto‑label (e.g., `new-template`, `needs-review`)
  - Quarantine area: `tasks/pending_approval/<Name>.py` if maintainers prefer drafting

Phase 4 (polish, 1 week)
- UX refinements, accessibility, copywriting
- Internationalization hooks (if needed)
- Documentation updates (docs site + wizard help tooltips)

Total: ~4–5 weeks including review cycles

## User Workflow Improvements

- “Pick a template” starts from lab‑approved defaults → fewer decisions required
- “Preview and explain” panel summarizes what the generated task does
- “One‑click publish” eliminates manual Git and PR creation
- Clear install command after merge helps users connect the dots to local use
- Offline: always offer a “Download .py” fallback; publish button disabled with explanation

## Security & Privacy

- Use a GitHub App with least privilege; only allow PRs to the task registry repo
- Never store user access tokens server‑side beyond session; prefer short‑lived tokens
- Server‑side sanitize/validate file content; reject suspicious code patterns
- Rate‑limit `/publish` and add basic abuse prevention (e.g., size limits)

## Open Questions / Decisions

- Do we accept “community” templates directly into main or require a `pending_approval/` path?
- Should we standardize a metadata block (e.g., YAML header) for provenance/versioning in each task file?
- Do we want per‑template screenshots or report thumbnails attached to PRs?

## Success Metrics

- 80% of wizard users can publish a template without leaving the site
- Review queue time to merge is under 3 days for routine updates
- Reduced support requests about “how do I add a task to the library?”

## Appendix: Publishing Checklist (what the backend enforces)

- [ ] Filename `Name.py` where `Name` == class name
- [ ] Path `tasks/<category>/Name.py` where category ∈ { resting, auditory, … }
- [ ] Class `class Name(Task):` and `from autoclean.core.task import Task`
- [ ] Minimal docstring present
- [ ] `registry.json` patched with `{"name": "Name", "path": "tasks/<category>/Name.py"}` (exactly once)
- [ ] PR title: `Add <Name> task template`
- [ ] PR body: brief summary + generated config JSON (inline) for reviewers

