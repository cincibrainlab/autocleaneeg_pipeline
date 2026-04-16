# Tracked Runtime Assets

This directory contains the checked-in frontend assets served by the packaged
API/Serve surface.

## Why These Files Are Tracked

- the current `web/` build writes directly into this directory
- the packaged API expects a runtime web bundle to exist here
- these files are treated as release/runtime assets, not disposable local build
  output

## Source Of Truth

- editable frontend source lives in `web/`
- built artifacts in this directory should be regenerated from `web/`, not
  edited by hand

## Policy

- `index.html` and `assets/*` stay tracked because they are part of the shipped
  runtime surface today
- if the release model changes later, this policy can be revisited in a focused
  packaging change
