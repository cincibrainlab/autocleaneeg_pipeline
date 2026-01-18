# RFC 006 Reasoning Plan: Serve Command Family

## Intent

Implement `serve list`, `serve validate`, and `serve deploy` to align with RFC 002, including a read-only deployed YAML set and clear workspace status reporting.

## Ordered Steps

1. Extend the serve subcommand parser to include list/validate/deploy.
2. Implement workspace discovery and status listing for YAMLs, runtimes, and automations.
3. Add YAML validation that checks required keys, paths, and mode alignment.
4. Deploy configs into a read-only `deploy/` directory after validation.
5. Run basic CLI tests against the test workspace.

## Rationale

The command family should remain minimal while enforcing the deployment gate described in RFC 002, keeping operator-edited YAMLs separate from deployed, read-only configs.
