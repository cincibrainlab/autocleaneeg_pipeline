import { FolderOpen, HeartPulse, RefreshCw, ShieldCheck, TriangleAlert, Wrench } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";

function BooleanPill({ ok, text }: { ok: boolean; text: string }) {
  return (
    <span
      className={[
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        ok ? "bg-emerald-500/15 text-emerald-400" : "bg-zinc-500/15 text-zinc-400",
      ].join(" ")}
    >
      {text}
    </span>
  );
}

export default function Utilities() {
  const {
    data,
    error,
    loading,
    refresh,
  } = usePolling(api.getWorkspaceUtilities, 30000);

  if (!loading && !data?.configured) {
    return (
      <div className="max-w-5xl space-y-5">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">Utilities</h2>
          <p className="mt-1 text-sm text-zinc-500">
            Workspace diagnostics appear here after you select a Serve workspace.
          </p>
        </div>
      </div>
    );
  }

  const details = data?.workspace_details;
  const doctor = data?.doctor;
  const checks = data?.status_checks ?? [];
  const originLabel = data?.bootstrapped_from_autoclean
    ? "Bootstrapped from an AutoClean workspace"
    : data?.bootstrap_origin === "new_serve_workspace"
      ? "Initialized as a Serve workspace"
      : "Origin unknown";

  return (
    <div className="max-w-6xl space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">Utilities</h2>
          <p className="mt-1 text-sm text-zinc-500">
            Read-only workspace status and doctor diagnostics for the current Serve workspace.
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm font-medium text-zinc-300 transition-colors hover:bg-surface-50 disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {error && <ErrorBanner message={error} />}

      <div className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="rounded-lg border border-border bg-surface-100">
          <div className="flex items-center gap-2 border-b border-border px-5 py-3">
            <FolderOpen className="h-4 w-4 text-brand" />
            <div>
              <h3 className="text-sm font-semibold text-zinc-100">Workspace Status</h3>
              <p className="text-xs text-zinc-500">Current path and required Serve components.</p>
            </div>
          </div>
          <div className="space-y-5 px-5 py-4">
            <div className="rounded-lg border border-border bg-surface-200/60 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-600">
                Selected Workspace
              </p>
              <p className="mt-2 break-all font-mono text-sm text-zinc-200">
                {data?.selected_workspace_path ?? data?.workspace_dir ?? "No workspace selected"}
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                <BooleanPill ok={!!details?.serve_test_exists} text="serve-test.yaml" />
                <BooleanPill ok={!!details?.serve_live_exists} text="serve-live.yaml" />
                <BooleanPill ok={!!details?.deploy_exists} text="deploy/" />
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                <BooleanPill ok={!!details?.runtimes_test_exists} text="runtimes/test" />
                <BooleanPill ok={!!details?.runtimes_live_exists} text="runtimes/live" />
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                <BooleanPill ok={!!details?.test_runtime_ready} text="test runtime ready" />
                <BooleanPill ok={!!details?.live_runtime_ready} text="live runtime ready" />
              </div>
              <p className="mt-3 text-sm text-zinc-400">{originLabel}</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {checks.map((check) => (
                <div
                  key={check.label}
                  className="rounded-lg border border-border bg-surface-200/40 px-4 py-3"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-medium text-zinc-200">{check.label}</p>
                      <p className="mt-1 break-all text-xs text-zinc-500">{check.detail}</p>
                    </div>
                    {check.ok ? (
                      <ShieldCheck className="mt-0.5 h-4 w-4 flex-shrink-0 text-emerald-400" />
                    ) : (
                      <TriangleAlert className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-border bg-surface-100">
          <div className="flex items-center gap-2 border-b border-border px-5 py-3">
            <Wrench className="h-4 w-4 text-brand" />
            <div>
              <h3 className="text-sm font-semibold text-zinc-100">Workspace Doctor</h3>
              <p className="text-xs text-zinc-500">Action-oriented guidance for the selected workspace.</p>
            </div>
          </div>
          <div className="space-y-4 px-5 py-4">
            <div
              className={[
                "rounded-lg border px-4 py-3",
                doctor?.ok
                  ? "border-emerald-500/30 bg-emerald-500/10"
                  : "border-amber-500/30 bg-amber-500/10",
              ].join(" ")}
            >
              <div className="flex items-start gap-2">
                <HeartPulse className={`mt-0.5 h-4 w-4 ${doctor?.ok ? "text-emerald-400" : "text-amber-400"}`} />
                <div>
                  <p className={`text-sm font-semibold ${doctor?.ok ? "text-emerald-400" : "text-amber-400"}`}>
                    {doctor?.summary ?? "Loading workspace diagnostics..."}
                  </p>
                  <p className="mt-1 text-xs text-zinc-400">
                    This mirrors the CLI workspace doctor view in a read-only UI panel.
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-600">
                Blocking Issues
              </h4>
              {doctor?.blocking_issues?.length ? (
                <ul className="mt-2 space-y-2">
                  {doctor.blocking_issues.map((issue) => (
                    <li
                      key={`${issue.label}-${issue.detail}`}
                      className="rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2"
                    >
                      <p className="text-sm font-medium text-red-300">{issue.label}</p>
                      <p className="mt-1 break-all text-xs text-red-200/70">{issue.detail}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-zinc-500">No blocking workspace issues detected.</p>
              )}
            </div>

            <div>
              <h4 className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-600">
                Guidance
              </h4>
              {doctor?.guidance?.length ? (
                <ul className="mt-2 space-y-2">
                  {doctor.guidance.map((item) => (
                    <li
                      key={item}
                      className="rounded-lg border border-border bg-surface-200/40 px-3 py-2 text-sm text-zinc-300"
                    >
                      {item}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-zinc-500">No additional guidance right now.</p>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
