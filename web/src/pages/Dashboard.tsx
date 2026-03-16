import { RefreshCw, AlertTriangle, Info, GitBranch, ListOrdered, Settings, Play, FolderInput, FolderOutput, GraduationCap, FolderOpen } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import type { DashboardStatus } from "../lib/api";
import StatusBadge from "../components/StatusBadge";
import ErrorBanner from "../components/ErrorBanner";
import { useNavigate } from "react-router-dom";
import { useTutorial } from "../contexts/TutorialContext";
import { useTutorialTarget } from "../hooks/useTutorialTarget";
import { formatUptime } from "../lib/format";

function getRecommendation(data: DashboardStatus) {
  if (data.routes.total === 0)
    return { severity: "info" as const, title: "Create the first route", description: "No routes exist yet. Start by mapping an input folder to a task and montage.", action: "Open Routes", target: "/routes" };
  if (data.config.errors.length > 0)
    return { severity: "error" as const, title: "Fix configuration errors", description: "Settings have validation errors that must be resolved before processing.", action: "Open Settings", target: "/settings" };
  if (data.queue.failed > 0)
    return { severity: "warning" as const, title: "Review failed jobs", description: `${data.queue.failed} file(s) failed processing. Check errors and retry or clear them.`, action: "Open Queue", target: "/queue" };
  if (data.config.needs_deploy)
    return { severity: "warning" as const, title: "Apply your changes", description: "Settings have changed since last applied. Apply to use the updated configuration.", action: "Open Settings", target: "/settings" };
  if (!data.service.running)
    return { severity: "info" as const, title: "Start the service", description: "Routes and settings look ready, but the processing service is not running.", action: "Start Service", target: "/service" };
  if (data.queue.pending > 0 || data.queue.processing > 0)
    return { severity: "info" as const, title: "Processing in progress", description: `${data.queue.pending + data.queue.processing} files in flight. Watch for stuck or failing jobs.`, action: "Open Queue", target: "/queue" };
  return null;
}

export default function Dashboard() {
  const { data, error, loading, refresh } = usePolling<DashboardStatus>(
    api.getStatus,
    5000
  );
  const navigate = useNavigate();
  const recommendation = data ? getRecommendation(data) : null;
  const { startTutorial, completed, isActive } = useTutorial();
  const statsRef = useTutorialTarget("dashboard-stats");

  const severityStyles = {
    error: { border: "border-red-500/40", icon: <AlertTriangle className="w-5 h-5 text-red-400" /> },
    warning: { border: "border-amber-500/40", icon: <AlertTriangle className="w-5 h-5 text-amber-400" /> },
    info: { border: "border-cyan-500/40", icon: <Info className="w-5 h-5 text-cyan-400" /> },
  };

  const showGettingStarted = data && data.routes.total === 0 && !completed && !isActive;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-zinc-100">Dashboard</h2>
        <button
          onClick={refresh}
          className="rounded-md p-1.5 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50"
          title="Refresh"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {error && <ErrorBanner message={error} />}

      {/* No workspace configured card */}
      {data && !data.configured && (
        <div className="rounded-lg border border-brand/30 bg-brand/5 p-5">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-brand/15 flex items-center justify-center flex-shrink-0">
              <FolderOpen className="w-4.5 h-4.5 text-brand" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-zinc-100 mb-1">
                No workspace configured
              </h3>
              <p className="text-sm text-zinc-400 mb-3">
                Choose a workspace folder to start managing routes, processing queues, and EEG files.
              </p>
              <button
                onClick={() => navigate("/setup")}
                className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500"
              >
                Choose Workspace
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Getting Started card (shown when no routes and tutorial not completed) */}
      {showGettingStarted && (
        <div className="rounded-lg border border-brand/30 bg-brand/5 p-5">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-brand/15 flex items-center justify-center flex-shrink-0">
              <GraduationCap className="w-4.5 h-4.5 text-brand" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-zinc-100 mb-1">
                Getting Started
              </h3>
              <p className="text-sm text-zinc-400 mb-3">
                Take the 2-minute tutorial to set up your first processing route, apply the config, and see a file processed end-to-end.
              </p>
              <button
                onClick={() => startTutorial()}
                className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500"
              >
                Start Tutorial
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Stat cards — clickable */}
      <div ref={statsRef} className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Routes */}
        <button
          onClick={() => navigate("/routes")}
          className="rounded-lg border border-border bg-surface-100 p-5 text-left hover:bg-surface-50/30 transition-colors group"
        >
          <div className="flex items-center gap-2 mb-2">
            <GitBranch className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400 transition-colors" />
            <p className="text-sm text-zinc-500">Routes</p>
          </div>
          <p className="text-2xl font-bold text-zinc-100">
            {data?.routes.active ?? "--"}
          </p>
          <p className="text-xs text-zinc-600 mt-1">
            {data
              ? `${data.routes.active} active${data.routes.archived > 0 ? `, ${data.routes.archived} archived` : ""}`
              : ""}
          </p>
        </button>

        {/* Queue */}
        <button
          onClick={() => navigate("/queue")}
          className="rounded-lg border border-border bg-surface-100 p-5 text-left hover:bg-surface-50/30 transition-colors group"
        >
          <div className="flex items-center gap-2 mb-2">
            <ListOrdered className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400 transition-colors" />
            <p className="text-sm text-zinc-500">Queue</p>
          </div>
          <p className="text-2xl font-bold text-zinc-100">
            {data?.queue.total ?? "--"}
          </p>
          {data && data.queue.total > 0 && (
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
              {data.queue.pending > 0 && (
                <span className="text-xs"><span className="text-amber-400 font-medium">{data.queue.pending}</span> <span className="text-zinc-600">pending</span></span>
              )}
              {data.queue.processing > 0 && (
                <span className="text-xs"><span className="text-amber-400 font-medium">{data.queue.processing}</span> <span className="text-zinc-600">active</span></span>
              )}
              {data.queue.failed > 0 && (
                <span className="text-xs"><span className="text-red-400 font-medium">{data.queue.failed}</span> <span className="text-zinc-600">failed</span></span>
              )}
              {data.queue.processed > 0 && (
                <span className="text-xs"><span className="text-brand font-medium">{data.queue.processed}</span> <span className="text-zinc-600">done</span></span>
              )}
            </div>
          )}
        </button>

        {/* Settings */}
        <button
          onClick={() => navigate("/settings")}
          className="rounded-lg border border-border bg-surface-100 p-5 text-left hover:bg-surface-50/30 transition-colors group"
        >
          <div className="flex items-center gap-2 mb-2">
            <Settings className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400 transition-colors" />
            <p className="text-sm text-zinc-500">Settings</p>
          </div>
          <div className="mt-1">
            {data ? (
              data.config.errors.length > 0 ? (
                <StatusBadge status="invalid" label="Has Errors" />
              ) : !data.config.needs_deploy ? (
                <StatusBadge status="valid" label="Applied" />
              ) : (
                <StatusBadge status="pending" label="Unapplied Changes" />
              )
            ) : (
              <p className="text-2xl font-bold text-zinc-100">--</p>
            )}
          </div>
          {data && (
            <p className="text-xs text-zinc-600 mt-2">
              {data.config.source}{data.config.errors.length > 0 ? ` · ${data.config.errors.length} error(s)` : ""}
            </p>
          )}
        </button>

        {/* Service */}
        <button
          onClick={() => navigate("/service")}
          className="rounded-lg border border-border bg-surface-100 p-5 text-left hover:bg-surface-50/30 transition-colors group"
        >
          <div className="flex items-center gap-2 mb-2">
            <Play className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400 transition-colors" />
            <p className="text-sm text-zinc-500">Service</p>
          </div>
          <div className="mt-1">
            {data ? (
              <StatusBadge
                status={data.service.running ? "running" : "stopped"}
                label={data.service.running ? "Running" : "Stopped"}
              />
            ) : (
              <p className="text-2xl font-bold text-zinc-100">--</p>
            )}
          </div>
          <p className="text-xs text-zinc-600 mt-2">
            {data?.service.running
              ? `PID ${data.service.pid} · ${formatUptime(data.service.uptime_seconds)}`
              : data ? "Not running" : ""}
          </p>
        </button>
      </div>

      {/* Recommendation */}
      {recommendation && (
        <div className={`rounded-lg border ${severityStyles[recommendation.severity].border} bg-surface-100 p-5`}>
          <div className="flex items-start gap-3">
            {severityStyles[recommendation.severity].icon}
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-zinc-100 mb-1">
                {recommendation.title}
              </h3>
              <p className="text-sm text-zinc-400 mb-3">
                {recommendation.description}
              </p>
              <button
                onClick={() => navigate(recommendation.target)}
                className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500"
              >
                {recommendation.action}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Workspace info bar */}
      {data && data.configured && (
        <div className="rounded-lg border border-border bg-surface-100 px-5 py-3">
          <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-6 text-sm">
            <div className="flex items-center gap-2 min-w-0">
              <FolderInput className="w-3.5 h-3.5 text-zinc-600 flex-shrink-0" />
              <span className="text-zinc-500 flex-shrink-0">Workspace</span>
              <code className="text-zinc-400 font-mono text-xs truncate" title={data.workspace_dir}>
                {data.workspace_dir}
              </code>
            </div>
            <div className="hidden sm:block h-4 w-px bg-border" />
            <div className="flex items-center gap-2 min-w-0">
              <FolderOutput className="w-3.5 h-3.5 text-zinc-600 flex-shrink-0" />
              <span className="text-zinc-500 flex-shrink-0">Output</span>
              <code className="text-zinc-400 font-mono text-xs truncate" title={data.output_dir}>
                {data.output_dir}
              </code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
