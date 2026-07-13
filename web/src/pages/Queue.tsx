import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { RotateCcw, Trash2, RefreshCw, X, Inbox, FileWarning, Clock, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import type { QueueEntry, QueueStats, RouteSpec } from "../lib/api";
import StatusBadge from "../components/StatusBadge";
import ErrorBanner from "../components/ErrorBanner";
import { useTutorial } from "../contexts/TutorialContext";
import { useTutorialTarget } from "../hooks/useTutorialTarget";
import { relativeTime, formatTime } from "../lib/format";

type StatusFilter = "all" | "pending" | "processing" | "processed" | "failed";

function basename(filepath: string): string {
  return filepath.split(/[/\\]/).pop() || filepath;
}

// ── Stat Card ───────────────────────────────────────────────────

function StatCard({
  label,
  value,
  icon: Icon,
  color,
  active,
  onClick,
}: {
  label: string;
  value: number;
  icon: React.ElementType;
  color: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "flex items-center gap-3 rounded-lg border px-4 py-3 transition-all duration-150 text-left min-w-[130px]",
        active
          ? `border-${color.replace("text-", "")}/40 bg-${color.replace("text-", "")}/5`
          : "border-border bg-surface-100 hover:bg-surface-50/50",
      ].join(" ")}
      style={active ? { borderColor: `var(--color-border)`, background: 'rgba(255,255,255,0.02)' } : {}}
    >
      <Icon className={`w-4 h-4 ${color} flex-shrink-0`} />
      <div>
        <p className={`text-lg font-bold ${active ? color : "text-zinc-200"}`}>{value}</p>
        <p className="text-[11px] text-zinc-500">{label}</p>
      </div>
    </button>
  );
}

// ── Page ─────────────────────────────────────────────────────────

export default function Queue() {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedRoute = searchParams.get("route") || "";
  const taskFilter = searchParams.get("task") || "";
  const montageFilter = searchParams.get("montage") || "";
  const {
    data: entriesData,
    error: entriesError,
    loading,
    refresh,
  } = usePolling(() => api.getQueueEntries(selectedRoute || undefined), 5000);
  const { data: stats, error: statsError, refresh: refreshStats } = usePolling(
    api.getQueueStats,
    5000
  );
  const { data: routes } = usePolling<RouteSpec[]>(api.getRoutes, 30000);
  const { data: status } = usePolling(api.getStatus, 5000);
  const [acting, setActing] = useState(false);
  const [notice, setNotice] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<QueueEntry | null>(null);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");

  // Clear notice timer on unmount
  useEffect(() => {
    return () => {
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    };
  }, []);

  // Tutorial integration
  const { isActive, currentStep, nextStep } = useTutorial();
  const queueTableRef = useTutorialTarget("queue-table");

  // Tutorial step 6: watch for a processed entry to advance to complete
  useEffect(() => {
    if (!isActive || currentStep !== 6) return;
    const entries = entriesData?.entries ?? [];
    const hasProcessed = entries.some((e) => e.status === "processed");
    if (hasProcessed) {
      nextStep();
    }
  }, [isActive, currentStep, entriesData, nextStep]);

  const entries = entriesData?.entries || [];
  const filtered = statusFilter === "all"
    ? entries
    : entries.filter((e) => e.status === statusFilter);
  const routeOptions = (routes ?? []).filter((route) => {
    const taskName = route.taskfile.split("/").pop()?.replace(".py", "") || route.taskfile;
    if (taskFilter && taskName !== taskFilter) return false;
    if (montageFilter && route.montage !== montageFilter) return false;
    return true;
  });
  const availableTasks = [...new Set((routes ?? []).map((route) => route.taskfile.split("/").pop()?.replace(".py", "") || route.taskfile))].sort();
  const availableMontages = [...new Set((routes ?? []).map((route) => route.montage))].sort();

  const updateContextParam = (key: "route" | "task" | "montage", value: string) => {
    const next = new URLSearchParams(searchParams);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== "route") next.delete("route");
    setSearchParams(next, { replace: true });
    setSelectedEntry(null);
  };

  const showNotice = (type: "success" | "error", text: string) => {
    if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    setNotice({ type, text });
    noticeTimerRef.current = setTimeout(() => setNotice(null), 4000);
  };

  const handleRetry = async () => {
    setActing(true);
    try {
      const res = await api.retryFailed();
      showNotice("success", `Retried ${res.retried} ${res.retried === 1 ? "entry" : "entries"}`);
      refresh();
      refreshStats();
    } catch (err) {
      showNotice("error", err instanceof Error ? err.message : "Retry failed");
    } finally {
      setActing(false);
    }
  };

  const handleClear = async () => {
    setActing(true);
    try {
      const res = await api.clearProcessed();
      showNotice("success", `Cleared ${res.cleared} ${res.cleared === 1 ? "entry" : "entries"}`);
      refresh();
      refreshStats();
    } catch (err) {
      showNotice("error", err instanceof Error ? err.message : "Clear failed");
    } finally {
      setActing(false);
    }
  };

  const handleRemove = async (path: string) => {
    setActing(true);
    try {
      await api.removeEntry(path);
      showNotice("success", `Removed ${basename(path)}`);
      setSelectedEntry(null);
      refresh();
      refreshStats();
    } catch (err) {
      showNotice("error", err instanceof Error ? err.message : "Remove failed");
    } finally {
      setActing(false);
    }
  };

  const toggleFilter = (f: StatusFilter) => {
    setStatusFilter((prev) => (prev === f ? "all" : f));
    setSelectedEntry(null);
  };

  // ── Render ─────────────────────────────────────────────────

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">Queue</h2>
          <p className="mt-0.5 text-xs text-zinc-500">
            Global processing view with optional route focus
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { refresh(); refreshStats(); }}
            className="rounded-md p-1.5 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50 transition-colors duration-150"
            title="Refresh"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
          <button
            onClick={handleRetry}
            disabled={acting || !stats?.failed}
            className="rounded-md px-3 py-1.5 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50 transition-colors duration-150 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Retry Failed
          </button>
          <button
            onClick={handleClear}
            disabled={acting || !stats?.processed}
            className="rounded-md px-3 py-1.5 text-sm font-medium border border-red-500/30 text-red-400 hover:bg-red-500/10 transition-colors duration-150 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Done
          </button>
        </div>
      </div>

      {(entriesError || statsError) && <ErrorBanner message={entriesError || statsError!} />}

      {status?.configured && status.operational_state !== "ready" && (
        <div className="rounded-lg border border-border bg-surface-100 px-5 py-3 text-sm text-zinc-400">
          <span className="font-medium text-zinc-200">Current state:</span>
          <span className="ml-2">{status.next_step || "Serve is not fully operational yet."}</span>
        </div>
      )}

      <div className="grid gap-3 rounded-lg border border-border bg-surface-100 p-4 xl:grid-cols-[minmax(0,1fr)_14rem_14rem_18rem]">
        <div>
          <p className="text-xs font-medium text-zinc-300">
            {selectedRoute
              ? `Showing queue entries for route '${selectedRoute}'`
              : "Showing queue entries across all routes"}
          </p>
          <p className="mt-1 text-xs text-zinc-500">
            Queue stays global. Use the route filter when you want to inspect one route's processing flow.
          </p>
        </div>
        <label className="block text-[11px] uppercase tracking-wider text-zinc-500">
          Task Filter
          <select
            value={taskFilter}
            onChange={(event) => updateContextParam("task", event.target.value)}
            className="mt-2 w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
          >
            <option value="">All tasks</option>
            {availableTasks.map((task) => (
              <option key={task} value={task}>{task}</option>
            ))}
          </select>
        </label>
        <label className="block text-[11px] uppercase tracking-wider text-zinc-500">
          Montage Filter
          <select
            value={montageFilter}
            onChange={(event) => updateContextParam("montage", event.target.value)}
            className="mt-2 w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
          >
            <option value="">All montages</option>
            {availableMontages.map((montage) => (
              <option key={montage} value={montage}>{montage}</option>
            ))}
          </select>
        </label>
        <label className="block text-[11px] uppercase tracking-wider text-zinc-500">
          Route Filter
          <select
            value={selectedRoute}
            onChange={(event) => updateContextParam("route", event.target.value)}
            className="mt-2 w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
          >
            <option value="">All routes</option>
            {routeOptions.map((route) => (
              <option key={route.id} value={route.id}>
                {route.id} · {route.taskfile.split("/").pop()?.replace(".py", "") || route.taskfile} · {route.montage}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Notice */}
      {notice && (
        <div
          className={`rounded-lg px-4 py-2 text-sm font-medium flex items-center justify-between ${
            notice.type === "success"
              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
              : "bg-red-500/10 text-red-400 border border-red-500/30"
          }`}
        >
          {notice.text}
          <button onClick={() => setNotice(null)} className="opacity-60 hover:opacity-100">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Stat cards (clickable filters) */}
      {stats && (
        <div className="flex flex-wrap gap-3">
          <StatCard
            label="Pending"
            value={stats.pending}
            icon={Clock}
            color="text-amber-400"
            active={statusFilter === "pending"}
            onClick={() => toggleFilter("pending")}
          />
          <StatCard
            label="Processing"
            value={stats.processing}
            icon={Loader2}
            color="text-amber-400"
            active={statusFilter === "processing"}
            onClick={() => toggleFilter("processing")}
          />
          <StatCard
            label="Processed"
            value={stats.processed}
            icon={CheckCircle2}
            color="text-brand"
            active={statusFilter === "processed"}
            onClick={() => toggleFilter("processed")}
          />
          <StatCard
            label="Failed"
            value={stats.failed}
            icon={AlertCircle}
            color="text-red-400"
            active={statusFilter === "failed"}
            onClick={() => toggleFilter("failed")}
          />
          {statusFilter !== "all" && (
            <button
              onClick={() => { setStatusFilter("all"); setSelectedEntry(null); }}
              className="self-center text-xs text-zinc-500 hover:text-zinc-300 underline underline-offset-2"
            >
              Show all
            </button>
          )}
        </div>
      )}

      {/* Main content: table + detail panel */}
      <div className="flex flex-col lg:flex-row gap-5">
        {/* Table */}
        <div ref={queueTableRef} className="flex-1 min-w-0 rounded-lg border border-border bg-surface-100 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-surface-100 border-b border-border">
                <th className="px-4 py-2.5 text-left text-xs uppercase text-zinc-500 font-medium tracking-wider">File</th>
                <th className="px-4 py-2.5 text-left text-xs uppercase text-zinc-500 font-medium tracking-wider">Status</th>
                <th className="px-4 py-2.5 text-left text-xs uppercase text-zinc-500 font-medium tracking-wider">Route</th>
                <th className="px-4 py-2.5 text-left text-xs uppercase text-zinc-500 font-medium tracking-wider">Time</th>
              </tr>
            </thead>
            <tbody>
              {loading && entries.length === 0 ? (
                Array.from({ length: 4 }).map((_, i) => (
                  <tr key={i} className="border-b border-border-subtle">
                    {Array.from({ length: 4 }).map((_, j) => (
                      <td key={j} className="px-4 py-3">
                        <div className="h-4 w-3/4 rounded bg-surface-50 animate-pulse" />
                      </td>
                    ))}
                  </tr>
                ))
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-4 py-12 text-center">
                    <div className="flex flex-col items-center gap-2 text-zinc-500">
                      {entries.length === 0 ? (
                        <>
                          <Inbox className="w-8 h-8 text-zinc-700" />
                          <p className="text-sm">Queue is empty</p>
                          <p className="text-xs text-zinc-600">
                            {selectedRoute
                              ? "No files are currently queued for the selected route"
                              : status?.service?.running
                                ? "Files will appear here when matching routes discover work."
                                : "Files will appear here when the processing service is running"}
                          </p>
                        </>
                      ) : (
                        <>
                          <FileWarning className="w-8 h-8 text-zinc-700" />
                          <p className="text-sm">No {statusFilter} entries</p>
                          <button
                            onClick={() => setStatusFilter("all")}
                            className="text-xs text-brand hover:underline"
                          >
                            Show all entries
                          </button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ) : (
                filtered.map((entry) => (
                  <tr
                    key={entry.path}
                    onClick={() => setSelectedEntry((prev) => prev?.path === entry.path ? null : entry)}
                    className={[
                      "border-b border-border-subtle transition-colors duration-150 cursor-pointer",
                      selectedEntry?.path === entry.path
                        ? "bg-brand/15 dark:bg-brand/10"
                        : "hover:bg-surface-50/30",
                      entry.status === "failed" ? "border-l-2 border-l-red-500/40" : "",
                    ].join(" ")}
                  >
                    <td className="px-4 py-3">
                      <span className="font-mono text-xs text-zinc-200" title={entry.path}>
                        {basename(entry.path)}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge
                        status={entry.status as "pending" | "processing" | "processed" | "failed"}
                      />
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-zinc-400 text-xs">{entry.route_id || "--"}</span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-zinc-500 text-xs" title={entry.added_at ? formatTime(entry.added_at) : ""}>
                        {entry.added_at ? relativeTime(entry.added_at) : "--"}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          {/* Footer count */}
          {filtered.length > 0 && (
            <div className="px-4 py-2 border-t border-border-subtle text-xs text-zinc-600">
              {statusFilter !== "all"
                ? `${filtered.length} of ${entries.length} entries`
                : `${entries.length} entries`}
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selectedEntry && (
          <div className="lg:w-80 flex-shrink-0 rounded-lg border border-border bg-surface-100 p-5 space-y-4 self-start lg:sticky lg:top-5">
            {/* Header */}
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-100">Details</h3>
              <button
                onClick={() => setSelectedEntry(null)}
                className="text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* File name */}
            <div>
              <p className="text-xs text-zinc-500 mb-1">File</p>
              <p className="text-sm text-zinc-200 font-mono break-all">
                {basename(selectedEntry.path)}
              </p>
            </div>

            {/* Full path */}
            <div>
              <p className="text-xs text-zinc-500 mb-1">Full Path</p>
              <p className="text-xs text-zinc-400 font-mono break-all bg-surface-50 rounded px-2 py-1.5">
                {selectedEntry.path}
              </p>
            </div>

            {/* Status + Route row */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-zinc-500 mb-1">Status</p>
                <StatusBadge
                  status={selectedEntry.status as "pending" | "processing" | "processed" | "failed"}
                />
              </div>
              <div>
                <p className="text-xs text-zinc-500 mb-1">Route</p>
                <p className="text-sm text-zinc-200">{selectedEntry.route_id || "--"}</p>
              </div>
            </div>

            {/* Timestamps */}
            <div className="space-y-1.5">
              {selectedEntry.added_at && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500">Added</span>
                  <span className="text-zinc-300">{formatTime(selectedEntry.added_at)}</span>
                </div>
              )}
              {selectedEntry.processed_at && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500">Processed</span>
                  <span className="text-zinc-300">{formatTime(selectedEntry.processed_at)}</span>
                </div>
              )}
              {selectedEntry.failed_at && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500">Failed</span>
                  <span className="text-zinc-300">{formatTime(selectedEntry.failed_at)}</span>
                </div>
              )}
            </div>

            {/* Error block */}
            {selectedEntry.status === "failed" && selectedEntry.last_error && (
              <div>
                <p className="text-xs text-zinc-500 mb-1">Error</p>
                <div className="bg-red-500/5 border border-red-500/20 rounded-md p-3 font-mono text-xs text-red-300 whitespace-pre-wrap break-all max-h-48 overflow-y-auto">
                  {selectedEntry.last_error}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="pt-2 border-t border-border-subtle space-y-2">
              {selectedEntry.status === "failed" && (
                <button
                  onClick={handleRetry}
                  disabled={acting}
                  className="w-full rounded-md px-3 py-1.5 text-sm font-medium bg-emerald-600 text-white hover:bg-emerald-500 transition-colors duration-150 flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Retry Failed
                </button>
              )}
              <button
                onClick={() => handleRemove(selectedEntry.path)}
                disabled={acting}
                className="w-full rounded-md px-3 py-1.5 text-sm font-medium border border-red-500/30 text-red-400 hover:bg-red-500/10 transition-colors duration-150 flex items-center justify-center gap-2 disabled:opacity-50"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Remove
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
