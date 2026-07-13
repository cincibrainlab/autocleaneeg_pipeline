import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Cpu,
  Search,
  ChevronDown,
  ChevronUp,
  Gauge,
  Filter,
  Brain,
  Clock,
  CheckCircle2,
  Code2,
  Workflow,
  Plus,
  RefreshCw,
  Download,
  Trash2,
  ArrowUpCircle,
  Loader2,
  X,
  ArrowUpDown,
  Github,
} from "lucide-react";
import { api } from "../lib/api";
import type { ManagedTask, TaskSyncStatus, TaskManagerResponse } from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";
import ConfirmDialog from "../components/ConfirmDialog";
import CodeViewer from "../components/CodeViewer";
import { usePolling } from "../hooks/usePolling";

import { relativeTime } from "../lib/format";

// ── Helpers ──────────────────────────────────────────────────────

const STATUS_STYLES: Record<TaskSyncStatus, string> = {
  installed: "bg-emerald-500/15 text-emerald-400",
  modified: "bg-amber-500/15 text-amber-400",
  not_installed: "bg-zinc-500/15 text-zinc-400",
  workspace_only: "bg-cyan-500/15 text-cyan-400",
};
const STATUS_LABELS: Record<TaskSyncStatus, string> = {
  installed: "Installed",
  modified: "Modified",
  not_installed: "Available",
  workspace_only: "Custom",
};

const CATEGORY_COLORS: Record<string, string> = {
  resting: "text-violet-400",
  auditory: "text-amber-400",
  motor: "text-blue-400",
  cognitive: "text-rose-400",
  clinical: "text-orange-400",
  custom: "text-cyan-400",
  builtin: "text-zinc-400",
};

type SortKey = "name" | "category" | "source" | "status" | "montage" | "sample_rate" | "ica_method";
type SortDir = "asc" | "desc";
type ViewFilter = "all" | "mine" | "available";

function sortTasks(tasks: ManagedTask[], key: SortKey, dir: SortDir): ManagedTask[] {
  const mult = dir === "asc" ? 1 : -1;
  return [...tasks].sort((a, b) => {
    let va = "", vb = "";
    switch (key) {
      case "name": va = a.name; vb = b.name; break;
      case "category": va = a.category; vb = b.category; break;
      case "source": va = a.source; vb = b.source; break;
      case "status": va = a.sync_status; vb = b.sync_status; break;
      case "montage": va = a.config?.montage ?? ""; vb = b.config?.montage ?? ""; break;
      case "sample_rate": va = String(a.config?.sample_rate ?? 0); vb = String(b.config?.sample_rate ?? 0); break;
      case "ica_method": va = a.config?.ica_method ?? ""; vb = b.config?.ica_method ?? ""; break;
    }
    return va.localeCompare(vb, undefined, { numeric: true }) * mult;
  });
}

// ── Pipeline visualization ───────────────────────────────────────

function PipelineView({ steps }: { steps: string[] }) {
  if (steps.length === 0) return <p className="text-xs text-zinc-600 italic py-2">Pipeline steps not available.</p>;
  return (
    <div className="pl-2 border-l-2 border-border-subtle ml-1 space-y-0">
      {steps.map((step, i) => {
        const isLast = i === steps.length - 1;
        const parenIdx = step.indexOf(" (");
        const stepName = parenIdx > -1 ? step.slice(0, parenIdx) : step;
        const stepParam = parenIdx > -1 ? step.slice(parenIdx + 1) : null;
        return (
          <div key={i} className="flex items-start gap-2 relative">
            <div className="flex flex-col items-center w-4 flex-shrink-0 mt-0.5">
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${isLast ? "bg-brand" : "bg-brand/60"}`} />
              {!isLast && <div className="w-px bg-zinc-700 flex-grow mt-0.5 min-h-[12px]" />}
            </div>
            <div className="pb-2.5">
              <span className="text-sm text-zinc-200">{stepName}</span>
              {stepParam && <span className="text-xs text-zinc-500 font-mono ml-1.5">{stepParam}</span>}
            </div>
          </div>
        );
      })}
      <div className="flex items-center gap-2">
        <div className="w-4 flex justify-center flex-shrink-0"><CheckCircle2 className="w-3.5 h-3.5 text-brand" /></div>
        <span className="text-xs text-zinc-600 font-medium pb-1">Done</span>
      </div>
    </div>
  );
}

// ── Python syntax highlighting ────────────────────────────────────

const PY_KEYWORDS = new Set([
  "def","class","return","if","elif","else","for","while","import",
  "from","as","with","try","except","finally","raise","pass","break",
  "continue","and","or","not","in","is","None","True","False",
  "self","yield","async","await","lambda",
]);

function colorizePython(line: string): React.ReactNode {
  if (/^\s*#/.test(line)) return <span className="text-zinc-600">{line}</span>;
  if (/^\s*@/.test(line)) return <span className="text-amber-400">{line}</span>;
  const trimmed = line.trim();
  if (trimmed.startsWith('"""') || trimmed.startsWith("'''") || trimmed.startsWith('"') || trimmed.startsWith("'"))
    return <span className="text-emerald-400">{line}</span>;
  const parts: React.ReactNode[] = [];
  const tokens = line.split(/(\b)/);
  let hasComment = false;
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i] ?? "";
    const prev = tokens[i - 1] ?? "";
    const prev2 = tokens[i - 2] ?? "";
    if (hasComment) { parts.push(<span key={i} className="text-zinc-600">{tok}</span>); continue; }
    if (tok === "#") { hasComment = true; parts.push(<span key={i} className="text-zinc-600">{tok}</span>); continue; }
    if (PY_KEYWORDS.has(tok)) parts.push(<span key={i} className="text-purple-400">{tok}</span>);
    else if (/^\d+(\.\d+)?$/.test(tok)) parts.push(<span key={i} className="text-amber-400">{tok}</span>);
    else if (prev === "def" || prev2 === "def") parts.push(<span key={i} className="text-cyan-400">{tok}</span>);
    else parts.push(<span key={i} className="text-zinc-300">{tok}</span>);
  }
  return <>{parts}</>;
}

// ── Create Task modal ────────────────────────────────────────────

function CreateTaskModal({ open, onClose, onCreate, creating, error }: {
  open: boolean; onClose: () => void; onCreate: (n: string) => Promise<void>; creating: boolean; error: string | null;
}) {
  const [className, setClassName] = useState("");
  const [valErr, setValErr] = useState<string | null>(null);
  const ok = /^[A-Za-z_][A-Za-z0-9_]*$/.test(className) && className.length > 0;
  const handleCreate = async () => {
    if (!ok) { setValErr("Must be a valid Python identifier."); return; }
    setValErr(null);
    await onCreate(className);
    setClassName("");
  };
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div className="w-full max-w-md rounded-lg border border-border bg-surface-200 p-6" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-zinc-100">New Task</h3>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300"><X className="w-4 h-4" /></button>
        </div>
        <p className="text-xs text-zinc-500 mb-4">Enter a Python class name. A starter file will be created in your workspace.</p>
        <input type="text" value={className} onChange={(e) => { setClassName(e.target.value); setValErr(null); }}
          onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); if (e.key === "Escape") onClose(); }}
          placeholder="e.g. MyRestingStateTask" autoFocus
          className="w-full px-3 py-2 text-sm bg-surface-50 border border-border rounded text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-brand/60 mb-1" />
        {(valErr || error) && <p className="text-xs text-red-400 mb-3">{valErr ?? error}</p>}
        {!valErr && !error && <p className="text-xs text-zinc-600 mb-3">{className ? ok ? `Will create: ${className}.py` : "Invalid" : ""}</p>}
        <div className="flex justify-end gap-3 mt-4">
          <button onClick={onClose} className="rounded-md px-4 py-2 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50">Cancel</button>
          <button disabled={creating || !className} onClick={handleCreate}
            className="flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 disabled:opacity-50">
            {creating && <Loader2 className="w-3.5 h-3.5 animate-spin" />} Create Task
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Sortable column header ───────────────────────────────────────

function SortHeader({ label, sortKey, currentKey, dir, onSort, className: cls }: {
  label: string; sortKey: SortKey; currentKey: SortKey; dir: SortDir; onSort: (k: SortKey) => void; className?: string;
}) {
  const active = currentKey === sortKey;
  return (
    <th className={`px-3 py-2 text-left text-[10px] uppercase text-zinc-500 font-medium tracking-wider cursor-pointer select-none hover:text-zinc-300 transition-colors ${cls ?? ""}`}
      onClick={() => onSort(sortKey)}>
      <span className="flex items-center gap-1">
        {label}
        {active ? (dir === "asc" ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)
          : <ArrowUpDown className="w-3 h-3 opacity-30" />}
      </span>
    </th>
  );
}

// ── Main page ────────────────────────────────────────────────────

export default function TasksPage() {
  const { data: managerData, error, loading, refresh } = usePolling<TaskManagerResponse>(api.getTaskManager, 60000);
  const tasks = managerData?.tasks ?? null;
  const registryInfo = managerData?.registry_status ?? null;

  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");
  const [viewFilter, setViewFilter] = useState<ViewFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [selected, setSelected] = useState<ManagedTask | null>(null);
  const [detailTab, setDetailTab] = useState<"pipeline" | "source">("pipeline");

  const [actionInFlight, setActionInFlight] = useState<string | null>(null);
  const [refreshingLibrary, setRefreshingLibrary] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [confirmRemove, setConfirmRemove] = useState<string | null>(null);
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear notice timer on unmount
  useEffect(() => {
    return () => {
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    };
  }, []);

  const categories = useMemo(() => (tasks ? [...new Set(tasks.map(t => t.category.toLowerCase()))].sort() : []), [tasks]);

  const filtered = useMemo(() => {
    if (!tasks) return [];
    let list = tasks;
    if (viewFilter === "mine") list = list.filter(t => ["installed", "modified", "workspace_only"].includes(t.sync_status));
    if (viewFilter === "available") list = list.filter(t => t.sync_status === "not_installed");
    if (activeCategory !== "all") list = list.filter(t => t.category.toLowerCase() === activeCategory);
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      list = list.filter(t => t.name.toLowerCase().includes(q) || t.description.toLowerCase().includes(q) || t.category.toLowerCase().includes(q) || (t.config?.montage ?? "").toLowerCase().includes(q));
    }
    return sortTasks(list, sortKey, sortDir);
  }, [tasks, viewFilter, activeCategory, searchQuery, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  };

  const showNotice = useCallback((msg: string) => {
    if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    setNotice(msg);
    noticeTimerRef.current = setTimeout(() => setNotice(null), 5000);
  }, []);

  const handleInstall = useCallback(async (name: string) => {
    setActionInFlight(name); setActionError(null);
    try { const r = await api.installTask(name); r.success ? showNotice(r.message) : setActionError(r.message); refresh(); }
    catch (e) { setActionError(e instanceof Error ? e.message : String(e)); }
    finally { setActionInFlight(null); }
  }, [refresh, showNotice]);

  const handleUpdate = useCallback(async (name: string) => {
    setActionInFlight(name); setActionError(null);
    try { const r = await api.updateTask(name); r.success ? showNotice(r.message) : setActionError(r.message); refresh(); }
    catch (e) { setActionError(e instanceof Error ? e.message : String(e)); }
    finally { setActionInFlight(null); }
  }, [refresh, showNotice]);

  const handleRemoveConfirmed = useCallback(async () => {
    const name = confirmRemove; setConfirmRemove(null); if (!name) return;
    setActionInFlight(name); setActionError(null);
    try { const r = await api.removeTask(name); r.success ? showNotice(r.message) : setActionError(r.message); refresh(); setSelected(null); }
    catch (e) { setActionError(e instanceof Error ? e.message : String(e)); }
    finally { setActionInFlight(null); }
  }, [confirmRemove, refresh, showNotice]);

  const handleCreate = useCallback(async (cn: string) => {
    setCreating(true); setCreateError(null);
    try { const r = await api.createTask(cn); r.success ? (setShowCreateModal(false), showNotice(r.message), refresh()) : setCreateError(r.message); }
    catch (e) { setCreateError(e instanceof Error ? e.message : String(e)); }
    finally { setCreating(false); }
  }, [refresh, showNotice]);

  const handleRefreshLibrary = useCallback(async () => {
    setRefreshingLibrary(true); setActionError(null);
    try { const r = await api.refreshLibrary(); showNotice(r.message); refresh(); }
    catch (e) { setActionError(e instanceof Error ? e.message : String(e)); }
    finally { setRefreshingLibrary(false); }
  }, [refresh, showNotice]);

  const installedCount = tasks ? tasks.filter(t => ["installed", "modified", "workspace_only"].includes(t.sync_status)).length : 0;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">Task Manager</h2>
          <p className="text-xs text-zinc-500 mt-0.5">Browse registry-backed tasks and manage workspace installs</p>
          {registryInfo && (
            <p className="text-xs text-zinc-600 mt-1">
              Registry: synced {relativeTime(registryInfo.synced_at)} · {registryInfo.task_count} in library · {installedCount} installed
            </p>
          )}
          <p className="mt-2 max-w-2xl text-xs text-zinc-500">
            Tasks are tied to the shared repository-backed registry. Use in-app actions for install, update, removal, and workspace-local creation, but treat source-of-truth task maintenance as a repository workflow.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <a
            href="https://github.com/cincibrainlab/autoclean_pipeline/tree/main/src/autoclean/tasks"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-surface-50"
          >
            <Github className="h-3.5 w-3.5" />
            Task Registry
          </a>
          <button onClick={handleRefreshLibrary} disabled={refreshingLibrary} title="Refresh library"
            className="flex items-center justify-center w-8 h-8 rounded border border-border text-zinc-400 hover:text-zinc-200 disabled:opacity-50 transition-colors">
            <RefreshCw className={`w-3.5 h-3.5 ${refreshingLibrary ? "animate-spin" : ""}`} />
          </button>
          <button onClick={() => { setCreateError(null); setShowCreateModal(true); }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-brand/40 bg-brand/10 text-brand text-xs font-medium hover:bg-brand/15 transition-colors">
            <Plus className="w-3.5 h-3.5" /> New Task
          </button>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}
      {actionError && <ErrorBanner message={actionError} onDismiss={() => setActionError(null)} />}
      {notice && (
        <div className="rounded-lg px-4 py-2 text-sm font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center justify-between">
          {notice}
          <button onClick={() => setNotice(null)} className="text-emerald-400/60 hover:text-emerald-400 ml-4"><X className="w-3.5 h-3.5" /></button>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-2">
        <div className="flex items-center gap-1">
          {(["all", "mine", "available"] as ViewFilter[]).map(v => (
            <button key={v} onClick={() => setViewFilter(v)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${viewFilter === v ? "bg-brand/15 text-brand border border-brand/30" : "text-zinc-400 hover:text-zinc-200 border border-transparent hover:border-border"}`}>
              {v === "mine" ? "My Tasks" : v === "available" ? "Available" : "All"}
            </button>
          ))}
        </div>
        <div className="relative flex-1 min-w-0 max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-600" />
          <input type="text" placeholder="Search name, montage, category..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-8 py-1.5 text-sm bg-surface-50 border border-border rounded text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-brand/60" />
          {searchQuery && <button onClick={() => setSearchQuery("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"><X className="w-3.5 h-3.5" /></button>}
        </div>
        <div className="flex items-center gap-1 flex-wrap">
          <button onClick={() => setActiveCategory("all")}
            className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${activeCategory === "all" ? "bg-surface-50 text-zinc-200" : "text-zinc-500 hover:text-zinc-300"}`}>All</button>
          {categories.map(cat => (
            <button key={cat} onClick={() => setActiveCategory(cat)}
              className={`px-2 py-1 rounded text-[11px] font-medium capitalize transition-colors ${activeCategory === cat ? "bg-surface-50 text-zinc-200" : "text-zinc-500 hover:text-zinc-300"}`}>{cat}</button>
          ))}
        </div>
        {(searchQuery || activeCategory !== "all" || viewFilter !== "all") && tasks && (
          <span className="text-xs text-zinc-600 self-center">{filtered.length} of {tasks.length}</span>
        )}
      </div>

      <div className="grid gap-3 rounded-lg border border-border bg-surface-100 p-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-600">Registry Workflow</p>
          <p className="mt-2 text-xs text-zinc-400">
            Installed tasks are linked to the shared registry. Update them from the registry when upstream changes land, and make substantive edits in the repository rather than inventing a separate in-app task editor.
          </p>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-600">Workspace Tasks</p>
          <p className="mt-2 text-xs text-zinc-400">
            Use <span className="text-zinc-200">New Task</span> for workspace-local variants. Treat rename and major refactors as file/repository operations so task identity, versioning, and registry linkage stay clear.
          </p>
        </div>
      </div>

      {/* Main: table + detail panel */}
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Table */}
        <div className="flex-1 min-w-0 rounded-lg border border-border bg-surface-100 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-surface-100 border-b border-border">
                  <SortHeader label="Name" sortKey="name" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                  <SortHeader label="Category" sortKey="category" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                  <SortHeader label="Montage" sortKey="montage" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                  <SortHeader label="Rate" sortKey="sample_rate" currentKey={sortKey} dir={sortDir} onSort={handleSort} className="w-20" />
                  <SortHeader label="ICA" sortKey="ica_method" currentKey={sortKey} dir={sortDir} onSort={handleSort} className="w-24" />
                  <SortHeader label="Status" sortKey="status" currentKey={sortKey} dir={sortDir} onSort={handleSort} className="w-24" />
                  <th className="px-3 py-2 w-20" />
                </tr>
              </thead>
              <tbody>
                {loading && !tasks ? (
                  Array.from({ length: 6 }).map((_, i) => (
                    <tr key={i} className="border-b border-border-subtle">
                      {Array.from({ length: 7 }).map((_, j) => (
                        <td key={j} className="px-3 py-2.5"><div className="h-4 w-3/4 rounded bg-surface-50 animate-pulse" /></td>
                      ))}
                    </tr>
                  ))
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-12 text-center">
                      <Cpu className="w-7 h-7 text-zinc-700 mx-auto mb-2" />
                      <p className="text-sm text-zinc-500">{searchQuery || activeCategory !== "all" || viewFilter !== "all" ? "No tasks match your filters." : "No tasks discovered."}</p>
                      {(searchQuery || activeCategory !== "all" || viewFilter !== "all") && (
                        <button onClick={() => { setSearchQuery(""); setActiveCategory("all"); setViewFilter("all"); }} className="mt-2 text-xs text-brand hover:underline">Clear filters</button>
                      )}
                    </td>
                  </tr>
                ) : filtered.map(task => (
                  <tr key={task.name} onClick={() => setSelected(s => s?.name === task.name ? null : task)}
                    className={`border-b border-border-subtle cursor-pointer transition-colors duration-100 ${selected?.name === task.name ? "bg-brand/15 dark:bg-brand/10" : "hover:bg-surface-50/30"}`}>
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2 min-w-0">
                        <Cpu className="w-3.5 h-3.5 text-brand/60 flex-shrink-0" />
                        <span className="text-sm font-medium text-zinc-200 truncate">{task.name}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className={`text-xs font-medium capitalize ${CATEGORY_COLORS[task.category.toLowerCase()] ?? "text-zinc-400"}`}>{task.category}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="text-xs font-mono text-zinc-400 truncate block max-w-[150px]">{task.config?.montage || "—"}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="text-xs font-mono text-zinc-400">{task.config?.sample_rate != null ? `${task.config.sample_rate}` : "—"}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="text-xs text-zinc-400">{task.config?.ica_method || "—"}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${STATUS_STYLES[task.sync_status]}`}>
                        {STATUS_LABELS[task.sync_status]}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      {task.sync_status === "not_installed" && (
                        <button disabled={actionInFlight === task.name} onClick={e => { e.stopPropagation(); handleInstall(task.name); }}
                          className="text-xs text-emerald-400 hover:text-emerald-300 disabled:opacity-50">
                          {actionInFlight === task.name ? <Loader2 className="w-3 h-3 animate-spin inline" /> : "Install"}
                        </button>
                      )}
                      {task.sync_status === "installed" && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500/50 inline" />}
                      {task.sync_status === "modified" && (
                        <button disabled={actionInFlight === task.name} onClick={e => { e.stopPropagation(); handleUpdate(task.name); }}
                          className="text-xs text-amber-400 hover:text-amber-300 disabled:opacity-50">
                          {actionInFlight === task.name ? <Loader2 className="w-3 h-3 animate-spin inline" /> : "Update"}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filtered.length > 0 && (
            <div className="px-3 py-2 border-t border-border-subtle text-xs text-zinc-600">
              {filtered.length} task{filtered.length !== 1 ? "s" : ""}
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selected && (
          <div className="lg:w-96 flex-shrink-0 rounded-lg border border-border bg-surface-100 self-start lg:sticky lg:top-5 overflow-hidden">
            {/* Header */}
            <div className="px-5 py-4 border-b border-border">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 min-w-0">
                  <Cpu className="w-4 h-4 text-brand flex-shrink-0" />
                  <h3 className="text-sm font-semibold text-zinc-100 truncate">{selected.name}</h3>
                </div>
                <button onClick={() => setSelected(null)} className="text-zinc-500 hover:text-zinc-300"><X className="w-4 h-4" /></button>
              </div>
              <div className="flex items-center gap-1.5 mb-2">
                <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${STATUS_STYLES[selected.sync_status]}`}>{STATUS_LABELS[selected.sync_status]}</span>
                <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium capitalize ${CATEGORY_COLORS[selected.category.toLowerCase()] ?? "text-zinc-400"} bg-surface-50`}>{selected.category}</span>
                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium text-zinc-500 bg-surface-50">{selected.source === "library" ? "Library" : selected.source === "builtin" ? "Built-in" : "Workspace"}</span>
              </div>
              {selected.description && <p className="text-xs text-zinc-500 leading-relaxed">{selected.description}</p>}
            </div>

            {/* Config summary */}
            {selected.config && (
              <div className="px-5 py-3 border-b border-border grid grid-cols-2 gap-2">
                <div className="flex items-center gap-1.5"><Gauge className="w-3 h-3 text-zinc-600" /><span className="text-xs text-zinc-400">{selected.config.sample_rate ?? "—"} Hz</span></div>
                <div className="flex items-center gap-1.5"><Filter className="w-3 h-3 text-zinc-600" /><span className="text-xs text-zinc-400">{selected.config.filter_low ?? ""}–{selected.config.filter_high ?? ""} Hz</span></div>
                <div className="flex items-center gap-1.5"><Brain className="w-3 h-3 text-zinc-600" /><span className="text-xs text-zinc-400">{selected.config.ica_method || "—"}</span></div>
                <div className="flex items-center gap-1.5"><Clock className="w-3 h-3 text-zinc-600" /><span className="text-xs text-zinc-400">{selected.config.epoch_tmin ?? ""}s – {selected.config.epoch_tmax ?? ""}s</span></div>
              </div>
            )}

            {/* Tab toggle: Pipeline / Source */}
            <div className="flex border-b border-border">
              <button onClick={() => setDetailTab("pipeline")}
                className={`flex-1 px-4 py-2 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors ${detailTab === "pipeline" ? "text-brand border-b-2 border-brand" : "text-zinc-500 hover:text-zinc-300"}`}>
                <Workflow className="w-3 h-3" /> Pipeline
              </button>
              <button onClick={() => setDetailTab("source")} disabled={!selected.source_code}
                className={`flex-1 px-4 py-2 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors disabled:opacity-30 ${detailTab === "source" ? "text-brand border-b-2 border-brand" : "text-zinc-500 hover:text-zinc-300"}`}>
                <Code2 className="w-3 h-3" /> Source
              </button>
            </div>

            {/* Tab content */}
            {detailTab === "pipeline" && (
              <div className="px-5 py-4 max-h-[400px] overflow-y-auto">
                <PipelineView steps={selected.pipeline} />
              </div>
            )}
            {detailTab === "source" && selected.source_code && (
              <CodeViewer
                lines={selected.source_code.split("\n")}
                colorize={colorizePython}
                maxHeight="400px"
              />
            )}

            <div className="px-5 py-3 border-t border-border-subtle bg-surface-50/40">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-600">Editing Guidance</p>
              <p className="mt-2 text-xs text-zinc-400">
                {selected.sync_status === "workspace_only"
                  ? "Workspace-only tasks can be iterated locally, but treat rename and major refactors as file/repository operations so route references and task identity stay clear."
                  : "Registry-linked tasks should be updated from the shared registry and edited in the repository rather than through a separate in-app editor."}
              </p>
            </div>

            {/* Actions */}
            <div className="px-5 py-3 border-t border-border flex items-center gap-2">
              {selected.sync_status === "not_installed" && (
                <button disabled={actionInFlight === selected.name} onClick={() => handleInstall(selected.name)}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50 transition-colors">
                  {actionInFlight === selected.name ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />} Install
                </button>
              )}
              {selected.sync_status === "modified" && (
                <button disabled={actionInFlight === selected.name} onClick={() => handleUpdate(selected.name)}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-amber-600 text-white hover:bg-amber-500 disabled:opacity-50 transition-colors">
                  {actionInFlight === selected.name ? <Loader2 className="w-3 h-3 animate-spin" /> : <ArrowUpCircle className="w-3 h-3" />} Update to Latest
                </button>
              )}
              {selected.sync_status === "installed" && (
                <span className="flex-1 flex items-center justify-center gap-1.5 text-xs text-emerald-500/70 font-medium"><CheckCircle2 className="w-3.5 h-3.5" /> Up to date</span>
              )}
              {selected.sync_status === "workspace_only" && (
                <button disabled={actionInFlight === selected.name} onClick={() => setConfirmRemove(selected.name)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium border border-red-500/30 text-red-400 hover:bg-red-500/10 disabled:opacity-50 transition-colors">
                  {actionInFlight === selected.name ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />} Remove
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      <CreateTaskModal open={showCreateModal} onClose={() => setShowCreateModal(false)} onCreate={handleCreate} creating={creating} error={createError} />
      <ConfirmDialog open={confirmRemove !== null} title="Remove task?" message={<>Delete <strong className="text-zinc-200">{confirmRemove}.py</strong> from workspace? It can be re-installed from the library.</>}
        confirmLabel="Remove" confirmVariant="danger" onConfirm={handleRemoveConfirmed} onCancel={() => setConfirmRemove(null)} />
    </div>
  );
}
