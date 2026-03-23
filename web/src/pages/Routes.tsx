import { useState, useEffect, useRef, useMemo } from "react";
import { createPortal } from "react-dom";
import { Plus, MoreVertical, RefreshCw, Pencil, X, FolderOpen, GitBranch } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import type { RouteSpec, RouteFormData, TaskOption, MontageOption } from "../lib/api";
import DataTable from "../components/DataTable";
import type { Column } from "../components/DataTable";
import StatusBadge from "../components/StatusBadge";
import ErrorBanner from "../components/ErrorBanner";
import ConfirmDialog from "../components/ConfirmDialog";
import TagInput from "../components/TagInput";
import FolderBrowser from "../components/FolderBrowser";
import { useTutorial } from "../contexts/TutorialContext";
import { useTutorialTarget } from "../hooks/useTutorialTarget";

type ViewFilter = "active" | "archived" | "all";

// ── Helpers ────────────────────────────────────────────────────

function routeStatusBadge(route: RouteSpec) {
  if (route.archived) return <StatusBadge status="archived" />;
  if (!route.enabled) return <StatusBadge status="disabled" />;
  return <StatusBadge status="ready" label="Active" />;
}

function modeDots(modes: string[]) {
  if (modes.length === 0) {
    return <StatusBadge status="attention" label="No Mode" />;
  }
  return (
    <div className="flex items-center gap-3">
      {modes.map((mode) => (
        <span key={mode} className="inline-flex items-center gap-1.5 text-xs font-medium">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              mode === "live" ? "bg-red-400" : "bg-cyan-400"
            }`}
          />
          <span className={mode === "live" ? "text-red-400" : "text-cyan-400"}>
            {mode === "live" ? "Live" : "Testing"}
          </span>
        </span>
      ))}
    </div>
  );
}

function truncatePath(p: string, max = 40) {
  if (p.length <= max) return p;
  return "..." + p.slice(-(max - 3));
}

function getRouteFormError(form: RouteFormData) {
  if (!form.id.trim()) return "Route ID is required.";
  if (!form.taskfile.trim()) return "Task is required.";
  if (!form.montage.trim()) return "Montage is required.";
  if (form.ingestion_folders.length === 0) return "Add at least one input folder.";
  return null;
}

const emptyForm: RouteFormData = {
  id: "",
  taskfile: "",
  montage: "",
  ingestion_folders: [],
  file_globs: [],
  modes: ["test"],
  enabled: true,
  recursive: false,
  priority: 100,
};

// ── Action Menu (portal-based) ──────────────────────────────────

function ActionMenu({
  route,
  onAction,
  onEdit,
  onClose,
  triggerRef,
}: {
  route: RouteSpec;
  onAction: (action: string, id: string) => void;
  onEdit: (r: RouteSpec) => void;
  onClose: () => void;
  triggerRef: HTMLButtonElement | null;
}) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  useEffect(() => {
    if (triggerRef) {
      const rect = triggerRef.getBoundingClientRect();
      setPos({ top: rect.bottom + 4, left: rect.right - 176 }); // 176 = w-44
    }
  }, [triggerRef]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const keyHandler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", handler);
    document.addEventListener("keydown", keyHandler);
    return () => {
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("keydown", keyHandler);
    };
  }, [onClose]);

  const item =
    "w-full text-left px-3 py-1.5 text-sm text-zinc-300 hover:bg-surface-50/50 transition-colors";

  return createPortal(
    <div
      ref={menuRef}
      className="fixed z-50 w-44 rounded-md border border-border bg-surface-200 py-1 shadow-xl"
      style={{ top: pos.top, left: pos.left }}
    >
      {!route.archived && route.enabled && !route.modes.includes("live") && (
        <button onClick={() => onAction("promote", route.id)} className={item}>
          <span className="inline-block w-2 h-2 rounded-full bg-red-400 mr-2" />
          Go Live
        </button>
      )}
      {route.enabled && !route.archived && (
        <button onClick={() => onAction("disable", route.id)} className={item}>
          Disable
        </button>
      )}
      {!route.enabled && !route.archived && (
        <button onClick={() => onAction("enable", route.id)} className={item}>
          Enable
        </button>
      )}
      {!route.archived ? (
        <button onClick={() => onAction("archive", route.id)} className={item}>
          Archive
        </button>
      ) : (
        <button onClick={() => onAction("unarchive", route.id)} className={item}>
          Restore
        </button>
      )}
    </div>,
    document.body
  );
}

// ── Page Component ──────────────────────────────────────────────

export default function RoutesPage() {
  const { data: routes, error, loading, refresh } = usePolling(api.getRoutes, 10000);
  const [viewFilter, setViewFilter] = useState<ViewFilter>("active");
  const [search, setSearch] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState<RouteFormData>({ ...emptyForm });
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [actionMenu, setActionMenu] = useState<{ id: string; trigger: HTMLButtonElement } | null>(null);
  const [editingRoute, setEditingRoute] = useState<RouteSpec | null>(null);
  const [taskOptions, setTaskOptions] = useState<TaskOption[]>([]);
  const [montageOptions, setMontageOptions] = useState<MontageOption[]>([]);
  const [confirmAction, setConfirmAction] = useState<{
    type: string;
    id: string;
    title: string;
    message: React.ReactNode;
  } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [showFolderBrowser, setShowFolderBrowser] = useState(false);

  // Tutorial integration
  const { isActive, currentStep, tutorialData, nextStep } = useTutorial();
  const newRouteButtonRef = useTutorialTarget("new-route-button");
  const routeModalRef = useTutorialTarget("route-modal");

  // Tutorial step 2: when active at "create-route-button", watch for modal open
  useEffect(() => {
    if (isActive && currentStep === 2 && showModal) {
      nextStep();
    }
  }, [isActive, currentStep, showModal, nextStep]);

  // Tutorial step 3: auto-fill form from tutorialData when modal opens
  useEffect(() => {
    if (isActive && currentStep === 3 && showModal && tutorialData?.suggestedRoute) {
      const r = tutorialData.suggestedRoute;
      setForm({
        id: r.id ?? "",
        taskfile: r.taskfile ?? "",
        montage: r.montage ?? "",
        ingestion_folders: r.ingestion_folders ?? [],
        file_globs: r.file_globs ?? [],
        modes: r.modes ?? ["test"],
        enabled: r.enabled ?? true,
        recursive: r.recursive ?? false,
        priority: r.priority ?? 100,
      });
    }
  }, [isActive, currentStep, showModal, tutorialData]);

  // Auto-dismissing notice helper — cancels any pending timer before setting a new one
  const showNotice = (msg: string) => {
    if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    setNotice(msg);
    noticeTimerRef.current = setTimeout(() => setNotice(null), 4000);
  };

  // Clear notice timer on unmount
  useEffect(() => {
    return () => {
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    };
  }, []);

  // Close modal on Escape key
  useEffect(() => {
    if (!showModal) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setShowModal(false);
        setEditingRoute(null);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [showModal]);

  // Load discovery options when modal opens
  useEffect(() => {
    if (showModal) {
      api.getTaskOptions().then(setTaskOptions).catch(() => {});
      api.getMontageOptions().then(setMontageOptions).catch(() => {});
    }
  }, [showModal]);

  // Filter routes (memoized to avoid new array ref every render)
  const filtered = useMemo(() => (routes || []).filter((r) => {
    if (viewFilter === "active" && (r.archived || !r.enabled)) return false;
    if (viewFilter === "archived" && !r.archived) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        r.id.toLowerCase().includes(q) ||
        r.taskfile.toLowerCase().includes(q) ||
        r.montage.toLowerCase().includes(q) ||
        r.ingestion_folders.some((f) => f.toLowerCase().includes(q))
      );
    }
    return true;
  }), [routes, viewFilter, search]);

  const isFiltered = search.length > 0 || viewFilter !== "active";
  const totalRoutes = routes?.length ?? 0;
  const formError = getRouteFormError(form);

  const handleSave = async () => {
    const validationError = getRouteFormError(form);
    if (validationError) {
      setSaveError(validationError);
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      await api.createRoute(form);
      await api.syncRoutes();
      setShowModal(false);
      setEditingRoute(null);
      setForm({ ...emptyForm });
      showNotice(
        editingRoute
          ? `Route '${form.id}' updated. Open Settings and click Apply to publish the latest config.`
          : `Route '${form.id}' created. Open Settings and click Apply to start using it for processing.`
      );
      refresh();
      // Advance tutorial from step 3 (route-form) to step 4 (apply-config)
      if (isActive && currentStep === 3) {
        nextStep();
      }
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const requestAction = (
    action: "promote" | "archive" | "unarchive" | "enable" | "disable",
    id: string
  ) => {
    setActionMenu(null);
    switch (action) {
      case "archive":
        setConfirmAction({
          type: "archive",
          id,
          title: `Archive route '${id}'?`,
          message: (
            <span>
              This will archive route <strong className="text-zinc-200">'{id}'</strong> and stop
              watching its input folders. Files already in the queue will not be affected.
            </span>
          ),
        });
        break;
      case "promote":
        setConfirmAction({
          type: "promote",
          id,
          title: `Enable '${id}' for Live processing?`,
          message: (
            <div className="space-y-2">
              <p>
                This will enable <strong className="text-red-400">live processing</strong> for
                route <strong className="text-zinc-200">'{id}'</strong>.
              </p>
              <p className="text-red-400/80">
                Real clinical data will be processed. Files in the queue will begin processing
                immediately.
              </p>
            </div>
          ),
        });
        break;
      default:
        executeAction(action, id);
    }
  };

  const executeAction = async (action: string, id: string) => {
    try {
      switch (action) {
        case "promote":
          await api.promoteRoute(id);
          break;
        case "archive":
          await api.archiveRoute(id);
          break;
        case "unarchive":
          await api.unarchiveRoute(id);
          break;
        case "enable":
          await api.enableRoute(id);
          break;
        case "disable":
          await api.disableRoute(id);
          break;
      }
      await api.syncRoutes();
      const labels: Record<string, string> = {
        promote: "promoted to Live",
        archive: "archived",
        unarchive: "restored",
        enable: "enabled",
        disable: "disabled",
      };
      showNotice(`Route '${id}' ${labels[action] || action}`);
      refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Route action failed");
    }
  };

  const handleConfirm = async () => {
    if (confirmAction) {
      const { type, id } = confirmAction;
      setConfirmAction(null);
      await executeAction(type, id);
    }
  };

  const openEditModal = (route: RouteSpec) => {
    setEditingRoute(route);
    setForm({
      id: route.id,
      taskfile: route.taskfile,
      montage: route.montage,
      ingestion_folders: [...route.ingestion_folders],
      file_globs: [...route.file_globs],
      modes: [...route.modes],
      enabled: route.enabled ?? true,
      recursive: route.recursive ?? false,
      priority: route.priority ?? 100,
    });
    setSaveError(null);
    setShowModal(true);
  };

  const clearFilters = () => {
    setSearch("");
    setViewFilter("active");
  };

  // ── Table columns ──────────────────────────────────────────

  const columns: Column<RouteSpec>[] = [
    {
      key: "route",
      header: "Route",
      render: (r) => {
        const taskName = r.taskfile.split("/").pop()?.replace(/\.py$/, "") || r.taskfile;
        return (
          <div>
            <span className="font-mono text-sm text-zinc-200">{r.id}</span>
            <div className="text-xs text-zinc-500 mt-0.5">
              {taskName} + {r.montage}
            </div>
          </div>
        );
      },
    },
    {
      key: "watches",
      header: "Watches",
      render: (r) => {
        const folders = r.ingestion_folders;
        const globs = r.file_globs.join(", ");
        const first = folders[0] ?? "";
        if (folders.length === 0) return <span className="text-zinc-600">No folders</span>;
        return (
          <div>
            <div className="flex items-center gap-1.5">
              <FolderOpen className="w-3 h-3 text-zinc-600 flex-shrink-0" />
              <span className="font-mono text-xs text-zinc-400 truncate max-w-[220px]" title={first}>
                {truncatePath(first)}
              </span>
              {folders.length > 1 && (
                <span className="text-[10px] text-zinc-600" title={folders.slice(1).join("\n")}>
                  +{folders.length - 1}
                </span>
              )}
            </div>
            {globs && (
              <span className="text-[11px] text-zinc-600 font-mono ml-[18px]">{globs}</span>
            )}
          </div>
        );
      },
    },
    {
      key: "modes",
      header: "Mode",
      render: (r) => modeDots(r.modes),
    },
    {
      key: "status",
      header: "Status",
      render: (r) => routeStatusBadge(r),
    },
    {
      key: "actions",
      header: "",
      className: "w-20 text-right",
      render: (r) => (
        <div className="flex items-center justify-end gap-1">
          <button
            onClick={(e) => {
              e.stopPropagation();
              openEditModal(r);
            }}
            title="Edit route"
            className="p-1 rounded hover:bg-surface-50 text-zinc-500 hover:text-zinc-300 transition-colors duration-150"
          >
            <Pencil className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setActionMenu(
                actionMenu?.id === r.id ? null : { id: r.id, trigger: e.currentTarget }
              );
            }}
            className="p-1 rounded hover:bg-surface-50 text-zinc-500 hover:text-zinc-300 transition-colors duration-150"
          >
            <MoreVertical className="w-4 h-4" />
          </button>
          {actionMenu?.id === r.id && (
            <ActionMenu
              route={r}
              triggerRef={actionMenu.trigger}
              onAction={(action, id) => requestAction(action as "promote" | "archive" | "unarchive" | "enable" | "disable", id)}
              onEdit={openEditModal}
              onClose={() => setActionMenu(null)}
            />
          )}
        </div>
      ),
    },
  ];

  // ── Empty states ──────────────────────────────────────────

  const emptyState = (() => {
    if (totalRoutes === 0 && !loading) {
      return (
        <div className="flex flex-col items-center gap-3 py-4">
          <GitBranch className="w-10 h-10 text-zinc-700" />
          <div className="text-center">
            <p className="text-sm text-zinc-400 font-medium">No routes yet</p>
            <p className="text-xs text-zinc-600 mt-1 max-w-xs">
              Routes map input folders to processing tasks. Create your first route to start
              watching for EEG files.
            </p>
          </div>
          <button
            onClick={() => {
              setEditingRoute(null);
              setForm({ ...emptyForm });
              setSaveError(null);
              setShowModal(true);
            }}
            className="mt-2 rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors duration-150 flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Route
          </button>
        </div>
      );
    }
    if (filtered.length === 0 && totalRoutes > 0) {
      return (
        <div className="flex flex-col items-center gap-2 py-4">
          <p className="text-sm text-zinc-500">No routes match your search</p>
          <button
            onClick={clearFilters}
            className="text-xs text-brand hover:underline"
          >
            Clear filters
          </button>
        </div>
      );
    }
    return "No routes found";
  })();

  // ── Render ─────────────────────────────────────────────────

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <h2 className="text-xl font-semibold text-zinc-100">Routes</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={refresh}
            className="rounded-md px-3 py-1.5 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50 transition-colors duration-150 flex items-center gap-2"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Sync
          </button>
          <button
            ref={newRouteButtonRef}
            onClick={() => {
              setEditingRoute(null);
              setForm({ ...emptyForm });
              setSaveError(null);
              setShowModal(true);
            }}
            className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors duration-150 flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Route
          </button>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-surface-100 px-5 py-3 text-sm text-zinc-400">
        Routes define what Serve watches and how files are processed. Saving a route updates the draft configuration immediately; use
        <span className="mx-1 font-medium text-zinc-200">Settings → Apply</span>
        to publish those changes for processing.
      </div>

      {error && <ErrorBanner message={error} />}
      {actionError && (
        <ErrorBanner message={actionError} onDismiss={() => setActionError(null)} />
      )}

      {/* Success notice */}
      {notice && (
        <div className="rounded-lg px-4 py-2 text-sm font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center justify-between">
          {notice}
          <button onClick={() => setNotice(null)} className="text-emerald-400/60 hover:text-emerald-400">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Filter bar */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        {/* Segmented filter */}
        <div className="flex rounded-md border border-border overflow-hidden">
          {(["active", "archived", "all"] as ViewFilter[]).map((v) => (
            <button
              key={v}
              onClick={() => setViewFilter(v)}
              className={[
                "px-3 py-1.5 text-sm font-medium transition-colors duration-150",
                viewFilter === v
                  ? "bg-surface-50 text-zinc-200"
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-surface-50/30",
              ].join(" ")}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative flex-1 min-w-0 max-w-xs">
          <input
            type="text"
            placeholder="Search by ID, task, montage, or folder..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-md border border-border bg-surface-100 text-sm text-zinc-300 px-3 py-1.5 pr-8 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-brand/50"
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Result count (only when filtering) */}
        {isFiltered && routes && (
          <span className="text-xs text-zinc-500">
            Showing {filtered.length} of {totalRoutes}
          </span>
        )}
      </div>

      {/* Table */}
      <div className="rounded-lg border border-border bg-surface-100 overflow-hidden">
        <DataTable
          columns={columns}
          data={filtered as (RouteSpec & Record<string, unknown>)[]}
          loading={loading}
          emptyMessage={emptyState}
          rowClassName={(r) => (r as unknown as RouteSpec).archived ? "opacity-50" : ""}
        />
      </div>

      {/* Confirm Dialog */}
      <ConfirmDialog
        open={confirmAction !== null}
        title={confirmAction?.title ?? ""}
        message={confirmAction?.message ?? ""}
        confirmLabel={
          confirmAction?.type === "archive"
              ? "Archive"
              : confirmAction?.type === "promote"
                ? "Enable Live"
                : "Confirm"
        }
        confirmVariant={
          confirmAction?.type === "promote" ? "danger"
              : "primary"
        }
        onConfirm={handleConfirm}
        onCancel={() => setConfirmAction(null)}
      />

      {/* ── Route Modal (Create / Edit) ─────────────────────── */}
      {showFolderBrowser && (
        <FolderBrowser
          onSelect={(path) =>
            setForm((prev) => ({
              ...prev,
              ingestion_folders: prev.ingestion_folders.includes(path)
                ? prev.ingestion_folders
                : [...prev.ingestion_folders, path],
            }))
          }
          onClose={() => setShowFolderBrowser(false)}
        />
      )}

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div ref={routeModalRef} className="w-full max-w-xl rounded-lg border border-border bg-surface-200 p-4 sm:p-6 mx-4 sm:mx-auto max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-lg font-semibold text-zinc-100">
                {editingRoute ? "Edit Route" : "New Route"}
              </h3>
              <button
                onClick={() => { setShowModal(false); setEditingRoute(null); }}
                className="p-1 rounded hover:bg-surface-50 text-zinc-500"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-5">
              {/* ── Section: Identity ─────────────────────────── */}
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-1">
                  Route ID
                </label>
                <input
                  type="text"
                  value={form.id}
                  onChange={(e) => setForm({ ...form, id: e.target.value })}
                  placeholder="e.g. resting-1020"
                  readOnly={!!editingRoute}
                  className={`w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200 focus:border-brand focus:outline-none placeholder-zinc-600${
                    editingRoute ? " opacity-60 cursor-not-allowed" : ""
                  }`}
                />
                {!editingRoute && form.id && !/^[a-z0-9][a-z0-9-]*$/.test(form.id) && (
                  <p className="text-xs text-red-400 mt-1">
                    Use lowercase letters, numbers, and hyphens only
                  </p>
                )}
              </div>

              {/* ── Section: Processing ───────────────────────── */}
              <div className="border-t border-border-subtle pt-4">
                <p className="text-xs uppercase tracking-wider text-zinc-600 mb-3">Processing</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-zinc-400 mb-1">Task</label>
                    {taskOptions.length > 0 ? (
                      <select
                        value={form.taskfile}
                        onChange={(e) => setForm({ ...form, taskfile: e.target.value })}
                        className="w-full rounded-md border border-border bg-surface-100 text-sm text-zinc-200 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-brand/50"
                      >
                        <option value="">Select a task...</option>
                        {taskOptions.map((t) => (
                          <option key={t.name} value={t.name}>
                            {t.name}
                            {t.description ? ` - ${t.description}` : ""}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={form.taskfile}
                        onChange={(e) => setForm({ ...form, taskfile: e.target.value })}
                        placeholder="task_name"
                        className="w-full rounded-md border border-border bg-surface-100 text-sm text-zinc-200 px-3 py-2 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-brand/50"
                      />
                    )}
                  </div>
                  <div>
                    <label className="block text-sm text-zinc-400 mb-1">Montage</label>
                    {montageOptions.length > 0 ? (
                      <select
                        value={form.montage}
                        onChange={(e) => setForm({ ...form, montage: e.target.value })}
                        className="w-full rounded-md border border-border bg-surface-100 text-sm text-zinc-200 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-brand/50"
                      >
                        <option value="">Select a montage...</option>
                        {montageOptions.map((m) => (
                          <option key={m.name} value={m.name}>
                            {m.name}
                            {m.description ? ` - ${m.description}` : ""}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={form.montage}
                        onChange={(e) => setForm({ ...form, montage: e.target.value })}
                        placeholder="montage_name"
                        className="w-full rounded-md border border-border bg-surface-100 text-sm text-zinc-200 px-3 py-2 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-brand/50"
                      />
                    )}
                  </div>
                </div>
              </div>

              {/* ── Section: File Matching ────────────────────── */}
              <div className="border-t border-border-subtle pt-4">
                <p className="text-xs uppercase tracking-wider text-zinc-600 mb-3">File Matching</p>
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="block text-sm text-zinc-400">
                        Input Folders
                      </label>
                      <button
                        type="button"
                        onClick={() => setShowFolderBrowser(true)}
                        className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium border border-border text-zinc-400 hover:text-zinc-200 hover:bg-surface-50 transition-colors"
                      >
                        <FolderOpen className="w-3 h-3" />
                        Browse
                      </button>
                    </div>
                    <TagInput
                      value={form.ingestion_folders}
                      onChange={(tags) => setForm({ ...form, ingestion_folders: tags })}
                      placeholder="/path/to/folder — press Enter to add"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-zinc-400 mb-1">
                      File Patterns
                    </label>
                    <TagInput
                      value={form.file_globs}
                      onChange={(tags) => setForm({ ...form, file_globs: tags })}
                      placeholder="*.edf — press Enter to add"
                    />
                  </div>
                  <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={form.recursive}
                      onChange={(e) => setForm({ ...form, recursive: e.target.checked })}
                      className="rounded border-border bg-surface-100 text-brand focus:ring-brand/50"
                    />
                    Scan subfolders recursively
                  </label>
                </div>
              </div>

              {/* ── Section: Operations ───────────────────────── */}
              <div className="border-t border-border-subtle pt-4">
                <p className="text-xs uppercase tracking-wider text-zinc-600 mb-3">Operations</p>
                <div className="flex flex-wrap items-end gap-4">
                  {/* Priority */}
                  <div className="w-24">
                    <label className="block text-sm text-zinc-400 mb-1">Priority</label>
                    <input
                      type="number"
                      min="0"
                      max="1000"
                      value={form.priority}
                      onChange={(e) => setForm({ ...form, priority: Number(e.target.value) })}
                      className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200 focus:border-brand focus:outline-none"
                    />
                  </div>

                  {/* Mode toggle */}
                  <div>
                    <label className="block text-sm text-zinc-400 mb-1">Mode</label>
                    <div className="flex rounded-md border border-border overflow-hidden">
                      <button
                        type="button"
                        onClick={() => setForm({ ...form, modes: ["test"] })}
                        className={[
                          "px-3 py-1.5 text-sm font-medium transition-colors",
                          !form.modes.includes("live")
                            ? "bg-cyan-500/15 text-cyan-400 border-r border-border"
                            : "text-zinc-500 hover:text-zinc-300 border-r border-border",
                        ].join(" ")}
                      >
                        Testing Only
                      </button>
                      <button
                        type="button"
                        onClick={() => setForm({ ...form, modes: ["test", "live"] })}
                        className={[
                          "px-3 py-1.5 text-sm font-medium transition-colors",
                          form.modes.includes("live")
                            ? "bg-red-500/15 text-red-400"
                            : "text-zinc-500 hover:text-zinc-300",
                        ].join(" ")}
                      >
                        Testing + Live
                      </button>
                    </div>
                  </div>

                  {/* Enabled toggle */}
                  <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer pb-1.5">
                    <input
                      type="checkbox"
                      checked={form.enabled}
                      onChange={(e) => setForm({ ...form, enabled: e.target.checked })}
                      className="rounded border-border bg-surface-100 text-brand focus:ring-brand/50"
                    />
                    Enabled
                  </label>
                </div>
              </div>

              {/* Output folder (read-only, computed) */}
              {editingRoute?.output_folder && (
                <div className="border-t border-border-subtle pt-4">
                  <label className="block text-sm text-zinc-400 mb-1">
                    Output Folder
                  </label>
                  <div className="w-full rounded-md border border-border bg-surface-50/50 px-3 py-2 text-xs font-mono text-zinc-400">
                    {editingRoute.output_folder}
                  </div>
                  <p className="text-[11px] text-zinc-600 mt-1">
                    Computed from workspace output directory, task, and montage
                  </p>
                </div>
              )}

              {/* Error display */}
              {saveError && (
                <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">
                  {saveError}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center justify-end gap-3 pt-2 border-t border-border-subtle">
                <button
                  onClick={() => { setShowModal(false); setEditingRoute(null); }}
                  className="rounded-md px-4 py-2 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50 transition-colors duration-150"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  disabled={saving || formError !== null}
                  title={formError ?? undefined}
                  className="rounded-md px-4 py-2 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {saving
                    ? editingRoute ? "Saving..." : "Creating..."
                    : editingRoute ? "Save Changes" : "Create Route"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
