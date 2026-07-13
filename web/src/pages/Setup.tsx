import { useState, useEffect, useCallback } from "react";
import {
  Folder,
  FolderOpen,
  CheckCircle2,
  Loader2,
  AlertCircle,
  RefreshCw,
  GitBranch,
  Cpu,
} from "lucide-react";
import { api } from "../lib/api";
import type { RecentWorkspace } from "../lib/api";
import FolderBrowser from "../components/FolderBrowser";

function runtimeLabel(ws: RecentWorkspace): string {
  if (ws.has_runtime_test && ws.has_runtime_live) return "Test + Live runtimes";
  if (ws.has_runtime_test) return "Test runtime";
  if (ws.has_runtime_live) return "Live runtime";
  return "No runtimes";
}

function routeLabel(ws: RecentWorkspace): string {
  if (ws.n_routes === 0) return "No routes";
  return `${ws.n_routes} route${ws.n_routes === 1 ? "" : "s"}`;
}

export default function Setup() {
  const [recentWorkspaces, setRecentWorkspaces] = useState<RecentWorkspace[]>([]);
  const [loadingRecent, setLoadingRecent] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [workspacePath, setWorkspacePath] = useState<string>("");
  const [showBrowser, setShowBrowser] = useState(false);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadRecent = useCallback(async () => {
    setLoadingRecent(true);
    try {
      const res = await api.getRecentWorkspaces();
      setRecentWorkspaces(res.workspaces);
    } catch {
      // Non-fatal: just show empty list
      setRecentWorkspaces([]);
    } finally {
      setLoadingRecent(false);
    }
  }, []);

  useEffect(() => {
    loadRecent();
  }, [loadRecent]);

  // When a recent workspace is clicked, populate the path input
  const handleSelectRecent = (ws: RecentWorkspace) => {
    setSelected(ws.path);
    setWorkspacePath(ws.path);
    setError(null);
  };

  const handleBrowseSelect = (path: string) => {
    setWorkspacePath(path);
    setSelected(null); // browsed path is not necessarily a recent item
    setShowBrowser(false);
    setError(null);
  };

  const openWorkspace = async (path: string, createNew: boolean) => {
    const trimmed = path.trim();
    if (!trimmed) {
      setError("Please enter or browse to a workspace folder.");
      return;
    }
    setError(null);
    setOpening(true);
    try {
      await api.setupWorkspace(trimmed, createNew);
      window.location.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setOpening(false);
    }
  };

  const handleOpen = () => openWorkspace(workspacePath, false);
  const handleCreateNew = () => openWorkspace(workspacePath, true);

  return (
    <div className="max-w-2xl mx-auto space-y-6 py-4">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-semibold text-zinc-100">Choose a workspace</h2>
        <p className="text-sm text-zinc-500 mt-1">
          Open a recent workspace or browse to the workspace root that Serve should use.
        </p>
      </div>

      <div className="rounded-lg border border-border bg-surface-100 px-5 py-3 text-sm text-zinc-400">
        CLI equivalent:
        <code className="mx-1 rounded bg-surface-50 px-1.5 py-0.5 text-xs text-zinc-200">autocleaneeg-pipeline serve workspace --mode new --path &lt;dir&gt;</code>
        or
        <code className="mx-1 rounded bg-surface-50 px-1.5 py-0.5 text-xs text-zinc-200">autocleaneeg-pipeline serve workspace --mode existing --path &lt;dir&gt;</code>
      </div>

      <div className="rounded-lg border border-border bg-surface-100 px-5 py-3 text-sm text-zinc-400 space-y-2">
        <p>
          <span className="font-medium text-zinc-200">Open Workspace</span> for an existing Serve workspace, or for an existing AutoClean workspace that already has normal project folders such as
          <code className="mx-1 rounded bg-surface-50 px-1.5 py-0.5 text-xs text-zinc-200">tasks/</code>
          and
          <code className="mx-1 rounded bg-surface-50 px-1.5 py-0.5 text-xs text-zinc-200">output/</code>.
          Serve will bootstrap the missing Serve-specific files in place when that is valid.
        </p>
        <p>
          <span className="font-medium text-zinc-200">Create New Workspace</span> only for a new, empty directory that Serve should initialize from scratch.
        </p>
      </div>

      {/* Recent workspaces list */}
      <div className="rounded-lg border border-border bg-surface-100 overflow-hidden">
        {/* List header */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-surface-200">
          <span className="text-xs font-medium text-zinc-500 uppercase tracking-wide">
            Recent workspaces
          </span>
          <button
            onClick={loadRecent}
            className="rounded p-1 text-zinc-600 hover:text-zinc-300 transition-colors"
            title="Refresh list"
            disabled={loadingRecent}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingRecent ? "animate-spin" : ""}`} />
          </button>
        </div>

        {loadingRecent ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="w-5 h-5 animate-spin text-zinc-600" />
          </div>
        ) : recentWorkspaces.length === 0 ? (
          <div className="px-5 py-8 text-center">
            <Folder className="w-8 h-8 text-zinc-700 mx-auto mb-2" />
            <p className="text-sm text-zinc-500">No recent workspaces</p>
            <p className="text-xs text-zinc-700 mt-1">
              Enter a path below to open or create a workspace.
            </p>
          </div>
        ) : (
          <ul>
            {recentWorkspaces.map((ws, idx) => {
              const isSelected = selected === ws.path;
              return (
                <li key={ws.path}>
                  {idx > 0 && <div className="border-t border-border" />}
                  <button
                    className={[
                      "w-full text-left px-4 py-3.5 transition-colors",
                      isSelected
                        ? "bg-brand/10 border-l-2 border-brand"
                        : "hover:bg-surface-50/40 border-l-2 border-transparent",
                    ].join(" ")}
                    onClick={() => handleSelectRecent(ws)}
                    onDoubleClick={() => openWorkspace(ws.path, false)}
                    title={ws.path}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-2.5 min-w-0">
                        <FolderOpen
                          className={`w-4 h-4 flex-shrink-0 mt-0.5 ${
                            isSelected ? "text-brand" : "text-zinc-500"
                          }`}
                        />
                        <div className="min-w-0">
                          <p
                            className={`text-sm font-medium truncate ${
                              isSelected ? "text-zinc-100" : "text-zinc-200"
                            }`}
                          >
                            {ws.name}
                          </p>
                          <p className="text-xs text-zinc-600 font-mono truncate mt-0.5">
                            {ws.path}
                          </p>
                          <div className="flex items-center gap-3 mt-1.5">
                            <span className="flex items-center gap-1 text-xs text-zinc-500">
                              <GitBranch className="w-3 h-3" />
                              {routeLabel(ws)}
                            </span>
                            <span className="flex items-center gap-1 text-xs text-zinc-500">
                              <Cpu className="w-3 h-3" />
                              {runtimeLabel(ws)}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        {ws.is_current && (
                          <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-brand/15 text-brand text-[10px] font-semibold uppercase tracking-wider">
                            <CheckCircle2 className="w-2.5 h-2.5" />
                            Current
                          </span>
                        )}
                      </div>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Path input + Browse */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
          Workspace path
        </label>
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <div className="absolute left-3 top-1/2 -translate-y-1/2">
              {workspacePath ? (
                <FolderOpen className="w-4 h-4 text-brand" />
              ) : (
                <Folder className="w-4 h-4 text-zinc-600" />
              )}
            </div>
            <input
              type="text"
              value={workspacePath}
              onChange={(e) => {
                setWorkspacePath(e.target.value);
                setSelected(null);
                setError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && workspacePath.trim()) handleOpen();
              }}
              placeholder="/Users/you/Documents/AutoClean"
              className="w-full rounded-md border border-border bg-surface-100 pl-9 pr-3 py-2 text-sm font-mono text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-brand focus:border-brand transition-colors"
            />
          </div>
          <button
            onClick={() => setShowBrowser(true)}
            className="flex-shrink-0 rounded-md px-3 py-2 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50 hover:text-zinc-100 transition-colors"
          >
            Browse
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="flex items-start gap-2.5 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2.5">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3">
        <div className="flex-1 space-y-1.5">
          <button
            onClick={handleOpen}
            disabled={opening || !workspacePath.trim()}
            className="w-full rounded-md py-2.5 text-sm font-semibold bg-brand text-brand-900 hover:bg-brand-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {opening ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Opening...
              </>
            ) : (
              "Open Workspace"
            )}
          </button>
          <p className="text-xs text-zinc-600">
            Use this for existing Serve workspaces and existing AutoClean workspaces.
          </p>
        </div>
        <div className="flex-1 space-y-1.5">
          <button
            onClick={handleCreateNew}
            disabled={opening || !workspacePath.trim()}
            className="w-full rounded-md py-2.5 text-sm font-semibold border border-border text-zinc-300 hover:bg-surface-50 hover:text-zinc-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {opening ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Creating...
              </>
            ) : (
              "Create New Workspace"
            )}
          </button>
          <p className="text-xs text-zinc-600">
            Use this only when the target directory is empty.
          </p>
        </div>
      </div>

      {/* Folder browser modal */}
      {showBrowser && (
        <FolderBrowser
          onSelect={handleBrowseSelect}
          onClose={() => setShowBrowser(false)}
        />
      )}
    </div>
  );
}
