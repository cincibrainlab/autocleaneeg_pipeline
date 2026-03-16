import { useState, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { Folder, FolderOpen, ChevronRight, X, Loader2 } from "lucide-react";
import { api } from "../lib/api";
import type { FolderEntry } from "../lib/api";

interface FolderBrowserProps {
  onSelect: (path: string) => void;
  onClose: () => void;
}

// ── Breadcrumb helpers ────────────────────────────────────────────────────────

/** Split an absolute path into labelled segments for breadcrumb rendering. */
function pathSegments(absPath: string): { label: string; path: string }[] {
  // Normalise to forward-slash (safe on all platforms served from Python)
  const parts = absPath.split("/").filter(Boolean);
  const segments: { label: string; path: string }[] = [
    { label: "/", path: "/" },
  ];
  let accumulated = "";
  for (const part of parts) {
    accumulated += "/" + part;
    segments.push({ label: part, path: accumulated });
  }
  return segments;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function FolderBrowser({ onSelect, onClose }: FolderBrowserProps) {
  const [currentPath, setCurrentPath] = useState<string | null>(null);
  const [displayPath, setDisplayPath] = useState<string>("");
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [entries, setEntries] = useState<FolderEntry[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const navigate = useCallback((path?: string) => {
    setLoading(true);
    setError(null);
    setSelectedPath(null);

    api
      .browseFolders(path)
      .then((res) => {
        setCurrentPath(res.path);
        setDisplayPath(res.path);
        setParentPath(res.parent ?? null);
        setEntries(res.entries);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        setError(msg);
      })
      .finally(() => setLoading(false));
  }, []);

  // Initial load — default to workspace root (server picks the default)
  useEffect(() => {
    navigate(undefined);
  }, [navigate]);

  // Keyboard: Escape closes the modal
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  const handleDoubleClick = (entry: FolderEntry) => {
    navigate(entry.path);
  };

  const handleSingleClick = (entry: FolderEntry) => {
    setSelectedPath((prev) => (prev === entry.path ? null : entry.path));
  };

  const handleSelect = () => {
    if (selectedPath) {
      onSelect(selectedPath);
      onClose();
    }
  };

  const segments = currentPath ? pathSegments(currentPath) : [];

  // ── Render ────────────────────────────────────────────────────────────────

  return createPortal(
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="w-full max-w-lg mx-4 rounded-lg border border-border bg-surface-200 shadow-2xl flex flex-col max-h-[80vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border flex-shrink-0">
          <span className="text-sm font-semibold text-zinc-200">Browse Folders</span>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-surface-50 text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Breadcrumb bar */}
        <div className="px-4 py-2 border-b border-border-subtle bg-surface-100/50 flex-shrink-0 overflow-x-auto">
          {currentPath ? (
            <div className="flex items-center gap-0.5 min-w-0 font-mono text-xs text-zinc-400 whitespace-nowrap">
              {segments.map((seg, idx) => (
                <span key={seg.path} className="flex items-center gap-0.5">
                  {idx > 0 && (
                    <ChevronRight className="w-3 h-3 text-zinc-600 flex-shrink-0" />
                  )}
                  <button
                    onClick={() => navigate(seg.path)}
                    className={[
                      "px-1 py-0.5 rounded transition-colors",
                      idx === segments.length - 1
                        ? "text-zinc-200 font-medium cursor-default"
                        : "hover:text-zinc-200 hover:bg-surface-50/50 cursor-pointer",
                    ].join(" ")}
                    disabled={idx === segments.length - 1}
                  >
                    {seg.label}
                  </button>
                </span>
              ))}
            </div>
          ) : (
            <span className="font-mono text-xs text-zinc-600">Loading...</span>
          )}
        </div>

        {/* Current path display */}
        {displayPath && (
          <div className="px-4 py-1.5 bg-surface-100/30 border-b border-border-subtle flex-shrink-0">
            <code className="text-[11px] text-zinc-500 break-all">{displayPath}</code>
          </div>
        )}

        {/* Directory listing */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {loading ? (
            <div className="flex items-center justify-center h-32 gap-2 text-zinc-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Loading...</span>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-32 gap-2 px-6 text-center">
              <span className="text-sm text-red-400">Error loading directory</span>
              <span className="text-xs text-zinc-600">{error}</span>
              {parentPath && (
                <button
                  onClick={() => navigate(parentPath)}
                  className="text-xs text-brand hover:underline mt-1"
                >
                  Go up
                </button>
              )}
            </div>
          ) : (
            <ul className="py-1">
              {/* Parent directory entry */}
              {parentPath && (
                <li>
                  <button
                    onClick={() => navigate(parentPath)}
                    className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-zinc-400 hover:bg-surface-50/30 transition-colors"
                  >
                    <Folder className="w-4 h-4 text-zinc-600 flex-shrink-0" />
                    <span className="font-mono text-zinc-500">..</span>
                  </button>
                </li>
              )}

              {entries.length === 0 && !parentPath ? (
                <li className="flex items-center justify-center h-20">
                  <span className="text-sm text-zinc-600">No subdirectories</span>
                </li>
              ) : entries.length === 0 ? (
                <li className="px-4 py-3">
                  <span className="text-sm text-zinc-600">No subdirectories</span>
                </li>
              ) : (
                entries.map((entry) => {
                  const isSelected = selectedPath === entry.path;
                  return (
                    <li key={entry.path}>
                      <button
                        onClick={() => handleSingleClick(entry)}
                        onDoubleClick={() => handleDoubleClick(entry)}
                        className={[
                          "w-full flex items-center gap-2.5 px-4 py-2 text-sm transition-colors",
                          isSelected
                            ? "bg-brand/15 text-zinc-100"
                            : "text-zinc-300 hover:bg-surface-50/30",
                        ].join(" ")}
                      >
                        {isSelected ? (
                          <FolderOpen className="w-4 h-4 text-brand flex-shrink-0" />
                        ) : (
                          <Folder className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                        )}
                        <span className="font-mono text-xs truncate" title={entry.name}>
                          {entry.name}
                        </span>
                      </button>
                    </li>
                  );
                })
              )}
            </ul>
          )}
        </div>

        {/* Footer / actions */}
        <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-border flex-shrink-0 bg-surface-200">
          <span className="text-xs text-zinc-600 font-mono truncate max-w-[260px]" title={selectedPath ?? ""}>
            {selectedPath ? selectedPath : <span className="italic">No folder selected</span>}
          </span>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={onClose}
              className="rounded-md px-3 py-1.5 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSelect}
              disabled={!selectedPath}
              className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Select
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
