import { useEffect } from "react";
import { X } from "lucide-react";
import { SHORTCUT_HELP } from "../hooks/useKeyboardShortcuts";

const PAGE_SHORTCUTS = [
  { key: "1", name: "Dashboard" },
  { key: "2", name: "Routes" },
  { key: "3", name: "Queue" },
  { key: "4", name: "Service" },
  { key: "5", name: "Tasks" },
  { key: "6", name: "Montages" },
  { key: "7", name: "Results" },
  { key: "8", name: "Events" },
  { key: "9", name: "Settings" },
  { key: "0", name: "Utilities" },
];

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function KeyboardShortcutsHelp({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="bg-surface-200 border border-border rounded-xl shadow-2xl w-full max-w-md mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 className="text-base font-semibold text-zinc-100">Keyboard Shortcuts</h2>
          <button onClick={onClose} className="p-1 rounded hover:bg-surface-50 text-zinc-500">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="px-5 py-4 space-y-4">
          {/* General shortcuts */}
          <div>
            <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">General</h3>
            <div className="space-y-1.5">
              {SHORTCUT_HELP.map(({ keys, description }) => (
                <div key={keys} className="flex items-center justify-between">
                  <span className="text-sm text-zinc-300">{description}</span>
                  <kbd className="px-2 py-0.5 rounded bg-surface-50 border border-border text-xs font-mono text-zinc-400">
                    {keys}
                  </kbd>
                </div>
              ))}
            </div>
          </div>

          {/* Page navigation */}
          <div>
            <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Page Navigation</h3>
            <div className="grid grid-cols-2 gap-1.5">
              {PAGE_SHORTCUTS.map(({ key, name }) => (
                <div key={name} className="flex items-center justify-between">
                  <span className="text-sm text-zinc-300">{name}</span>
                  <kbd className="px-2 py-0.5 rounded bg-surface-50 border border-border text-xs font-mono text-zinc-400">
                    {key}
                  </kbd>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
