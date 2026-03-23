import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

const PAGE_SHORTCUTS: Record<string, string> = {
  "1": "/",
  "2": "/routes",
  "3": "/queue",
  "4": "/service",
  "5": "/tasks",
  "6": "/montages",
  "7": "/results",
  "8": "/events",
  "9": "/settings",
  "0": "/utilities",
};

export function useKeyboardShortcuts(onToggleHelp: () => void) {
  const navigate = useNavigate();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Ignore when typing in inputs
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.target as HTMLElement)?.isContentEditable) return;

      // ? — show keyboard shortcut help
      if (e.key === "?" && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        onToggleHelp();
        return;
      }

      // Number keys for page navigation (no modifiers)
      const targetPath = PAGE_SHORTCUTS[e.key];
      if (!e.ctrlKey && !e.metaKey && !e.altKey && targetPath) {
        e.preventDefault();
        navigate(targetPath);
        return;
      }
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [navigate, onToggleHelp]);
}

export const SHORTCUT_HELP = [
  { keys: "0-9", description: "Navigate to page" },
  { keys: "?", description: "Show keyboard shortcuts" },
  { keys: "Esc", description: "Close dialogs / overlays" },
];
