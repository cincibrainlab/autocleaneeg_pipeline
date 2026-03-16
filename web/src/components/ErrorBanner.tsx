import { AlertTriangle, X } from "lucide-react";
import { useState, useEffect } from "react";

export default function ErrorBanner({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
  const [dismissed, setDismissed] = useState(false);

  // Reset dismissed state when message changes
  useEffect(() => {
    setDismissed(false);
  }, [message]);

  if (dismissed) return null;

  return (
    <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 flex items-center gap-3">
      <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
      <p className="text-sm text-red-300 flex-1">{message}</p>
      <button
        onClick={() => { setDismissed(true); onDismiss?.(); }}
        className="text-red-400 hover:text-red-300"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
