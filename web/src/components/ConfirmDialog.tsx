import { useEffect, useState } from "react";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: React.ReactNode;
  confirmLabel?: string;
  confirmVariant?: "danger" | "primary";
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  confirmVariant = "primary",
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const [clicked, setClicked] = useState(false);

  useEffect(() => {
    if (open) setClicked(false);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onCancel}
    >
      <div
        className="w-full max-w-md rounded-lg border border-border bg-surface-200 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-lg font-semibold text-zinc-100 mb-2">{title}</h3>
        <div className="text-sm text-zinc-400 mb-6">{message}</div>
        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            className="rounded-md px-4 py-2 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50"
          >
            Cancel
          </button>
          <button
            disabled={clicked}
            onClick={() => { setClicked(true); onConfirm(); }}
            className={
              confirmVariant === "danger"
                ? "rounded-md px-4 py-2 text-sm font-medium bg-red-600 text-white hover:bg-red-700"
                : "rounded-md px-4 py-2 text-sm font-medium bg-brand text-brand-900 hover:bg-brand-500"
            }
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
