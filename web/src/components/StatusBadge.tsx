type BadgeStatus =
  | "ready"
  | "attention"
  | "archived"
  | "disabled"
  | "running"
  | "stopped"
  | "pending"
  | "processing"
  | "processed"
  | "failed"
  | "valid"
  | "invalid"
  | "testing"
  | "live";

interface StatusBadgeProps {
  status: BadgeStatus;
  label?: string;
}

const statusStyles: Record<BadgeStatus, string> = {
  ready: "bg-brand/15 text-brand",
  running: "bg-brand/15 text-brand",
  processed: "bg-brand/15 text-brand",
  valid: "bg-brand/15 text-brand",
  attention: "bg-red-500/15 text-red-400",
  failed: "bg-red-500/15 text-red-400",
  invalid: "bg-red-500/15 text-red-400",
  pending: "bg-amber-500/15 text-amber-400",
  processing: "bg-amber-500/15 text-amber-400",
  archived: "bg-zinc-500/15 text-zinc-400",
  disabled: "bg-zinc-500/15 text-zinc-400",
  stopped: "bg-zinc-500/15 text-zinc-400",
  testing: "bg-cyan-500/15 text-cyan-400",
  live: "bg-red-500/15 text-red-400",
};

export default function StatusBadge({ status, label }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${statusStyles[status]}`}
    >
      {label || status}
    </span>
  );
}
