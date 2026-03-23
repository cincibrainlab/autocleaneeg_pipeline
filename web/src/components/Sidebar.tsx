import { NavLink, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  GitBranch,
  ListOrdered,
  MapPin,
  Play,
  Settings,
  Wrench,
  Github,
  BookOpen,
  FileJson,
  Brain,
  GraduationCap,
  Cpu,
  FileCheck,
  Scissors,
  FolderOpen,
  Zap,
} from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import { useTutorial } from "../contexts/TutorialContext";

const primaryNavItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/service", label: "Service", icon: Play },
  { to: "/routes", label: "Routes", icon: GitBranch },
  { to: "/queue", label: "Queue", icon: ListOrdered },
  { to: "/results", label: "Results", icon: FileCheck },
  { to: "/exclude", label: "Exclude", icon: Scissors },
];

const utilityNavItems = [
  { to: "/utilities", label: "Utilities", icon: Wrench },
  { to: "/tasks", label: "Tasks", icon: Cpu },
  { to: "/montages", label: "Montages", icon: MapPin },
  { to: "/events", label: "Events", icon: Zap },
  { to: "/settings", label: "Settings", icon: Settings },
];

interface SidebarProps {
  open?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ open, onClose }: SidebarProps) {
  const { data: health } = usePolling(api.getHealth, 30000);
  const mode = health?.mode ?? "test";
  const version = health?.pipeline_version ?? "";
  const workspaceConfigured = health?.workspace_configured ?? true;
  const { startTutorial, completed, isActive } = useTutorial();

  // Derive workspace display info from the status endpoint
  const { data: status } = usePolling(api.getStatus, 30000);
  const workspacePath = status?.workspace_dir ?? "";
  const workspaceName = workspacePath
    ? workspacePath.split("/").filter(Boolean).pop() ?? workspacePath
    : "";
  const navigate = useNavigate();

  const handleTutorial = async () => {
    await startTutorial();
  };

  return (
    <aside
      className={[
        "w-56 flex-shrink-0 bg-surface-200 border-r border-border flex flex-col",
        // Mobile: fixed overlay that slides in/out
        "fixed inset-y-0 left-0 z-40 transition-transform duration-200 md:relative md:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full",
      ].join(" ")}
    >
      {/* Logo */}
      <div className="h-14 flex items-center gap-2.5 px-5 border-b border-border">
        <Brain className="w-5 h-5 text-brand" />
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-bold text-zinc-100 tracking-tight">
            AutoClean<span className="text-brand">EEG</span>
          </span>
          <span className="text-[10px] text-zinc-600 -mt-0.5">Serve</span>
        </div>
      </div>

      <div className="border-b border-border px-4 py-3">
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-600">
          Workspace
        </p>
        <button
          onClick={() => navigate("/setup")}
          className="w-full rounded-lg border border-border bg-surface-100/70 px-3 py-2 text-left hover:bg-surface-50/30 transition-colors"
          title={workspacePath || "Choose workspace"}
        >
          <div className="flex items-center gap-2">
            <FolderOpen className="w-3.5 h-3.5 text-zinc-500 flex-shrink-0" />
            <p className="text-xs font-medium text-zinc-200 truncate">
              {workspaceConfigured && workspaceName ? workspaceName : "No workspace selected"}
            </p>
          </div>
          <p className="mt-1 text-[10px] text-zinc-500 truncate pl-5.5">
            {workspaceConfigured && workspacePath ? workspacePath : "Choose or create a workspace"}
          </p>
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-3">
        <div className="space-y-0.5">
          <p className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-600">
            Workflow
          </p>
          {primaryNavItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              onClick={onClose}
              className={({ isActive }) =>
                [
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-150",
                  isActive
                    ? "text-brand bg-brand/10 border-l-2 border-brand -ml-px"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-surface-50/50 border-l-2 border-transparent -ml-px",
                ].join(" ")
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </div>

        <div className="mt-4 space-y-0.5">
          <p className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-600">
            Utilities
          </p>
          {utilityNavItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              onClick={onClose}
              className={({ isActive }) =>
                [
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-150",
                  isActive
                    ? "text-brand bg-brand/10 border-l-2 border-brand -ml-px"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-surface-50/50 border-l-2 border-transparent -ml-px",
                ].join(" ")
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </div>
      </nav>

      {/* External links + Tutorial */}
      <div className="px-3 pb-2 space-y-0.5">
        {/* Tutorial button */}
        <button
          onClick={handleTutorial}
          disabled={isActive}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm text-zinc-500 hover:text-zinc-300 hover:bg-surface-50/50 transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="relative">
            <GraduationCap className="w-4 h-4" />
            {!completed && (
              <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-emerald-400" />
            )}
          </div>
          Tutorial
        </button>

        <a
          href="https://github.com/cincibrainlab/autoclean_pipeline"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 px-3 py-2 rounded-md text-sm text-zinc-500 hover:text-zinc-300 hover:bg-surface-50/50 transition-colors duration-150"
        >
          <Github className="w-4 h-4" />
          GitHub
        </a>
        <a
          href="/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 px-3 py-2 rounded-md text-sm text-zinc-500 hover:text-zinc-300 hover:bg-surface-50/50 transition-colors duration-150"
        >
          <FileJson className="w-4 h-4" />
          API Docs
        </a>
        <a
          href="https://docs.autocleaneeg.org/welcome"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 px-3 py-2 rounded-md text-sm text-zinc-500 hover:text-zinc-300 hover:bg-surface-50/50 transition-colors duration-150"
        >
          <BookOpen className="w-4 h-4" />
          Documentation
        </a>
      </div>

      {/* Bottom section */}
      <div className="px-5 py-3 border-t border-border space-y-1.5">
        <div className="flex items-center gap-2">
          <span
            className={[
              "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium",
              mode === "live"
                ? "bg-red-500/15 text-red-400"
                : "bg-cyan-500/15 text-cyan-400",
            ].join(" ")}
          >
            {mode === "live" ? "Live" : "Testing"}
          </span>
        </div>
        {version && (
          <p className="text-[11px] text-zinc-600">Pipeline v{version}</p>
        )}
        <p className="text-[10px] text-zinc-700">Cincinnati Brain Lab</p>
      </div>
    </aside>
  );
}
