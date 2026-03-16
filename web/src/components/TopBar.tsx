import { useState, useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";
import { Share2, Copy, Check, Loader2, X, Menu, Sun, Moon, Settings2 } from "lucide-react";
import { useTheme } from "../contexts/ThemeContext";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import type { TunnelStatus } from "../lib/api";
import StatusBadge from "./StatusBadge";

const pageTitles: Record<string, string> = {
  "/": "Dashboard",
  "/routes": "Routes",
  "/queue": "Queue",
  "/service": "Service",
  "/tasks": "Task Manager",
  "/montages": "Montages",
  "/results": "Results",
  "/exclude": "Exclude",
  "/events": "Events",
  "/settings": "Settings",
  "/setup": "Setup",
};

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    };
  }, []);

  const copy = async () => {
    await navigator.clipboard.writeText(text);
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    setCopied(true);
    copyTimerRef.current = setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={copy}
      className="p-1 rounded hover:bg-surface-50 text-zinc-500 hover:text-zinc-300 transition-colors"
      title="Copy"
    >
      {copied ? <Check className="w-3.5 h-3.5 text-brand" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  );
}

interface TopBarProps {
  onToggleSidebar?: () => void;
}

export default function TopBar({ onToggleSidebar }: TopBarProps) {
  const location = useLocation();
  const { data: health, refresh: refreshHealth } = usePolling(api.getHealth, 10000);
  const { data: tunnel, refresh: refreshTunnel } = usePolling<TunnelStatus>(
    api.getTunnelStatus,
    10000
  );

  const [showPopover, setShowPopover] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [configToken, setConfigToken] = useState("");
  const [configUrl, setConfigUrl] = useState("");
  const [configSaving, setConfigSaving] = useState(false);
  const [configLoaded, setConfigLoaded] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  const { theme, toggle: toggleTheme } = useTheme();
  const title = pageTitles[location.pathname] || "AutoClean";
  const tunnelActive = tunnel?.active ?? false;

  // Close popover on outside click
  useEffect(() => {
    if (!showPopover) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setShowPopover(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showPopover]);

  // Close popover on Escape
  useEffect(() => {
    if (!showPopover) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setShowPopover(false);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [showPopover]);

  const handleShare = async () => {
    if (tunnelActive) {
      setShowPopover(!showPopover);
      return;
    }
    setStarting(true);
    setError(null);
    try {
      const result = await api.startTunnel();
      if (result.success) {
        refreshTunnel();
        setShowPopover(true);
      } else {
        setError(result.message || "Failed to start tunnel");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // Extract detail from FastAPI error response
      const match = msg.match(/API \d+: (.+)/);
      const detail: string = match?.[1] ?? msg;
      try {
        const parsed = JSON.parse(detail);
        setError(parsed.detail ?? detail);
      } catch {
        setError(detail);
      }
      setShowPopover(true);
    } finally {
      setStarting(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      await api.stopTunnel();
      refreshTunnel();
      setShowPopover(false);
    } catch {
      // ignore
    } finally {
      setStopping(false);
    }
  };

  return (
    <header className="h-14 flex-shrink-0 flex items-center justify-between px-6 bg-surface-300 border-b border-border">
      {/* Left: Hamburger + Page title */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="md:hidden p-1.5 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-surface-50 transition-colors"
          aria-label="Toggle sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>
        <h1 className="text-lg font-semibold text-zinc-100">{title}</h1>
      </div>

      {/* Right: Status indicators */}
      <div className="flex items-center gap-3">
        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-surface-50 transition-colors"
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Light mode" : "Dark mode"}
        >
          {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        {/* Share button */}
        <div className="relative" ref={popoverRef}>
          <button
            onClick={handleShare}
            disabled={starting}
            className={[
              "rounded-md px-3 py-1.5 text-sm font-medium flex items-center gap-2 transition-colors duration-150",
              tunnelActive
                ? "bg-brand/20 text-brand border border-brand/40 hover:bg-brand/30"
                : "border border-border text-zinc-400 hover:text-zinc-200 hover:bg-surface-50",
              starting ? "opacity-60 cursor-wait" : "",
            ].join(" ")}
          >
            {starting ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : tunnelActive ? (
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-brand" />
              </span>
            ) : (
              <Share2 className="w-3.5 h-3.5" />
            )}
            {starting ? "Starting..." : tunnelActive ? "Sharing" : "Share"}
          </button>

          {/* Popover */}
          {showPopover && (
            <div className="absolute right-0 top-full mt-2 z-50 w-80 rounded-lg border border-border bg-surface-200 shadow-xl">
              <div className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-zinc-100">
                    {tunnelActive ? "Public Access" : "Share Publicly"}
                  </h3>
                  <button
                    onClick={() => setShowPopover(false)}
                    className="p-1 rounded hover:bg-surface-50 text-zinc-500"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {error && !tunnelActive && (
                  <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-400 mb-3">
                    {error}
                  </div>
                )}

                {tunnelActive && tunnel?.url && (
                  <div className="space-y-3">
                    {/* Mode badge */}
                    <div className="flex items-center gap-2">
                      <span className={[
                        "inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider",
                        tunnel.mode === "named"
                          ? "bg-brand/15 text-brand"
                          : "bg-cyan-500/15 text-cyan-400",
                      ].join(" ")}>
                        {tunnel.mode === "named" ? "Permanent" : "Temporary"}
                      </span>
                      {tunnel.mode === "quick" && (
                        <span className="text-[10px] text-zinc-600">URL changes on restart</span>
                      )}
                    </div>

                    {/* URL */}
                    <div>
                      <label className="block text-xs text-zinc-500 mb-1">URL</label>
                      <div className="flex items-center gap-1">
                        <code className="flex-1 text-xs font-mono text-brand bg-surface-50 rounded px-2 py-1.5 break-all">
                          {tunnel.url}
                        </code>
                        <CopyButton text={tunnel.url} />
                      </div>
                    </div>

                    {/* Credentials */}
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-xs text-zinc-500 mb-1">Username</label>
                        <div className="flex items-center gap-1">
                          <code className="flex-1 text-xs font-mono text-zinc-300 bg-surface-50 rounded px-2 py-1.5">
                            autoclean
                          </code>
                          <CopyButton text="autoclean" />
                        </div>
                      </div>
                      <div>
                        <label className="block text-xs text-zinc-500 mb-1">Password</label>
                        <div className="flex items-center gap-1">
                          <code className="flex-1 text-xs font-mono text-zinc-300 bg-surface-50 rounded px-2 py-1.5">
                            {tunnel.password}
                          </code>
                          <CopyButton text={tunnel.password ?? ""} />
                        </div>
                      </div>
                    </div>

                    {/* Stop button */}
                    <button
                      onClick={handleStop}
                      disabled={stopping}
                      className="w-full rounded-md px-3 py-1.5 text-sm font-medium bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 transition-colors mt-1"
                    >
                      {stopping ? "Stopping..." : "Stop Sharing"}
                    </button>
                  </div>
                )}

                {!tunnelActive && !error && !showConfig && (
                  <div className="space-y-3">
                    <p className="text-xs text-zinc-500">
                      Click Share to create a public HTTPS URL for this dashboard.
                      Anyone with the link and password can access it.
                    </p>
                    <button
                      onClick={() => {
                        setShowConfig(true);
                        if (!configLoaded) {
                          api.getTunnelConfig().then((cfg) => {
                            setConfigUrl(cfg.url);
                            setConfigLoaded(true);
                          }).catch(() => {});
                        }
                      }}
                      className="flex items-center gap-1.5 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
                    >
                      <Settings2 className="w-3 h-3" />
                      Configure permanent tunnel
                    </button>
                  </div>
                )}

                {/* Tunnel config panel */}
                {!tunnelActive && showConfig && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-semibold text-zinc-300">Named Tunnel Setup</h4>
                      <button
                        onClick={() => setShowConfig(false)}
                        className="text-[10px] text-zinc-600 hover:text-zinc-400"
                      >
                        Back
                      </button>
                    </div>
                    <p className="text-[11px] text-zinc-600 leading-relaxed">
                      For a permanent URL, create a free tunnel at{" "}
                      <a href="https://one.dash.cloudflare.com/" target="_blank" rel="noopener noreferrer" className="text-brand hover:underline">
                        Cloudflare Zero Trust
                      </a>
                      {" "}&rarr; Networks &rarr; Tunnels &rarr; Create.
                      Copy the token and your tunnel's public hostname below.
                    </p>
                    <div>
                      <label className="block text-[11px] text-zinc-500 mb-1">Tunnel Token</label>
                      <input
                        type="password"
                        value={configToken}
                        onChange={(e) => setConfigToken(e.target.value)}
                        placeholder="eyJhIjoi..."
                        className="w-full px-2 py-1.5 rounded bg-surface-50 border border-border text-xs font-mono text-zinc-300 placeholder-zinc-700 focus:outline-none focus:border-brand/40"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] text-zinc-500 mb-1">Public URL</label>
                      <input
                        type="text"
                        value={configUrl}
                        onChange={(e) => setConfigUrl(e.target.value)}
                        placeholder="https://eeg-lab.example.com"
                        className="w-full px-2 py-1.5 rounded bg-surface-50 border border-border text-xs font-mono text-zinc-300 placeholder-zinc-700 focus:outline-none focus:border-brand/40"
                      />
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={async () => {
                          setConfigSaving(true);
                          try {
                            await api.setTunnelConfig(configToken, configUrl);
                            setConfigToken("");
                            setShowConfig(false);
                          } catch {
                            // ignore
                          } finally {
                            setConfigSaving(false);
                          }
                        }}
                        disabled={!configToken || !configUrl || configSaving}
                        className="flex-1 rounded-md px-3 py-1.5 text-xs font-medium bg-brand text-surface-500 hover:bg-brand-500 disabled:opacity-40 transition-colors"
                      >
                        {configSaving ? "Saving..." : "Save"}
                      </button>
                      <button
                        onClick={async () => {
                          await api.clearTunnelConfig();
                          setConfigToken("");
                          setConfigUrl("");
                          setConfigLoaded(false);
                        }}
                        className="rounded-md px-3 py-1.5 text-xs font-medium border border-border text-zinc-500 hover:text-zinc-300 hover:bg-surface-50 transition-colors"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Stripe-style mode toggle */}
        {health && (
          <>
            <div className="flex items-center rounded-full border border-border overflow-hidden">
              <button
                onClick={async () => {
                  if (health.mode !== "test") {
                    try {
                      await api.switchMode("test");
                      window.location.reload();
                    } catch {
                      // Silently handle — health poll will show actual state
                    }
                  }
                }}
                className={[
                  "px-3 py-1 text-xs font-semibold transition-colors duration-150",
                  health.mode === "test"
                    ? "bg-cyan-500/20 text-cyan-400"
                    : "text-zinc-500 hover:text-zinc-300",
                ].join(" ")}
              >
                Test
              </button>
              <div className="w-px h-4 bg-border" />
              <button
                onClick={async () => {
                  if (health.mode !== "live") {
                    try {
                      await api.switchMode("live");
                      window.location.reload();
                    } catch {
                      // Silently handle — health poll will show actual state
                    }
                  }
                }}
                className={[
                  "px-3 py-1 text-xs font-semibold transition-colors duration-150",
                  health.mode === "live"
                    ? "bg-red-500/20 text-red-400"
                    : "text-zinc-500 hover:text-zinc-300",
                ].join(" ")}
              >
                Live
              </button>
            </div>
            <StatusBadge
              status={health.status === "healthy" ? "ready" : "attention"}
              label={health.status === "healthy" ? "Healthy" : "Unhealthy"}
            />
          </>
        )}
        {!health && (
          <span className="text-xs text-zinc-600">Connecting...</span>
        )}
      </div>
    </header>
  );
}
