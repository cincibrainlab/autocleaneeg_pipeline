import { useState, useEffect, useRef, useCallback } from "react";
import { Play, Square, ChevronDown, ChevronRight } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import type { ServiceStartSettings } from "../lib/api";
import StatusBadge from "../components/StatusBadge";
import ErrorBanner from "../components/ErrorBanner";
import { useTutorial } from "../contexts/TutorialContext";
import { useTutorialTarget } from "../hooks/useTutorialTarget";
import { formatUptime } from "../lib/format";

export default function Service() {
  const { data: service, error, refresh } = usePolling(
    api.getServiceStatus,
    3000
  );
  const [acting, setActing] = useState(false);
  const [actionResult, setActionResult] = useState<string | null>(null);
  const actionResultTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear action result timer on unmount
  useEffect(() => {
    return () => {
      if (actionResultTimerRef.current) clearTimeout(actionResultTimerRef.current);
    };
  }, []);

  // Tutorial integration
  const { isActive, currentStep, nextStep } = useTutorial();
  const serviceControlRef = useTutorialTarget("service-control");

  // ── Log viewer state ────────────────────────────────────────────
  const [logLines, setLogLines] = useState<string[]>([]);
  const [logTotal, setLogTotal] = useState(0);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const prevLineCountRef = useRef(0);

  // Poll logs when service is running
  const fetchLogs = useCallback(async () => {
    try {
      const res = await api.getServiceLogs();
      setLogLines(res.lines);
      setLogTotal(res.total);
    } catch {
      // Silently ignore log fetch errors
    }
  }, []);

  useEffect(() => {
    if (!service?.running) return;
    fetchLogs();
    const id = setInterval(fetchLogs, 3000);
    return () => clearInterval(id);
  }, [service?.running, fetchLogs]);

  // Auto-scroll to bottom when new lines appear
  useEffect(() => {
    if (logLines.length > prevLineCountRef.current && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
    prevLineCountRef.current = logLines.length;
  }, [logLines]);

  // ── Settings state ──────────────────────────────────────────────
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [maxCycles, setMaxCycles] = useState(0);
  const [idleLimit, setIdleLimit] = useState(2);
  const [sleepSeconds, setSleepSeconds] = useState(1.0);
  const [noWatch, setNoWatch] = useState(false);
  const [noSentinel, setNoSentinel] = useState(false);

  const handleStart = async () => {
    setActing(true);
    setActionResult(null);
    try {
      const settings: ServiceStartSettings = {
        max_cycles: maxCycles,
        idle_limit: idleLimit,
        sleep_seconds: sleepSeconds,
        no_watch: noWatch,
        no_sentinel: noSentinel,
      };
      const res = await api.startService(settings);
      setActionResult(res.message);
      refresh();
      // Advance tutorial from start-service (step 5) to watch-queue (step 6)
      if (isActive && currentStep === 5) {
        nextStep();
      }
    } catch (err) {
      setActionResult(err instanceof Error ? err.message : String(err));
    } finally {
      setActing(false);
      if (actionResultTimerRef.current) clearTimeout(actionResultTimerRef.current);
      actionResultTimerRef.current = setTimeout(() => setActionResult(null), 5000);
    }
  };

  const handleStop = async () => {
    setActing(true);
    setActionResult(null);
    try {
      const res = await api.stopService();
      setActionResult(res.message);
      refresh();
    } catch (err) {
      setActionResult(err instanceof Error ? err.message : String(err));
    } finally {
      setActing(false);
      if (actionResultTimerRef.current) clearTimeout(actionResultTimerRef.current);
      actionResultTimerRef.current = setTimeout(() => setActionResult(null), 5000);
    }
  };

  const inputClass =
    "w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200 focus:border-brand focus:outline-none";

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] gap-3">
      {error && <ErrorBanner message={error} />}

      {/* ── Control bar ──────────────────────────────────────────── */}
      <div ref={serviceControlRef} className="flex items-center justify-between rounded-lg border border-border bg-surface-100 px-5 py-3 flex-shrink-0">
        <div className="flex items-center gap-4">
          {service ? (
            <>
              <StatusBadge
                status={service.running ? "running" : "stopped"}
                label={service.running ? "Running" : "Stopped"}
              />
              {service.running && (
                <>
                  <div className="h-4 w-px bg-border" />
                  <span className="text-xs text-zinc-500">
                    PID {service.pid}
                  </span>
                  <div className="h-4 w-px bg-border" />
                  <span className="text-xs font-mono text-zinc-400">
                    {formatUptime(service.uptime_seconds)}
                  </span>
                </>
              )}
            </>
          ) : (
            <span className="text-sm text-zinc-600">Connecting...</span>
          )}
        </div>

        {service && (
          service.running ? (
            <button
              onClick={handleStop}
              disabled={acting}
              className="rounded-md px-5 py-2 text-sm font-medium border border-red-500/30 text-red-400 hover:bg-red-500/10 transition-colors duration-150 flex items-center gap-2 disabled:opacity-50"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={acting}
              className="rounded-md px-5 py-2 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors duration-150 flex items-center gap-2 disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              Start Service
            </button>
          )
        )}
      </div>

      {/* ── Log viewer (hero) ────────────────────────────────────── */}
      <div className="flex-1 min-h-0 flex flex-col rounded-lg border border-border bg-surface-100 overflow-hidden">
        <div className="px-5 py-3 border-b border-border flex items-center justify-between flex-shrink-0">
          <h3 className="text-sm font-semibold text-zinc-100">Service Logs</h3>
          {logTotal > 0 && (
            <span className="text-xs text-zinc-500">
              {logLines.length} of {logTotal} lines
            </span>
          )}
        </div>

        <div
          ref={logContainerRef}
          className="flex-1 min-h-0 overflow-y-auto p-4 font-mono text-xs text-zinc-400 bg-surface-500"
        >
          {logLines.length > 0 ? (
            <>
              {logLines.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-all leading-5">
                  {line}
                </div>
              ))}
              {actionResult && (
                <div className="whitespace-pre-wrap break-all leading-5 text-brand bg-brand/5 rounded px-2 py-1 mt-1 border-l-2 border-brand">
                  &gt; {actionResult}
                </div>
              )}
            </>
          ) : service?.running ? (
            <p className="text-zinc-600">Waiting for output...</p>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center">
              {actionResult ? (
                <div className="whitespace-pre-wrap break-all leading-5 text-brand bg-brand/5 rounded px-3 py-2 border-l-2 border-brand text-left self-stretch">
                  &gt; {actionResult}
                </div>
              ) : (
                <>
                  <Play className="w-8 h-8 text-zinc-700 mb-3" />
                  <p className="text-sm text-zinc-500">
                    Start the service to begin processing
                  </p>
                  <p className="text-xs text-zinc-700 mt-1">
                    Logs will stream here in real time
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Advanced settings accordion ──────────────────────────── */}
      <div className="rounded-lg border border-border bg-surface-100 flex-shrink-0">
        <button
          onClick={() => setSettingsOpen(!settingsOpen)}
          className="w-full px-5 py-3 flex items-center justify-between text-left hover:bg-surface-50/50 transition-colors duration-150 rounded-lg"
        >
          <h3 className="text-sm font-semibold text-zinc-100">
            Advanced Settings
          </h3>
          {settingsOpen ? (
            <ChevronDown className="w-4 h-4 text-zinc-500" />
          ) : (
            <ChevronRight className="w-4 h-4 text-zinc-500" />
          )}
        </button>

        {settingsOpen && (
          <div className="px-5 pb-4 border-t border-border pt-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm text-zinc-400 mb-1">
                  Max Cycles
                  <span className="text-zinc-600 ml-1">(0 = unlimited)</span>
                </label>
                <input
                  type="number"
                  min={0}
                  value={maxCycles}
                  onChange={(e) => setMaxCycles(Number(e.target.value))}
                  className={inputClass}
                  disabled={service?.running}
                />
              </div>
              <div>
                <label className="block text-sm text-zinc-400 mb-1">
                  Idle Limit
                  <span className="text-zinc-600 ml-1">(cycles before exit)</span>
                </label>
                <input
                  type="number"
                  min={0}
                  value={idleLimit}
                  onChange={(e) => setIdleLimit(Number(e.target.value))}
                  className={inputClass}
                  disabled={service?.running}
                />
              </div>
              <div>
                <label className="block text-sm text-zinc-400 mb-1">
                  Sleep Interval
                  <span className="text-zinc-600 ml-1">(seconds)</span>
                </label>
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  value={sleepSeconds}
                  onChange={(e) => setSleepSeconds(Number(e.target.value))}
                  className={inputClass}
                  disabled={service?.running}
                />
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-6">
              <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={noWatch}
                  onChange={(e) => setNoWatch(e.target.checked)}
                  disabled={service?.running}
                  className="rounded border-border bg-surface-50 text-brand focus:ring-brand"
                />
                Disable file watching
              </label>
              <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={noSentinel}
                  onChange={(e) => setNoSentinel(e.target.checked)}
                  disabled={service?.running}
                  className="rounded border-border bg-surface-50 text-brand focus:ring-brand"
                />
                Disable sentinel requirement
              </label>
            </div>

            {service?.running && (
              <p className="mt-3 text-xs text-zinc-600">
                Settings cannot be changed while the service is running.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
