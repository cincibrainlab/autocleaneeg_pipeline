import { useState } from "react";
import { Zap, Loader2, FolderOpen } from "lucide-react";
import { api } from "../lib/api";
import type { EventsResponse } from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";
import FolderBrowser from "../components/FolderBrowser";
import EventsDisplay from "../components/EventsDisplay";

// ── File info card ────────────────────────────────────────────────

function FileInfoCard({ info }: { info: NonNullable<EventsResponse["file_info"]> }) {
  const durationSec = info.duration;
  const mins = Math.floor(durationSec / 60);
  const secs = Math.round(durationSec % 60);
  const durationStr =
    mins > 0 ? `${mins}m ${secs}s` : `${durationSec.toFixed(1)}s`;

  return (
    <div className="rounded-lg border border-border bg-surface-100 px-4 py-3 flex flex-wrap gap-x-6 gap-y-1 text-xs">
      <span className="font-mono font-medium text-zinc-200">{info.filename}</span>
      <span className="text-zinc-500">
        {info.n_channels} channel{info.n_channels !== 1 ? "s" : ""}
      </span>
      <span className="text-zinc-500">{info.sfreq} Hz</span>
      <span className="text-zinc-500">{durationStr}</span>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────

export default function EventAnalyzerPage() {
  const [filePath, setFilePath] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EventsResponse | null>(null);
  const [showBrowser, setShowBrowser] = useState(false);

  const handleAnalyze = async () => {
    if (!filePath.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.analyzeEvents(filePath.trim());
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleAnalyze();
  };

  // FolderBrowser selects a directory — user still needs to type the filename.
  // We use it to navigate to the directory quickly, then complete the filename.
  const handleBrowseSelect = (path: string) => {
    // If path looks like a directory, add trailing slash for the operator to
    // complete. The browser currently only returns directories, so we do a
    // best-effort append.
    setFilePath(path.endsWith("/") ? path : path + "/");
    setShowBrowser(false);
  };

  return (
    <div className="space-y-5 max-w-3xl">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
          <Zap className="w-5 h-5 text-brand" />
          Event Analyzer
        </h2>
        <p className="text-xs text-zinc-500 mt-0.5">
          Utility tool for file-level event inspection
        </p>
        <p className="mt-2 max-w-2xl text-xs text-zinc-500">
          Events is intentionally not route-scoped. Use it when you need to inspect one file’s markers, timing, and event structure outside the route processing workflow.
        </p>
      </div>

      {/* File input row */}
      <div className="flex items-stretch gap-2">
        <div className="flex-1 relative">
          <input
            type="text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="/path/to/subject.set"
            className="w-full px-3 py-2 text-sm bg-surface-50 border border-border rounded text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-brand/60 font-mono"
          />
        </div>
        <button
          onClick={() => setShowBrowser(true)}
          title="Browse to folder"
          className="flex items-center gap-1.5 px-3 py-2 rounded border border-border bg-surface-50 text-zinc-400 hover:text-zinc-200 hover:bg-surface-50/60 transition-colors text-xs"
        >
          <FolderOpen className="w-4 h-4" />
          Browse
        </button>
        <button
          onClick={handleAnalyze}
          disabled={loading || !filePath.trim()}
          className="flex items-center gap-1.5 px-4 py-2 rounded bg-brand text-surface-500 text-sm font-medium hover:bg-brand/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Analyzing&hellip;
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              Analyze
            </>
          )}
        </button>
      </div>

      {/* Hint text */}
      <div className="rounded-lg border border-border bg-surface-100/60 px-3 py-2 text-[11px] text-zinc-500">
        Supported formats: .set, .edf, .bdf, .fif, .vhdr. Browse to a folder, then complete the filename in the input. This utility reads one file at a time and does not depend on route selection.
      </div>

      {/* Error */}
      {error && <ErrorBanner message={error} />}

      {/* File info card */}
      {result?.file_info && <FileInfoCard info={result.file_info} />}

      {/* Results visualization */}
      {result && (
        <div className="rounded-lg border border-border bg-surface-100 p-5">
          <EventsDisplay data={result} />
        </div>
      )}

      {/* Empty state before first analysis */}
      {!result && !loading && !error && (
        <div className="flex flex-col items-center justify-center py-16 gap-3 text-zinc-600">
          <Zap className="w-8 h-8 opacity-30" />
          <p className="text-sm">Enter one file path and click Analyze</p>
        </div>
      )}

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
