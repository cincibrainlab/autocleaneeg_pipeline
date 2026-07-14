import { useState, useMemo, useEffect, useCallback, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import {
  FileCheck,
  Search,
  X,
  ChevronDown,
  ChevronUp,
  ArrowUpDown,
  CheckCircle2,
  XCircle,
  Loader2,
  FileText,
  BarChart3,
  Layers,
  Braces,
  LayoutDashboard,
  Zap,
  Download,
} from "lucide-react";
import { api } from "../lib/api";
import type { RunSummary, RunDetail, IcaComponent, IcaSummaryResponse, EventsResponse, RouteSpec } from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";
import CodeViewer from "../components/CodeViewer";
import EventsDisplay from "../components/EventsDisplay";
import { usePolling } from "../hooks/usePolling";
import { formatTime } from "../lib/format";

// ── Types ──────────────────────────────────────────────────────────

type SortKey = "filename" | "task" | "status" | "created_at";
type SortDir = "asc" | "desc";
type StatusFilter = "all" | "completed" | "failed";
type DetailTab = "summary" | "report" | "plots" | "ica" | "events" | "metadata";

// ── Helpers ────────────────────────────────────────────────────────

function formatDate(iso: string): string {
  if (!iso) return "—";
  // Normalize SQLite timestamps (missing T separator and timezone)
  const normalized = iso.includes("T") ? iso : iso.replace(" ", "T") + "Z";
  return formatTime(normalized);
}

function pct(numerator: number | null, denominator: number | null): string {
  if (numerator == null || denominator == null || denominator === 0) return "";
  return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function MiniProgress({
  value,
  max,
  color = "bg-brand",
}: {
  value: number | null;
  max: number | null;
  color?: string;
}) {
  const frac =
    value != null && max != null && max > 0
      ? Math.min(1, Math.max(0, value / max))
      : 0;
  return (
    <div className="mt-1.5 h-1.5 w-full rounded-full bg-surface-50 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all ${color}`}
        style={{ width: `${frac * 100}%` }}
      />
    </div>
  );
}

// ── Sort header ────────────────────────────────────────────────────

function SortHeader({
  label,
  sortKey,
  currentKey,
  dir,
  onSort,
  className: cls,
}: {
  label: string;
  sortKey: SortKey;
  currentKey: SortKey;
  dir: SortDir;
  onSort: (k: SortKey) => void;
  className?: string;
}) {
  const active = currentKey === sortKey;
  return (
    <th
      className={`px-3 py-2 text-left text-[10px] uppercase text-zinc-500 font-medium tracking-wider cursor-pointer select-none hover:text-zinc-300 transition-colors ${cls ?? ""}`}
      onClick={() => onSort(sortKey)}
    >
      <span className="flex items-center gap-1">
        {label}
        {active ? (
          dir === "asc" ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )
        ) : (
          <ArrowUpDown className="w-3 h-3 opacity-30" />
        )}
      </span>
    </th>
  );
}

// ── Tab button ────────────────────────────────────────────────────

function TabButton({
  tab,
  active,
  label,
  icon: Icon,
  onClick,
}: {
  tab: DetailTab;
  active: DetailTab;
  label: string;
  icon: React.ElementType;
  onClick: (t: DetailTab) => void;
}) {
  return (
    <button
      onClick={() => onClick(tab)}
      className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition-colors whitespace-nowrap ${
        active === tab
          ? "border-brand text-brand"
          : "border-transparent text-zinc-500 hover:text-zinc-300"
      }`}
    >
      <Icon className="w-3.5 h-3.5" />
      {label}
    </button>
  );
}

// ── Status badge ──────────────────────────────────────────────────

function StatusBadge({ run }: { run: RunSummary }) {
  if (run.success) {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/15 text-emerald-400">
        <CheckCircle2 className="w-3 h-3" />
        {run.status || "completed"}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-500/15 text-red-400">
      <XCircle className="w-3 h-3" />
      {run.status || "failed"}
    </span>
  );
}

// ── Metric card ───────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  sub,
  numerator,
  denominator,
  progressColor,
}: {
  label: string;
  value: string;
  sub?: string;
  numerator?: number | null;
  denominator?: number | null;
  progressColor?: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface-50 px-4 py-3">
      <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-1">
        {label}
      </p>
      <p className="text-sm font-semibold text-zinc-100">{value}</p>
      {sub && <p className="text-xs text-zinc-500 mt-0.5">{sub}</p>}
      {numerator != null && denominator != null && (
        <MiniProgress
          value={numerator}
          max={denominator}
          color={progressColor}
        />
      )}
    </div>
  );
}

// ── Asset unavailable placeholder ─────────────────────────────────

function Unavailable({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3 text-zinc-600">
      <FileCheck className="w-8 h-8 opacity-40" />
      <p className="text-sm">{label} not available</p>
    </div>
  );
}

function PdfViewer({
  title,
  src,
  linkLabel,
}: {
  title: string;
  src: string;
  linkLabel: string;
}) {
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <a
          href={src}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-brand hover:underline"
        >
          {linkLabel}
        </a>
      </div>
      <iframe
        key={src}
        title={title}
        src={src}
        className="w-full min-h-[38rem] rounded-lg border border-border bg-white"
      />
    </div>
  );
}

// ── Summary tab ───────────────────────────────────────────────────

function SummaryTab({ detail }: { detail: RunDetail }) {
  const m = detail.metrics;

  const chPct = pct(m.channels_retained, m.channels_original);
  const epPct = pct(m.epochs_kept, m.epochs_total);
  const icaRetained =
    m.ica_n_components != null
      ? m.ica_n_components - m.ica_removed.length
      : null;
  const icaPct = pct(icaRetained, m.ica_n_components);

  return (
    <div className="space-y-5">
      {/* 2×2 metric cards */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Channels Retained"
          value={
            m.channels_original > 0
              ? `${m.channels_retained} / ${m.channels_original}`
              : m.channels_retained > 0
              ? String(m.channels_retained)
              : "—"
          }
          sub={chPct ? `${chPct} retained` : undefined}
          numerator={m.channels_retained}
          denominator={m.channels_original || null}
          progressColor="bg-brand"
        />
        <MetricCard
          label="Epochs Kept"
          value={
            m.epochs_total != null && m.epochs_kept != null
              ? `${m.epochs_kept} / ${m.epochs_total}`
              : m.epochs_kept != null
              ? String(m.epochs_kept)
              : "—"
          }
          sub={epPct ? `${epPct} kept` : undefined}
          numerator={m.epochs_kept}
          denominator={m.epochs_total}
          progressColor="bg-violet-500"
        />
        <MetricCard
          label="ICA Components"
          value={
            m.ica_n_components != null
              ? `${icaRetained} / ${m.ica_n_components} retained`
              : "—"
          }
          sub={icaPct ? `${m.ica_removed.length} removed` : undefined}
          numerator={icaRetained}
          denominator={m.ica_n_components}
          progressColor="bg-amber-500"
        />
        <MetricCard
          label="Duration"
          value={
            m.duration_post != null
              ? `${m.duration_post.toFixed(0)}s`
              : m.duration_raw != null
              ? `${m.duration_raw.toFixed(0)}s`
              : "—"
          }
          sub={
            m.duration_raw != null && m.duration_post != null
              ? `raw ${m.duration_raw.toFixed(0)}s`
              : undefined
          }
          numerator={m.duration_post}
          denominator={m.duration_raw}
          progressColor="bg-cyan-500"
        />
      </div>

      {/* Channel removals */}
      {m.bad_channels.length > 0 && (
        <div>
          <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
            Channel Removals ({m.bad_channels.length})
          </p>
          <div className="rounded border border-border overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-surface-100">
                  <th className="px-3 py-1.5 text-left text-[10px] uppercase text-zinc-500 font-medium">
                    Channel
                  </th>
                  <th className="px-3 py-1.5 text-left text-[10px] uppercase text-zinc-500 font-medium">
                    Reason
                  </th>
                </tr>
              </thead>
              <tbody>
                {m.bad_channels.map((ch, i) => (
                  <tr
                    key={i}
                    className="border-b border-border-subtle last:border-0 hover:bg-surface-50/30"
                  >
                    <td className="px-3 py-1.5 font-mono font-medium text-zinc-200">
                      {ch.channel}
                    </td>
                    <td className="px-3 py-1.5 text-zinc-400">{ch.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Processing params */}
      <div>
        <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
          Processing Parameters
        </p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
          {m.filter_low != null && (
            <>
              <span className="text-zinc-500">Low-pass filter</span>
              <span className="text-zinc-200 font-mono">{m.filter_high ?? "—"} Hz</span>
              <span className="text-zinc-500">High-pass filter</span>
              <span className="text-zinc-200 font-mono">{m.filter_low} Hz</span>
            </>
          )}
          {(m.notch_freqs ?? []).length > 0 && (
            <>
              <span className="text-zinc-500">Notch filters</span>
              <span className="text-zinc-200 font-mono">
                {(m.notch_freqs ?? []).join(", ")} Hz
              </span>
            </>
          )}
          {m.ica_method && (
            <>
              <span className="text-zinc-500">ICA method</span>
              <span className="text-zinc-200 font-mono">{m.ica_method}</span>
            </>
          )}
          {m.sample_rate != null && (
            <>
              <span className="text-zinc-500">Sample rate</span>
              <span className="text-zinc-200 font-mono">{m.sample_rate} Hz</span>
            </>
          )}
        </div>
      </div>

      {/* Error */}
      {detail.error && (
        <div className="rounded border border-red-500/30 bg-red-500/10 px-3 py-2">
          <p className="text-[10px] uppercase font-medium text-red-400 mb-1">
            Error
          </p>
          <p className="text-xs text-red-300 font-mono break-all">{detail.error}</p>
        </div>
      )}
    </div>
  );
}

// ── Metadata tab ──────────────────────────────────────────────────

function MetadataTab({ runId }: { runId: string }) {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setLoading(true);
    setError(null);
    api
      .getRunMetadata(runId)
      .then((d) => {
        if (!cancelled) { setData(d); setLoading(false); }
      })
      .catch((e: unknown) => {
        if (!cancelled) { setError(e instanceof Error ? e.message : String(e)); setLoading(false); }
      });
    return () => { cancelled = true; };
  }, [runId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 gap-2 text-zinc-500 text-xs">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading metadata&hellip;
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-1">
        <ErrorBanner message={error} />
      </div>
    );
  }

  if (!data) {
    return <Unavailable label="Metadata" />;
  }

  const lines = JSON.stringify(data, null, 2).split("\n");

  const colorize = (line: string) => {
    // Key: "something":
    if (/^\s+"[^"]+"\s*:/.test(line)) {
      const colonIdx = line.indexOf(":");
      const keyPart = line.substring(0, colonIdx + 1);
      const rest = line.substring(colonIdx + 1);
      return (
        <>
          <span className="text-sky-400">{keyPart}</span>
          <span className="text-zinc-300">{rest}</span>
        </>
      );
    }
    // String value
    if (/^\s+"/.test(line)) return <span className="text-amber-300">{line}</span>;
    // Number or bool
    if (/:\s*(true|false|null|\d)/.test(line)) {
      return <span className="text-violet-300">{line}</span>;
    }
    return <span className="text-zinc-400">{line}</span>;
  };

  return (
    <div className="rounded border border-border overflow-hidden">
      <CodeViewer lines={lines} colorize={colorize} maxHeight="480px" />
    </div>
  );
}

// ── Events Tab ────────────────────────────────────────────────────

function EventsTab({ runId }: { runId: string }) {
  const [data, setData] = useState<EventsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setLoading(true);
    setError(null);
    api
      .getRunEvents(runId)
      .then((d) => {
        if (!cancelled) { setData(d); setLoading(false); }
      })
      .catch((e: unknown) => {
        if (!cancelled) { setError(e instanceof Error ? e.message : String(e)); setLoading(false); }
      });
    return () => { cancelled = true; };
  }, [runId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 gap-2 text-zinc-500 text-xs">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading event data&hellip;
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-1">
        <ErrorBanner message={error} />
      </div>
    );
  }

  if (!data) {
    return <Unavailable label="Event data" />;
  }

  return <EventsDisplay data={data} />;
}

// ── ICA Tab ───────────────────────────────────────────────────────

const ICA_TYPE_COLORS: Record<string, string> = {
  brain: "bg-emerald-500/15 text-emerald-400",
  eog: "bg-amber-500/15 text-amber-400",
  muscle: "bg-pink-500/15 text-pink-400",
  ecg: "bg-red-500/15 text-red-400",
  ch_noise: "bg-orange-500/15 text-orange-400",
  line_noise: "bg-blue-500/15 text-blue-400",
  other: "bg-zinc-500/15 text-zinc-400",
};

function IcaComponentCard({
  comp,
  pageNum,
  onClick,
}: {
  comp: IcaComponent;
  pageNum: number | undefined;
  onClick: () => void;
}) {
  const colorClass =
    ICA_TYPE_COLORS[comp.type] ?? "bg-zinc-500/15 text-zinc-400";
  const pct = Math.round(comp.confidence * 100);
  return (
    <button
      onClick={pageNum !== undefined ? onClick : undefined}
      disabled={pageNum === undefined}
      className={`rounded-lg border p-3 text-left transition-colors w-full ${
        comp.rejected
          ? "border-red-500/40 bg-red-500/5 hover:bg-red-500/10"
          : "border-border bg-surface-50 hover:bg-surface-50/60"
      } ${pageNum === undefined ? "opacity-50 cursor-default" : "cursor-pointer"}`}
    >
      <p className="text-xs font-semibold font-mono text-zinc-200 mb-1.5">
        {comp.component}
      </p>
      <div className="flex items-center gap-1.5 flex-wrap">
        <span
          className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${colorClass}`}
        >
          {comp.type}
        </span>
        <span className="text-[10px] text-zinc-500">{pct}%</span>
      </div>
      {comp.rejected && (
        <p className="mt-1.5 text-[10px] font-semibold text-red-400 uppercase tracking-wide">
          Rejected
        </p>
      )}
    </button>
  );
}

type IcaView = "grid" | "topo" | { ic: string; pageNum: number };

function IcaTab({
  runId,
  available,
}: {
  runId: string;
  available: boolean;
}) {
  const [summary, setSummary] = useState<IcaSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<IcaView>("grid");

  useEffect(() => {
    if (!available) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSummary(null);
    setView("grid");
    api
      .getIcaSummary(runId)
      .then((d) => {
        if (!cancelled) {
          setSummary(d);
          setLoading(false);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [runId, available]);

  if (!available) {
    return <Unavailable label="ICA report" />;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 gap-2 text-zinc-500 text-xs">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading ICA components&hellip;
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="px-1">
        <ErrorBanner message={error ?? "Failed to load ICA data"} />
      </div>
    );
  }

  const { components, structure } = summary;
  const pageUrl = (n: number) =>
    `/api/results/${encodeURIComponent(runId)}/ica/page/${n}`;
  const icaReportUrl = api.getRunIcaReportUrl(runId);

  // ── Detail / topo view ──
  if (view === "topo") {
    const topoPage = structure.topo_grid_page;
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setView("grid")}
              className="text-xs text-brand hover:underline"
            >
              &larr; Back to components
            </button>
            <span className="text-xs text-zinc-500">Topography Grid</span>
          </div>
          <a
            href={typeof topoPage === "number" ? `${icaReportUrl}#page=${topoPage + 1}` : icaReportUrl}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-brand hover:underline"
          >
            Open ICA PDF in new tab
          </a>
        </div>
        {topoPage !== null ? (
          <div className="space-y-3">
            <iframe
              key={`${icaReportUrl}#page=${topoPage + 1}`}
              title="ICA report"
              src={`${icaReportUrl}#page=${topoPage + 1}`}
              className="w-full min-h-[38rem] rounded-lg border border-border bg-white"
            />
            <img
              src={pageUrl(topoPage)}
              alt="ICA topography grid"
              className="rounded border border-border max-w-full"
            />
          </div>
        ) : (
          <Unavailable label="Topo grid page" />
        )}
      </div>
    );
  }

  if (typeof view === "object" && "ic" in view) {
    const { ic, pageNum } = view;
    const comp = components.find((c) => c.component === ic);
    const colorClass =
      comp ? (ICA_TYPE_COLORS[comp.type] ?? "bg-zinc-500/15 text-zinc-400") : "";
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={() => setView("grid")}
              className="text-xs text-brand hover:underline"
            >
              &larr; Back to components
            </button>
            <span className="text-xs font-mono font-semibold text-zinc-200">
              {ic}
            </span>
            {comp && (
              <>
                <span
                  className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${colorClass}`}
                >
                  {comp.type}
                </span>
                <span className="text-[10px] text-zinc-500">
                  {Math.round(comp.confidence * 100)}% confidence
                </span>
                {comp.rejected && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-red-500/15 text-red-400 uppercase tracking-wide">
                    Rejected
                  </span>
                )}
              </>
            )}
          </div>
          <a
            href={`${icaReportUrl}#page=${pageNum + 1}`}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-brand hover:underline"
          >
            Open ICA PDF in new tab
          </a>
        </div>
        <div className="space-y-3">
          <iframe
            key={`${icaReportUrl}#page=${pageNum + 1}`}
            title={`ICA report ${ic}`}
            src={`${icaReportUrl}#page=${pageNum + 1}`}
            className="w-full min-h-[38rem] rounded-lg border border-border bg-white"
          />
          <img
            src={pageUrl(pageNum)}
            alt={`${ic} detail`}
            className="rounded border border-border max-w-full"
          />
        </div>
      </div>
    );
  }

  // ── Grid view ──
  const n_rejected = components.filter((c) => c.rejected).length;
  const n_total = components.length;

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="text-xs text-zinc-500">
          {n_total} component{n_total !== 1 ? "s" : ""}
          {n_rejected > 0 && (
            <span className="ml-2 text-red-400">
              {n_rejected} rejected
            </span>
          )}
        </div>
        {structure.topo_grid_page !== null && (
          <div className="flex items-center gap-3">
            <button
              onClick={() => setView("topo")}
              className="px-2.5 py-1 rounded text-xs font-medium border border-border bg-surface-50 text-zinc-300 hover:bg-surface-50/60 transition-colors"
            >
              Topo Grid
            </button>
            <a
              href={icaReportUrl}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-brand hover:underline"
            >
              Open ICA PDF in new tab
            </a>
          </div>
        )}
        {structure.topo_grid_page === null && (
          <a
            href={icaReportUrl}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-brand hover:underline"
          >
            Open ICA PDF in new tab
          </a>
        )}
      </div>

      <iframe
        key={icaReportUrl}
        title="ICA report"
        src={icaReportUrl}
        className="w-full min-h-[32rem] rounded-lg border border-border bg-white"
      />

      {n_total === 0 ? (
        <div className="py-8 text-center text-xs text-zinc-500">
          No component data found in this PDF.
          <br />
          <span className="text-zinc-600">
            The summary table format may not match the expected layout.
          </span>
        </div>
      ) : (
        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
          {components.map((comp) => {
            const pageNum = structure.detail_page_map[comp.component];
            return (
              <IcaComponentCard
                key={comp.component}
                comp={comp}
                pageNum={pageNum}
                onClick={() => {
                  if (pageNum !== undefined) {
                    setView({ ic: comp.component, pageNum });
                  }
                }}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Detail panel ──────────────────────────────────────────────────

type DecisionValue = "pass" | "fail" | "review" | null;

const DECISION_COLORS: Record<NonNullable<DecisionValue>, string> = {
  pass: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  fail: "bg-red-500/15 text-red-400 border-red-500/30",
  review: "bg-amber-500/15 text-amber-400 border-amber-500/30",
};

const DECISION_BUTTONS: ReadonlyArray<{ key: NonNullable<DecisionValue>; label: string; shortcut: string; colors: string }> = [
  { key: "pass", label: "Pass", shortcut: "P", colors: "hover:bg-emerald-500/20 hover:text-emerald-400" },
  { key: "fail", label: "Fail", shortcut: "F", colors: "hover:bg-red-500/20 hover:text-red-400" },
  { key: "review", label: "Review", shortcut: "R", colors: "hover:bg-amber-500/20 hover:text-amber-400" },
];

function isTypingInInput(e: KeyboardEvent): boolean {
  const tag = (e.target as HTMLElement)?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if ((e.target as HTMLElement)?.isContentEditable) return true;
  return false;
}

export function DecisionBar({
  runId,
  currentDecision,
  currentNotes,
  onDecisionChange,
}: {
  runId: string;
  currentDecision: DecisionValue;
  currentNotes: string;
  onDecisionChange: (decision: DecisionValue, notes: string) => void;
}) {
  const [notes, setNotes] = useState(currentNotes);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(false);
  const [failedDecision, setFailedDecision] = useState<DecisionValue>(null);
  const notesRef = useRef(notes);
  notesRef.current = notes;

  useEffect(() => { setNotes(currentNotes); }, [currentNotes]);

  const handleDecision = useCallback(async (decision: DecisionValue) => {
    setSaving(true);
    try {
      if (decision === null) {
        await api.setDecision(runId, "clear");
      } else {
        await api.setDecision(runId, decision, notesRef.current);
      }
      onDecisionChange(decision, notesRef.current);
      setSaveError(false);
      setFailedDecision(null);
    } catch {
      setSaveError(true);
      setFailedDecision(decision);
    } finally {
      setSaving(false);
    }
  }, [runId, onDecisionChange]);

  // Keyboard shortcuts for decisions
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (isTypingInInput(e)) return;
      if (e.key === "p" || e.key === "P") { handleDecision("pass"); return; }
      if (e.key === "f" || e.key === "F") { handleDecision("fail"); return; }
      if (e.key === "r" || e.key === "R") { handleDecision("review"); return; }
      if (e.key === "c" || e.key === "C") { handleDecision(null); return; }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [handleDecision]);

  return (
    <div className="px-5 py-3 border-t border-border bg-surface-200 flex-shrink-0">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-medium text-zinc-500">Decision:</span>
        {DECISION_BUTTONS.map((b) => (
          <button
            key={b.key}
            onClick={() => handleDecision(b.key)}
            disabled={saving}
            className={[
              "px-2.5 py-1 rounded text-xs font-medium border transition-colors",
              currentDecision === b.key
                ? DECISION_COLORS[b.key]
                : `border-border text-zinc-500 ${b.colors}`,
            ].join(" ")}
          >
            {b.label}
            <kbd className="ml-1.5 text-[10px] opacity-50">{b.shortcut}</kbd>
          </button>
        ))}
        {currentDecision && (
          <button
            onClick={() => handleDecision(null)}
            disabled={saving}
            className="px-2 py-1 rounded text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            Clear
            <kbd className="ml-1 text-[10px] opacity-50">C</kbd>
          </button>
        )}
      </div>
      {saveError && (
        <div role="alert" className="mb-2 flex items-center gap-2 text-xs text-red-400">
          <span>Decision could not be saved.</span>
          <button
            type="button"
            disabled={saving}
            onClick={() => handleDecision(failedDecision)}
            className="underline underline-offset-2 hover:text-red-300 disabled:opacity-50"
          >
            Retry
          </button>
        </div>
      )}
      <input
        type="text"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        onBlur={() => {
          if (!saveError && currentDecision) handleDecision(currentDecision);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !saveError && currentDecision) {
            handleDecision(currentDecision);
          }
        }}
        placeholder="Add notes..."
        className="w-full px-2.5 py-1.5 rounded bg-surface-50 border border-border text-xs text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-brand/40"
      />
    </div>
  );
}


function RunDetailPanel({
  run,
  onClose,
  decision,
  decisionNotes,
  onDecisionChange,
}: {
  run: RunSummary;
  onClose: () => void;
  decision: DecisionValue;
  decisionNotes: string;
  onDecisionChange: (runId: string, decision: DecisionValue, notes: string) => void;
}) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(true);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<DetailTab>("summary");

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setLoadingDetail(true);
    setDetailError(null);
    setActiveTab("summary");
    api
      .getRunDetail(run.run_id)
      .then((d) => {
        if (!cancelled) { setDetail(d); setLoadingDetail(false); }
      })
      .catch((e: unknown) => {
        if (!cancelled) { setDetailError(e instanceof Error ? e.message : String(e)); setLoadingDetail(false); }
      });
    return () => { cancelled = true; };
  }, [run.run_id]);

  const tabs: Array<{ tab: DetailTab; label: string; icon: React.ElementType }> = [
    { tab: "summary", label: "Summary", icon: LayoutDashboard },
    { tab: "report", label: "Run Report", icon: FileText },
    { tab: "plots", label: "PSD & Plots", icon: BarChart3 },
    { tab: "ica", label: "ICA", icon: Layers },
    { tab: "events", label: "Events", icon: Zap },
    { tab: "metadata", label: "Metadata", icon: Braces },
  ];

  return (
    <div className="flex-1 min-w-0 rounded-lg border border-border bg-surface-100 overflow-hidden flex flex-col">
      {/* Panel header */}
      <div className="px-5 py-3 border-b border-border flex items-start justify-between gap-4 flex-shrink-0">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {run.success ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            ) : (
              <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            )}
            <h3 className="text-sm font-semibold text-zinc-100 truncate font-mono">
              {run.filename}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">{run.task}</span>
            <span className="text-zinc-700">·</span>
            <span className="text-xs text-zinc-600">{formatDate(run.created_at)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <a
            href={api.getRunDownloadUrl(run.run_id)}
            download
            className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-300 hover:bg-surface-50 transition-colors"
            title="Download all artifacts (ZIP)"
          >
            <Download className="w-4 h-4" />
          </a>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-300"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border overflow-x-auto flex-shrink-0 bg-surface-100">
        {tabs.map((t) => (
          <TabButton
            key={t.tab}
            tab={t.tab}
            active={activeTab}
            label={t.label}
            icon={t.icon}
            onClick={setActiveTab}
          />
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto p-5">
        {loadingDetail && (
          <div className="flex items-center justify-center py-12 gap-2 text-zinc-500 text-xs">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading run details&hellip;
          </div>
        )}

        {detailError && !loadingDetail && (
          <ErrorBanner message={detailError} />
        )}

        {detail && !loadingDetail && (
          <>
            {activeTab === "summary" && <SummaryTab detail={detail} />}

            {activeTab === "report" && (
              detail.assets.report ? (
                <PdfViewer
                  title="Run report"
                  src={api.getRunReportUrl(run.run_id)}
                  linkLabel="Open report in new tab"
                />
              ) : (
                <Unavailable label="Run report" />
              )
            )}

            {activeTab === "plots" && (
              <div className="space-y-6">
                {detail.assets.psd ? (
                  <div className="space-y-3">
                    <div className="flex justify-end">
                      <a
                        href={api.getRunPsdUrl(run.run_id)}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs text-brand hover:underline"
                      >
                        Open PSD in new tab
                      </a>
                    </div>
                    <img
                      src={api.getRunPsdUrl(run.run_id)}
                      alt="PSD topomap"
                      className="w-full rounded-lg border border-border"
                    />
                  </div>
                ) : (
                  <div>
                    <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-3">
                      PSD Topomap
                    </p>
                    <Unavailable label="PSD topomap" />
                  </div>
                )}
                {detail.assets.overlay ? (
                  <div>
                    <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-3">
                      Raw vs Cleaned Overlay
                    </p>
                    <img
                      src={`/api/results/${encodeURIComponent(run.run_id)}/overlay`}
                      alt="Raw vs cleaned overlay"
                      className="rounded border border-border max-w-full"
                    />
                  </div>
                ) : (
                  <div>
                    <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-3">
                      Raw vs Cleaned Overlay
                    </p>
                    <Unavailable label="Overlay" />
                  </div>
                )}
              </div>
            )}

            {activeTab === "ica" && (
              detail.assets.ica_report ? (
                <PdfViewer
                  title="ICA report"
                  src={api.getRunIcaReportUrl(run.run_id)}
                  linkLabel="Open ICA PDF in new tab"
                />
              ) : (
                <Unavailable label="ICA report" />
              )
            )}

            {activeTab === "events" && (
              <EventsTab runId={run.run_id} />
            )}

            {activeTab === "metadata" && (
              <MetadataTab runId={run.run_id} />
            )}
          </>
        )}
      </div>

      {/* Decision bar */}
      <DecisionBar
        key={run.run_id}
        runId={run.run_id}
        currentDecision={decision}
        currentNotes={decisionNotes}
        onDecisionChange={(d, n) => onDecisionChange(run.run_id, d, n)}
      />
    </div>
  );
}

// ── Sorting ────────────────────────────────────────────────────────

function sortRuns(runs: RunSummary[], key: SortKey, dir: SortDir): RunSummary[] {
  const mult = dir === "asc" ? 1 : -1;
  return [...runs].sort((a, b) => {
    switch (key) {
      case "filename":
        return a.filename.localeCompare(b.filename, undefined, { numeric: true }) * mult;
      case "task":
        return a.task.localeCompare(b.task) * mult;
      case "status":
        return a.status.localeCompare(b.status) * mult;
      case "created_at":
        return a.created_at.localeCompare(b.created_at) * mult;
    }
  });
}

// ── Main page ──────────────────────────────────────────────────────

export default function ResultsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedRoute = searchParams.get("route") || "";
  const taskFilter = searchParams.get("task") || "";
  const montageFilter = searchParams.get("montage") || "";
  const {
    data: resultsData,
    error,
    loading,
  } = usePolling<{ runs: RunSummary[]; total: number }>(
    () => (selectedRoute ? api.getResults(selectedRoute) : Promise.resolve({ runs: [], total: 0 })),
    30000,
  );
  const { data: routes } = usePolling<RouteSpec[]>(api.getRoutes, 30000);

  const runs = resultsData?.runs ?? null;

  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [selected, setSelected] = useState<RunSummary | null>(null);
  const [decisions, setDecisions] = useState<Record<string, { decision: string; notes: string }>>({});
  const routeOptions = useMemo(() => {
    const allRoutes = routes ?? [];
    return allRoutes.filter((route) => {
      if (taskFilter) {
        const taskName = route.taskfile.split("/").pop()?.replace(".py", "") || route.taskfile;
        if (taskName !== taskFilter) return false;
      }
      if (montageFilter && route.montage !== montageFilter) return false;
      return true;
    });
  }, [routes, taskFilter, montageFilter]);
  const availableTasks = useMemo(
    () => [...new Set((routes ?? []).map((route) => route.taskfile.split("/").pop()?.replace(".py", "") || route.taskfile))].sort(),
    [routes],
  );
  const availableMontages = useMemo(
    () => [...new Set((routes ?? []).map((route) => route.montage))].sort(),
    [routes],
  );

  useEffect(() => {
    if (selectedRoute || routeOptions.length !== 1) return;
    const onlyRoute = routeOptions[0];
    if (!onlyRoute) return;
    const next = new URLSearchParams(searchParams);
    next.set("route", onlyRoute.id);
    setSearchParams(next, { replace: true });
  }, [routeOptions, searchParams, selectedRoute, setSearchParams]);

  const updateContextParam = (key: "route" | "task" | "montage", value: string) => {
    const next = new URLSearchParams(searchParams);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== "route" && selectedRoute) {
      const stillValid = routeOptions.some((route) => route.id === selectedRoute);
      if (!stillValid) next.delete("route");
    }
    setSearchParams(next, { replace: true });
    setSelected(null);
  };

  // Load decisions on mount
  useEffect(() => {
    api.getDecisions().then((data) => {
      const map: Record<string, { decision: string; notes: string }> = {};
      for (const [runId, rec] of Object.entries(data.decisions)) {
        map[runId] = { decision: rec.decision, notes: rec.notes };
      }
      setDecisions(map);
    }).catch(() => {});
  }, []);

  const handleDecisionChange = (runId: string, decision: DecisionValue, notes: string) => {
    setDecisions((prev) => {
      const next = { ...prev };
      if (decision === null) {
        delete next[runId];
      } else {
        next[runId] = { decision, notes };
      }
      return next;
    });
  };

  const filtered = useMemo(() => {
    if (!runs) return [];
    if (!selectedRoute) return [];
    let list = runs;
    if (statusFilter === "completed") {
      list = list.filter((r) => r.success);
    } else if (statusFilter === "failed") {
      list = list.filter((r) => !r.success);
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (r) =>
          r.filename.toLowerCase().includes(q) ||
          r.task.toLowerCase().includes(q) ||
          r.status.toLowerCase().includes(q)
      );
    }
    return sortRuns(list, sortKey, sortDir);
  }, [runs, statusFilter, searchQuery, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  // Keep selected run in sync with polling data, or clear if filtered out
  useEffect(() => {
    if (!selected) return;
    const fresh = filtered.find((r) => r.run_id === selected.run_id);
    if (!fresh) setSelected(null);
    else if (fresh !== selected) setSelected(fresh);
  }, [filtered, selected]);

  const filterButtons: Array<{ key: StatusFilter; label: string }> = [
    { key: "all", label: "All" },
    { key: "completed", label: "Completed" },
    { key: "failed", label: "Failed" },
  ];

  return (
    <div className="space-y-4 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-shrink-0">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">Results</h2>
          <p className="text-xs text-zinc-500 mt-0.5">
            Route-scoped run review with reports and metrics
          </p>
          {resultsData && (
            <p className="text-xs text-zinc-600 mt-1">
            {selectedRoute
              ? `${resultsData.total} run${resultsData.total !== 1 ? "s" : ""}`
              : "Select a route to view results"}
            </p>
          )}
        </div>
        {resultsData && resultsData.total > 0 && (
          <a
            href={api.getResultsCsvUrl()}
            download
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-border text-xs font-medium text-zinc-400 hover:text-zinc-200 hover:bg-surface-50 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </a>
        )}
      </div>

      {error && <ErrorBanner message={error} />}

      <div className="grid gap-3 rounded-lg border border-border bg-surface-100 p-4 xl:grid-cols-[minmax(0,1fr)_14rem_14rem_18rem]">
        <div>
          <p className="text-xs font-medium text-zinc-300">
            {selectedRoute ? `Showing results for route '${selectedRoute}'` : "Select a route to focus review results"}
          </p>
          <p className="mt-1 text-xs text-zinc-500">
            Results are reviewed in route context. Use task and montage filters to narrow the route list when the workspace grows.
          </p>
        </div>
        <label className="block text-[11px] uppercase tracking-wider text-zinc-500">
          Task Filter
          <select
            value={taskFilter}
            onChange={(event) => updateContextParam("task", event.target.value)}
            className="mt-2 w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
          >
            <option value="">All tasks</option>
            {availableTasks.map((task) => (
              <option key={task} value={task}>{task}</option>
            ))}
          </select>
        </label>
        <label className="block text-[11px] uppercase tracking-wider text-zinc-500">
          Montage Filter
          <select
            value={montageFilter}
            onChange={(event) => updateContextParam("montage", event.target.value)}
            className="mt-2 w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
          >
            <option value="">All montages</option>
            {availableMontages.map((montage) => (
              <option key={montage} value={montage}>{montage}</option>
            ))}
          </select>
        </label>
        <label className="block text-[11px] uppercase tracking-wider text-zinc-500">
          Route
          <select
            value={selectedRoute}
            onChange={(event) => updateContextParam("route", event.target.value)}
            className="mt-2 w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
          >
            <option value="">Select route</option>
            {routeOptions.map((route) => (
              <option key={route.id} value={route.id}>
                {route.id} · {route.taskfile.split("/").pop()?.replace(".py", "") || route.taskfile} · {route.montage}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-2 flex-wrap flex-shrink-0">
        {/* Status filter tabs */}
        <div className="flex items-center gap-1">
          {filterButtons.map((f) => (
            <button
              key={f.key}
              onClick={() => setStatusFilter(f.key)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                statusFilter === f.key
                  ? "bg-brand/15 text-brand border border-brand/30"
                  : "text-zinc-400 hover:text-zinc-200 border border-transparent hover:border-border"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative flex-1 min-w-0 max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-600" />
          <input
            type="text"
            placeholder="Search filename or task..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-8 py-1.5 text-sm bg-surface-50 border border-border rounded text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-brand/60"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {(searchQuery || statusFilter !== "all") && runs && (
          <span className="text-xs text-zinc-600 self-center">
            {filtered.length} of {runs.length}
          </span>
        )}
      </div>

      {/* Main: table + detail panel */}
      <div className="flex flex-col lg:flex-row gap-4 flex-1 min-h-0">
        {/* Run table */}
        <div
          className={`rounded-lg border border-border bg-surface-100 overflow-hidden flex flex-col ${
            selected ? "lg:w-80 flex-shrink-0" : "flex-1"
          }`}
        >
          <div className="overflow-x-auto flex-1 overflow-y-auto">
            <table className="w-full">
              <thead className="sticky top-0 z-10">
                <tr className="bg-surface-100 border-b border-border">
                  <SortHeader
                    label="File"
                    sortKey="filename"
                    currentKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                  />
                  <SortHeader
                    label="Task"
                    sortKey="task"
                    currentKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                    className="w-36"
                  />
                  <SortHeader
                    label="Status"
                    sortKey="status"
                    currentKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                    className="w-28"
                  />
                  <SortHeader
                    label="Date"
                    sortKey="created_at"
                    currentKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                    className="w-36"
                  />
                </tr>
              </thead>
              <tbody>
                {loading && !runs
                  ? Array.from({ length: 6 }).map((_, i) => (
                      <tr key={i} className="border-b border-border-subtle">
                        {Array.from({ length: 4 }).map((_, j) => (
                          <td key={j} className="px-3 py-2.5">
                            <div className="h-4 w-3/4 rounded bg-surface-50 animate-pulse" />
                          </td>
                        ))}
                      </tr>
                    ))
                  : filtered.length === 0
                  ? (
                    <tr>
                      <td colSpan={4} className="px-4 py-12 text-center">
                        <FileCheck className="w-7 h-7 text-zinc-700 mx-auto mb-2" />
                        <p className="text-sm text-zinc-500">
                          {!selectedRoute
                            ? "Select a route to load results."
                            : searchQuery || statusFilter !== "all"
                            ? "No runs match your filters."
                            : "No processed runs found."}
                        </p>
                        {(searchQuery || statusFilter !== "all") && (
                          <button
                            onClick={() => {
                              setSearchQuery("");
                              setStatusFilter("all");
                            }}
                            className="mt-2 text-xs text-brand hover:underline"
                          >
                            Clear filters
                          </button>
                        )}
                      </td>
                    </tr>
                  )
                  : filtered.map((run) => (
                      <tr
                        key={run.run_id}
                        onClick={() =>
                          setSelected((s) =>
                            s?.run_id === run.run_id ? null : run
                          )
                        }
                        className={`border-b border-border-subtle cursor-pointer transition-colors duration-100 ${
                          selected?.run_id === run.run_id
                            ? "bg-brand/5"
                            : "hover:bg-surface-50/30"
                        }`}
                      >
                        <td className="px-3 py-2.5">
                          <div className="flex items-center gap-2 min-w-0">
                            {run.success ? (
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                            ) : (
                              <XCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                            )}
                            <span className="text-sm font-medium text-zinc-200 truncate font-mono">
                              {run.filename || run.run_id}
                            </span>
                          </div>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="text-xs text-zinc-400 truncate block max-w-[130px]">
                            {run.task}
                          </span>
                        </td>
                        <td className="px-3 py-2.5">
                          <StatusBadge run={run} />
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="text-xs text-zinc-500">
                            {formatDate(run.created_at)}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 w-8">
                          {(() => {
                            const decision = decisions[run.run_id];
                            if (!decision) return null;
                            return (
                            <span
                              className={`inline-block w-2 h-2 rounded-full ${
                                decision.decision === "pass"
                                  ? "bg-emerald-400"
                                  : decision.decision === "fail"
                                  ? "bg-red-400"
                                  : "bg-amber-400"
                              }`}
                              title={decision.decision}
                            />
                            );
                          })()}
                        </td>
                      </tr>
                    ))}
              </tbody>
            </table>
          </div>
          {filtered.length > 0 && (
            <div className="px-3 py-2 border-t border-border-subtle text-xs text-zinc-600 flex-shrink-0">
              {filtered.length} run{filtered.length !== 1 ? "s" : ""}
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selected && (
          <RunDetailPanel
            run={selected}
            onClose={() => setSelected(null)}
            decision={(decisions[selected.run_id]?.decision as DecisionValue) ?? null}
            decisionNotes={decisions[selected.run_id]?.notes ?? ""}
            onDecisionChange={handleDecisionChange}
          />
        )}
      </div>
    </div>
  );
}
