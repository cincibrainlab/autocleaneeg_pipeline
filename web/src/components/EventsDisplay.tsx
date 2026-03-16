/**
 * EventsDisplay — shared event visualization component.
 *
 * Accepts an EventsResponse (from either the Results events endpoint or the
 * standalone Event Analyzer page) and renders the full visualization:
 * metric cards, bar chart, per-type timing table, long gaps, ISI analysis,
 * and event transitions.
 */

import { Zap } from "lucide-react";
import type { EventsResponse } from "../lib/api";

// ── Palette ───────────────────────────────────────────────────────

const EVENT_BAR_COLORS = [
  "bg-brand",
  "bg-violet-500",
  "bg-amber-500",
  "bg-cyan-500",
  "bg-emerald-500",
  "bg-pink-500",
  "bg-orange-500",
  "bg-sky-500",
];

// ── Internal sub-components ───────────────────────────────────────

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
        <MiniProgress value={numerator} max={denominator} color={progressColor} />
      )}
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────────

export default function EventsDisplay({ data }: { data: EventsResponse }) {
  // Resting state: no or minimal events
  if (data.recording_type === "resting_state") {
    return (
      <div className="flex flex-col items-center justify-center py-12 gap-3">
        <div className="rounded-lg border border-border bg-surface-50 px-6 py-5 max-w-sm text-center">
          <Zap className="w-6 h-6 text-zinc-600 mx-auto mb-3" />
          <p className="text-sm font-semibold text-zinc-300 mb-1">
            Resting State Recording
          </p>
          {data.event_count > 0 ? (
            <p className="text-xs text-zinc-500">
              {data.event_count} event marker{data.event_count !== 1 ? "s" : ""}{" "}
              detected, but below the threshold for event-related analysis. This
              is typical for resting-state paradigms.
            </p>
          ) : (
            <p className="text-xs text-zinc-500">
              No event markers detected. This is expected for resting-state
              paradigms.
            </p>
          )}
        </div>
      </div>
    );
  }

  const maxCount =
    data.event_types.length > 0
      ? Math.max(...data.event_types.map((t) => t.count), 1)
      : 1;
  const totalCounted = data.event_types.reduce((s, t) => s + t.count, 0);

  // Derived flags for conditional sections
  const hasTimingData = data.event_types.some(
    (et) => et.first_onset != null || et.mean_isi != null
  );
  const hasMultipleTypes = data.event_types.length > 1;
  const hasLongGaps = (data.long_gaps ?? []).length > 0;
  const hasTransitions =
    (data.transitions ?? []).length > 0 && hasMultipleTypes;
  const totalTransitions = (data.transitions ?? []).reduce(
    (s, t) => s + t.count,
    0
  );

  return (
    <div className="space-y-6">
      {/* Metric row */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Total Events"
          value={data.event_count > 0 ? String(data.event_count) : "—"}
        />
        <MetricCard
          label="Unique Types"
          value={data.unique_type_count > 0 ? String(data.unique_type_count) : "—"}
        />
        {data.duration_sec != null && (
          <MetricCard
            label="Duration"
            value={
              data.duration_sec >= 60
                ? `${(data.duration_sec / 60).toFixed(1)} min`
                : `${data.duration_sec}s`
            }
            sub="event span"
          />
        )}
        {data.events_per_min != null && (
          <MetricCard
            label="Rate"
            value={`${data.events_per_min}/min`}
            sub="events per minute"
          />
        )}
      </div>

      {/* Event type distribution bar chart */}
      {data.event_types.length > 0 && totalCounted > 0 && (
        <div>
          <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-3">
            Event Type Distribution
          </p>
          <div className="space-y-2">
            {data.event_types.map((et, i) => {
              const pctWidth = maxCount > 0 ? (et.count / maxCount) * 100 : 0;
              const pctOfTotal =
                totalCounted > 0
                  ? `${((et.count / totalCounted) * 100).toFixed(1)}%`
                  : "";
              const barColor = EVENT_BAR_COLORS[i % EVENT_BAR_COLORS.length];
              return (
                <div key={et.label} className="flex items-center gap-3 text-xs">
                  <span className="w-24 text-right text-zinc-400 font-mono truncate flex-shrink-0">
                    {et.label}
                  </span>
                  <div className="flex-1 h-4 rounded-sm bg-surface-50 overflow-hidden">
                    <div
                      className={`h-full rounded-sm transition-all ${barColor} opacity-70`}
                      style={{ width: `${pctWidth}%` }}
                    />
                  </div>
                  <span className="text-zinc-400 tabular-nums w-10 text-right flex-shrink-0">
                    {et.count > 0 ? et.count : "—"}
                  </span>
                  {pctOfTotal && (
                    <span className="text-zinc-600 tabular-nums w-12 flex-shrink-0">
                      ({pctOfTotal})
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Per-type timing table */}
      {hasMultipleTypes && hasTimingData && totalCounted > 0 && (
        <div>
          <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
            Per-Type Timing
          </p>
          <div className="rounded border border-border overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-surface-100">
                  <th className="px-3 py-1.5 text-left text-[10px] uppercase text-zinc-500 font-medium">
                    Type
                  </th>
                  <th className="px-2 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium">
                    Count
                  </th>
                  <th className="px-2 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium">
                    First (s)
                  </th>
                  <th className="px-2 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium">
                    Last (s)
                  </th>
                  <th className="px-2 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium">
                    Mean ISI
                  </th>
                  <th className="px-2 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium">
                    Med ISI
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.event_types.map((et) => (
                  <tr
                    key={et.label}
                    className="border-b border-border-subtle last:border-0 hover:bg-surface-50/30"
                  >
                    <td className="px-3 py-1.5 font-mono text-zinc-200">
                      {et.label}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-zinc-400 tabular-nums">
                      {et.count > 0 ? et.count : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-zinc-400 tabular-nums">
                      {et.first_onset != null ? et.first_onset : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-zinc-400 tabular-nums">
                      {et.last_onset != null ? et.last_onset : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-zinc-400 tabular-nums">
                      {et.mean_isi != null ? `${et.mean_isi}s` : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-zinc-400 tabular-nums">
                      {et.median_isi != null ? `${et.median_isi}s` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Long gaps warning */}
      {hasLongGaps && (
        <div>
          <div className="rounded border border-amber-500/30 bg-amber-500/10 px-3 py-2">
            <p className="text-xs font-medium text-amber-300 mb-1.5">
              {(data.long_gaps ?? []).length} gap
              {(data.long_gaps ?? []).length !== 1 ? "s" : ""} &gt; 30s detected
            </p>
            <div className="space-y-0.5">
              {(data.long_gaps ?? []).map((g, i) => (
                <p key={i} className="text-xs text-amber-400/80 font-mono">
                  {g.start}s &ndash; {g.end}s ({g.duration}s gap)
                </p>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ISI timing table */}
      {data.isi_stats && (
        <div>
          <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
            Timing Analysis (ISI)
          </p>
          {/* Jitter warning when std > 20% of mean */}
          {data.isi_stats.mean > 0 &&
            data.isi_stats.std / data.isi_stats.mean > 0.2 && (
              <div className="mb-2 rounded border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-300">
                High ISI variability detected (std / mean ={" "}
                {((data.isi_stats.std / data.isi_stats.mean) * 100).toFixed(0)}
                %). Stimulus timing may be jittered or irregular.
              </div>
            )}
          <div className="rounded border border-border overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-surface-100">
                  <th className="px-3 py-1.5 text-left text-[10px] uppercase text-zinc-500 font-medium">
                    Statistic
                  </th>
                  <th className="px-3 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium">
                    Value (s)
                  </th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["Min", data.isi_stats.min],
                    ["Max", data.isi_stats.max],
                    ["Mean", data.isi_stats.mean],
                    ["Median", data.isi_stats.median],
                    ["Std Dev", data.isi_stats.std],
                    ["N Intervals", data.isi_stats.count],
                  ] as Array<[string, number]>
                ).map(([label, val]) => (
                  <tr
                    key={label}
                    className="border-b border-border-subtle last:border-0 hover:bg-surface-50/30"
                  >
                    <td className="px-3 py-1.5 text-zinc-400">{label}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-zinc-200">
                      {val}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Event transitions table */}
      {hasTransitions && (
        <div>
          <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
            Event Transitions
          </p>
          <div className="rounded border border-border overflow-hidden">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-border bg-surface-100">
                  <th className="px-3 py-1.5 text-left text-[10px] uppercase text-zinc-500 font-medium font-sans">
                    From
                  </th>
                  <th className="px-2 py-1.5 text-left text-[10px] uppercase text-zinc-500 font-medium font-sans">
                    To
                  </th>
                  <th className="px-2 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium font-sans">
                    Count
                  </th>
                  <th className="px-2 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium font-sans">
                    %
                  </th>
                </tr>
              </thead>
              <tbody>
                {(data.transitions ?? []).map((txn, i) => (
                  <tr
                    key={i}
                    className="border-b border-border-subtle last:border-0 hover:bg-surface-50/30"
                  >
                    <td className="px-3 py-1.5 text-zinc-300">{txn.from}</td>
                    <td className="px-2 py-1.5 text-zinc-300">{txn.to}</td>
                    <td className="px-2 py-1.5 text-right text-zinc-400 tabular-nums">
                      {txn.count}
                    </td>
                    <td className="px-2 py-1.5 text-right text-zinc-500 tabular-nums">
                      {totalTransitions > 0
                        ? `${((txn.count / totalTransitions) * 100).toFixed(0)}%`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Event code table — shown when types available but no timeline counts */}
      {data.event_types.length > 0 && totalCounted === 0 && (
        <div>
          <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
            Event Codes
          </p>
          <div className="rounded border border-border overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-surface-100">
                  <th className="px-3 py-1.5 text-left text-[10px] uppercase text-zinc-500 font-medium">
                    Label
                  </th>
                  <th className="px-3 py-1.5 text-right text-[10px] uppercase text-zinc-500 font-medium">
                    Code
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.event_types.map((et) => (
                  <tr
                    key={et.label}
                    className="border-b border-border-subtle last:border-0 hover:bg-surface-50/30"
                  >
                    <td className="px-3 py-1.5 font-mono text-zinc-200">
                      {et.label}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-zinc-400">
                      {et.code}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
