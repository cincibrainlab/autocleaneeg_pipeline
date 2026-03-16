import { useState, useMemo, useEffect } from "react";
import {
  MapPin,
  Search,
  ChevronDown,
  ChevronUp,
  ArrowUpDown,
  X,
  Loader2,
} from "lucide-react";
import { api } from "../lib/api";
import type { MontageInfo, MontageDetail, MontageListResponse } from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";
import { usePolling } from "../hooks/usePolling";

// ── Helpers ───────────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  standard: "text-violet-400",
  geodesic: "text-amber-400",
  biosemi: "text-blue-400",
  specialty: "text-rose-400",
  custom: "text-cyan-400",
};

const CATEGORY_BADGE_STYLES: Record<string, string> = {
  standard: "bg-violet-500/15 text-violet-400",
  geodesic: "bg-amber-500/15 text-amber-400",
  biosemi: "bg-blue-500/15 text-blue-400",
  specialty: "bg-rose-500/15 text-rose-400",
  custom: "bg-cyan-500/15 text-cyan-400",
};

// Human-readable category labels for the filter tabs
const CATEGORY_LABELS: Record<string, string> = {
  standard: "Standard",
  geodesic: "Geodesic",
  biosemi: "BioSemi",
  specialty: "Specialty",
  custom: "Custom",
};

type SortKey = "name" | "n_channels" | "category" | "compatible_tasks";
type SortDir = "asc" | "desc";

function sortMontages(
  montages: MontageInfo[],
  key: SortKey,
  dir: SortDir
): MontageInfo[] {
  const mult = dir === "asc" ? 1 : -1;
  return [...montages].sort((a, b) => {
    let va = 0,
      vb = 0;
    let sa = "",
      sb = "";
    switch (key) {
      case "name":
        sa = a.name;
        sb = b.name;
        return sa.localeCompare(sb, undefined, { numeric: true }) * mult;
      case "category":
        sa = a.category;
        sb = b.category;
        return sa.localeCompare(sb) * mult;
      case "n_channels":
        va = a.n_channels;
        vb = b.n_channels;
        return (va - vb) * mult;
      case "compatible_tasks":
        va = a.compatible_tasks.length;
        vb = b.compatible_tasks.length;
        return (va - vb) * mult;
    }
  });
}

// ── Sort column header ────────────────────────────────────────────

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

// ── Detail panel ─────────────────────────────────────────────────

function MontageDetailPanel({
  montage,
  onClose,
}: {
  montage: MontageInfo;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<MontageDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setLoading(true);
    setError(null);

    api
      .getMontageDetail(montage.name)
      .then((d) => {
        if (!cancelled) { setDetail(d); setLoading(false); }
      })
      .catch((e: unknown) => {
        if (!cancelled) { setError(e instanceof Error ? e.message : String(e)); setLoading(false); }
      });
    return () => { cancelled = true; };
  }, [montage.name]);

  const badgeStyle =
    CATEGORY_BADGE_STYLES[montage.category] ?? "bg-zinc-500/15 text-zinc-400";

  return (
    <div className="lg:w-96 flex-shrink-0 rounded-lg border border-border bg-surface-100 self-start lg:sticky lg:top-5 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2 min-w-0">
            <MapPin className="w-4 h-4 text-brand flex-shrink-0" />
            <h3 className="text-sm font-semibold text-zinc-100 truncate">
              {montage.name}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-300"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex items-center gap-1.5 mb-2">
          <span
            className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${badgeStyle}`}
          >
            {CATEGORY_LABELS[montage.category] ?? montage.category}
          </span>
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium text-zinc-400 bg-surface-50">
            {montage.n_channels} ch
          </span>
        </div>
        {montage.description && (
          <p className="text-xs text-zinc-500 leading-relaxed">
            {montage.description}
          </p>
        )}
      </div>

      {loading && (
        <div className="flex items-center justify-center py-10 gap-2 text-zinc-500 text-xs">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading montage data&hellip;
        </div>
      )}

      {error && !loading && (
        <div className="px-5 py-4">
          <ErrorBanner message={error} />
        </div>
      )}

      {detail && !loading && (
        <>
          {/* Topomap */}
          {detail.topomap_png && (
            <div className="px-5 py-4 border-b border-border flex justify-center">
              <img
                src={`data:image/png;base64,${detail.topomap_png}`}
                alt={`${montage.name} electrode topomap`}
                className="rounded max-w-[200px] w-full"
                style={{ imageRendering: "auto" }}
              />
            </div>
          )}

          {/* Compatible tasks */}
          {detail.compatible_tasks.length > 0 && (
            <div className="px-5 py-3 border-b border-border">
              <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
                Compatible Tasks
              </p>
              <ul className="space-y-1">
                {detail.compatible_tasks.map((t) => (
                  <li key={t} className="flex items-center gap-1.5">
                    <span className="w-1 h-1 rounded-full bg-brand/60 flex-shrink-0" />
                    <span className="text-xs text-zinc-300">{t}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {detail.compatible_tasks.length === 0 && (
            <div className="px-5 py-3 border-b border-border">
              <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-1">
                Compatible Tasks
              </p>
              <p className="text-xs text-zinc-600 italic">
                No tasks currently reference this montage.
              </p>
            </div>
          )}

          {/* Landmarks */}
          {detail.landmarks && Object.keys(detail.landmarks).length > 0 && (
            <div className="px-5 py-3 border-b border-border">
              <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
                Landmarks
              </p>
              <div className="space-y-1">
                {Object.entries(detail.landmarks).map(([lm, pos]) => (
                  <div key={lm} className="flex items-center justify-between">
                    <span className="text-xs font-medium text-zinc-400 capitalize">
                      {lm}
                    </span>
                    <span className="text-[11px] font-mono text-zinc-600">
                      {pos.map((v) => v.toFixed(3)).join(", ")}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Channel list */}
          <div className="px-5 py-3">
            <p className="text-[10px] uppercase font-medium text-zinc-500 tracking-wider mb-2">
              Channels ({detail.channels.length})
            </p>
            <div className="max-h-[260px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-zinc-600 text-[10px] uppercase border-b border-border-subtle">
                    <th className="pb-1 text-left font-medium">Name</th>
                    <th className="pb-1 text-right font-medium">X</th>
                    <th className="pb-1 text-right font-medium">Y</th>
                    <th className="pb-1 text-right font-medium">Z</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.channels.map((ch) => (
                    <tr
                      key={ch.name}
                      className="border-b border-border-subtle/40 hover:bg-surface-50/20"
                    >
                      <td className="py-0.5 font-medium text-zinc-300 font-mono">
                        {ch.name}
                      </td>
                      <td className="py-0.5 text-right font-mono text-zinc-500">
                        {ch.x?.toFixed(3) ?? "—"}
                      </td>
                      <td className="py-0.5 text-right font-mono text-zinc-500">
                        {ch.y?.toFixed(3) ?? "—"}
                      </td>
                      <td className="py-0.5 text-right font-mono text-zinc-500">
                        {ch.z?.toFixed(3) ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────

export default function MontagesPage() {
  const {
    data: montageData,
    error,
    loading,
  } = usePolling<MontageListResponse>(api.getMontages, 120000);

  const montages = montageData?.montages ?? null;

  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [selected, setSelected] = useState<MontageInfo | null>(null);

  // Derive distinct categories from data
  const categories = useMemo(
    () =>
      montages
        ? [...new Set(montages.map((m) => m.category))].sort()
        : [],
    [montages]
  );

  const filtered = useMemo(() => {
    if (!montages) return [];
    let list = montages;
    if (activeCategory !== "all") {
      list = list.filter((m) => m.category === activeCategory);
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          (m.description ?? "").toLowerCase().includes(q) ||
          m.category.toLowerCase().includes(q)
      );
    }
    return sortMontages(list, sortKey, sortDir);
  }, [montages, activeCategory, searchQuery, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  // Clear selection when it disappears from filtered results
  useEffect(() => {
    if (selected && !filtered.find((m) => m.name === selected.name)) {
      setSelected(null);
    }
  }, [filtered, selected]);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">Montages</h2>
          <p className="text-xs text-zinc-500 mt-0.5">
            Browse electrode montage configurations
          </p>
          {montageData && (
            <p className="text-xs text-zinc-600 mt-1">
              {montageData.total} available
            </p>
          )}
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-2 flex-wrap">
        {/* Category tabs */}
        <div className="flex items-center gap-1 flex-wrap">
          <button
            onClick={() => setActiveCategory("all")}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              activeCategory === "all"
                ? "bg-brand/15 text-brand border border-brand/30"
                : "text-zinc-400 hover:text-zinc-200 border border-transparent hover:border-border"
            }`}
          >
            All
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                activeCategory === cat
                  ? "bg-brand/15 text-brand border border-brand/30"
                  : "text-zinc-400 hover:text-zinc-200 border border-transparent hover:border-border"
              }`}
            >
              {CATEGORY_LABELS[cat] ?? cat}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative flex-1 min-w-0 max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-600" />
          <input
            type="text"
            placeholder="Search name or category..."
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

        {/* Result count when filters are active */}
        {(searchQuery || activeCategory !== "all") && montages && (
          <span className="text-xs text-zinc-600 self-center">
            {filtered.length} of {montages.length}
          </span>
        )}
      </div>

      {/* Main: table + detail panel */}
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Montage table */}
        <div className="flex-1 min-w-0 rounded-lg border border-border bg-surface-100 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-surface-100 border-b border-border">
                  <SortHeader
                    label="Name"
                    sortKey="name"
                    currentKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                  />
                  <SortHeader
                    label="Ch"
                    sortKey="n_channels"
                    currentKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                    className="w-16"
                  />
                  <SortHeader
                    label="Category"
                    sortKey="category"
                    currentKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                    className="w-28"
                  />
                  <SortHeader
                    label="Tasks"
                    sortKey="compatible_tasks"
                    currentKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                    className="w-16"
                  />
                </tr>
              </thead>
              <tbody>
                {loading && !montages
                  ? Array.from({ length: 8 }).map((_, i) => (
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
                        <MapPin className="w-7 h-7 text-zinc-700 mx-auto mb-2" />
                        <p className="text-sm text-zinc-500">
                          {searchQuery || activeCategory !== "all"
                            ? "No montages match your filters."
                            : "No montages found."}
                        </p>
                        {(searchQuery || activeCategory !== "all") && (
                          <button
                            onClick={() => {
                              setSearchQuery("");
                              setActiveCategory("all");
                            }}
                            className="mt-2 text-xs text-brand hover:underline"
                          >
                            Clear filters
                          </button>
                        )}
                      </td>
                    </tr>
                  )
                  : filtered.map((montage) => (
                      <tr
                        key={montage.name}
                        onClick={() =>
                          setSelected((s) =>
                            s?.name === montage.name ? null : montage
                          )
                        }
                        className={`border-b border-border-subtle cursor-pointer transition-colors duration-100 ${
                          selected?.name === montage.name
                            ? "bg-brand/5"
                            : "hover:bg-surface-50/30"
                        }`}
                      >
                        <td className="px-3 py-2.5">
                          <div className="flex items-center gap-2 min-w-0">
                            <MapPin className="w-3.5 h-3.5 text-brand/60 flex-shrink-0" />
                            <span className="text-sm font-medium text-zinc-200 truncate font-mono">
                              {montage.name}
                            </span>
                          </div>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="text-xs font-mono text-zinc-400">
                            {montage.n_channels}
                          </span>
                        </td>
                        <td className="px-3 py-2.5">
                          <span
                            className={`text-xs font-medium ${
                              CATEGORY_COLORS[montage.category] ??
                              "text-zinc-400"
                            }`}
                          >
                            {CATEGORY_LABELS[montage.category] ??
                              montage.category}
                          </span>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="text-xs font-mono text-zinc-400">
                            {montage.compatible_tasks.length > 0
                              ? montage.compatible_tasks.length
                              : "—"}
                          </span>
                        </td>
                      </tr>
                    ))}
              </tbody>
            </table>
          </div>
          {filtered.length > 0 && (
            <div className="px-3 py-2 border-t border-border-subtle text-xs text-zinc-600">
              {filtered.length} montage{filtered.length !== 1 ? "s" : ""}
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selected && (
          <MontageDetailPanel
            montage={selected}
            onClose={() => setSelected(null)}
          />
        )}
      </div>
    </div>
  );
}
