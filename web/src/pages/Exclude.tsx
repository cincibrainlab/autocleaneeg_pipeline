import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Activity, Brain, FileText, Loader2, MonitorDown, Search, SlidersHorizontal, StickyNote, Waves, X } from "lucide-react";
import { api } from "../lib/api";
import type {
  DashboardStatus,
  EpochManifest,
  EpochWindowResponse,
  ExcludeEpochTopographyResponse,
  ExcludeFileDetail,
  ExcludeFileSummary,
  ExcludeIcaSummaryResponse,
  RouteSpec,
} from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";
import { usePolling } from "../hooks/usePolling";

type TabKey = "eeg" | "psd" | "report" | "ica";
type DragMode = "reject" | "restore";
type TopographyRequest = {
  epochIndex: number;
  sampleIndex: number;
  latencyMs: number;
};

function TabButton({
  active,
  label,
  icon: Icon,
  onClick,
}: {
  active: boolean;
  label: string;
  icon: React.ElementType;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "flex items-center gap-2 px-3 py-2 text-xs font-medium border-b-2 transition-colors",
        active ? "border-brand text-brand" : "border-transparent text-zinc-500 hover:text-zinc-300",
      ].join(" ")}
    >
      <Icon className="w-3.5 h-3.5" />
      {label}
    </button>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs">
      <span className="text-zinc-500">{label}</span>
      <span className="text-zinc-200 font-medium text-right">{value}</span>
    </div>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function buildEpochRange(start: number, end: number) {
  const lower = Math.min(start, end);
  const upper = Math.max(start, end);
  return Array.from({ length: upper - lower + 1 }, (_, index) => lower + index);
}

function normalizeBadChannelInput(value: string) {
  const text = value.trim().toUpperCase();
  if (!text) return "";
  if (/^\d+$/.test(text)) return `E${text}`;
  return text;
}

function arraysEqual<T>(left: T[], right: T[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function isValidIcaComponent(value: number, maxComponents: number) {
  if (!Number.isInteger(value) || value < 0) return false;
  if (maxComponents <= 0) return true;
  return value < maxComponents;
}

function DiffChips({
  label,
  baseline,
  manual,
  prefix = "",
}: {
  label: string;
  baseline: Array<string | number>;
  manual: Array<string | number>;
  prefix?: string;
}) {
  const baselineSet = new Set(baseline.map(String));
  const added = manual.map(String).filter((value) => !baselineSet.has(value));
  const shared = manual.map(String).filter((value) => baselineSet.has(value));
  return (
    <div className="space-y-2">
      <label className="block text-[11px] uppercase tracking-wider text-zinc-600">{label}</label>
      <div className="text-[11px] text-zinc-500">
        Baseline: {baseline.length ? baseline.map((value) => `${prefix}${value}`).join(", ") : "None"}
      </div>
      <div className="flex flex-wrap gap-2">
        {manual.length ? manual.map((value) => {
          const text = `${prefix}${value}`;
          const isAdded = added.includes(String(value));
          return (
            <span
              key={text}
              className={[
                "rounded-full border px-2 py-1 text-xs",
                isAdded ? "border-brand/40 bg-brand/10 text-brand" : "border-border text-zinc-300",
              ].join(" ")}
              title={isAdded ? "Manual addition" : "Also present in baseline pipeline output"}
            >
              {text}
              {isAdded ? " +" : ""}
            </span>
          );
        }) : <span className="text-xs text-zinc-600">No manual overrides</span>}
      </div>
      {shared.length > 0 ? (
        <div className="text-[11px] text-zinc-600">Shared with baseline: {shared.map((value) => `${prefix}${value}`).join(", ")}</div>
      ) : null}
    </div>
  );
}

function EegBrowser({
  epochWindow,
  manifest,
  badEpochs,
  focusedEpoch,
  scaleUv,
  channelHeight,
  visibleEpochCount,
  epochStart,
  onFocusEpoch,
  onToggleEpoch,
  onApplyEpochRange,
  onEpochStartChange,
  onOpenTopography,
}: {
  epochWindow: EpochWindowResponse | null;
  manifest: EpochManifest | null;
  badEpochs: number[];
  focusedEpoch: number | null;
  scaleUv: number;
  channelHeight: number;
  visibleEpochCount: number;
  epochStart: number;
  onFocusEpoch: (epochIndex: number) => void;
  onToggleEpoch: (epochIndex: number) => void;
  onApplyEpochRange: (startEpoch: number, endEpoch: number, mode: DragMode) => void;
  onEpochStartChange: (epochStart: number) => void;
  onOpenTopography: (request: TopographyRequest) => void;
}) {
  const headerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bodyCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const footerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const headerHeight = 34;
  const footerHeight = 26;
  const labelWidth = 88;
  const [frameWidth, setFrameWidth] = useState(1200);
  const [dragAnchorEpoch, setDragAnchorEpoch] = useState<number | null>(null);
  const [dragHoverEpoch, setDragHoverEpoch] = useState<number | null>(null);
  const [dragMode, setDragMode] = useState<DragMode | null>(null);
  const [isDraggingEpochRange, setIsDraggingEpochRange] = useState(false);
  const dragAnchorRef = useRef<number | null>(null);
  const dragHoverRef = useRef<number | null>(null);
  const dragModeRef = useRef<DragMode | null>(null);
  const channels = epochWindow?.channel_names ?? [];
  const startIndex = clamp(epochStart, 0, Math.max(0, (epochWindow?.epochs.length ?? 0) - visibleEpochCount));
  const visibleEpochs = epochWindow?.epochs.slice(startIndex, startIndex + visibleEpochCount) ?? [];
  const canvasWidth = Math.max(360, frameWidth - 2);
  const traceWidth = Math.max(220, canvasWidth - labelWidth);
  const epochWidth = traceWidth / Math.max(1, visibleEpochCount);
  const bodyCanvasHeight = channels.length * channelHeight;
  const maxStart = Math.max(0, (manifest?.n_epochs ?? 0) - visibleEpochCount);
  const previewEpochRange = useMemo(() => {
    if (!isDraggingEpochRange || dragAnchorEpoch == null || dragHoverEpoch == null) return [];
    return buildEpochRange(dragAnchorEpoch, dragHoverEpoch);
  }, [dragAnchorEpoch, dragHoverEpoch, isDraggingEpochRange]);
  const previewEpochSet = useMemo(() => new Set(previewEpochRange), [previewEpochRange]);

  useEffect(() => {
    const updateWidth = () => {
      const nextWidth = frameRef.current?.clientWidth ?? 0;
      if (nextWidth > 0) setFrameWidth(nextWidth);
    };
    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  useEffect(() => {
    const headerCanvas = headerCanvasRef.current;
    const bodyCanvas = bodyCanvasRef.current;
    const footerCanvas = footerCanvasRef.current;
    if (!headerCanvas || !bodyCanvas || !footerCanvas || !epochWindow) return;
    const headerCtx = headerCanvas.getContext("2d");
    const bodyCtx = bodyCanvas.getContext("2d");
    const footerCtx = footerCanvas.getContext("2d");
    if (!headerCtx || !bodyCtx || !footerCtx) return;

    const channels = epochWindow.channel_names;
    const startIndex = clamp(epochStart, 0, Math.max(0, epochWindow.epochs.length - visibleEpochCount));
    const epochs = epochWindow.epochs.slice(startIndex, startIndex + visibleEpochCount);
    const width = Math.max(360, frameWidth - 2);
    const traceWidth = Math.max(220, width - labelWidth);
    const epochWidth = traceWidth / Math.max(1, visibleEpochCount);
    const traceHeight = channels.length * channelHeight;
    headerCanvas.width = width;
    headerCanvas.height = headerHeight;
    bodyCanvas.width = width;
    bodyCanvas.height = traceHeight;
    footerCanvas.width = width;
    footerCanvas.height = footerHeight;

    headerCtx.clearRect(0, 0, width, headerHeight);
    bodyCtx.clearRect(0, 0, width, traceHeight);
    footerCtx.clearRect(0, 0, width, footerHeight);

    headerCtx.fillStyle = "#0f172a";
    headerCtx.fillRect(0, 0, labelWidth, headerHeight);
    headerCtx.fillStyle = "#111827";
    headerCtx.fillRect(labelWidth, 0, width - labelWidth, headerHeight);

    bodyCtx.fillStyle = "rgba(255,255,255,0.02)";
    bodyCtx.fillRect(0, 0, width, traceHeight);
    bodyCtx.fillStyle = "#0f172a";
    bodyCtx.fillRect(0, 0, labelWidth, traceHeight);

    footerCtx.fillStyle = "#0f172a";
    footerCtx.fillRect(0, 0, labelWidth, footerHeight);
    footerCtx.fillStyle = "#0b1220";
    footerCtx.fillRect(labelWidth, 0, width - labelWidth, footerHeight);

    headerCtx.font = "12px monospace";
    bodyCtx.font = "12px monospace";
    footerCtx.font = "12px monospace";
    headerCtx.textBaseline = "middle";
    bodyCtx.textBaseline = "middle";
    footerCtx.textBaseline = "middle";

    epochs.forEach((epoch, epochIdx) => {
      const x0 = labelWidth + epochIdx * epochWidth;
      const isBad = badEpochs.includes(epoch.epoch_index);
      const isFocused = focusedEpoch === epoch.epoch_index;
      const isPreview = previewEpochSet.has(epoch.epoch_index);
      const previewFill = dragMode === "restore" ? "rgba(45,212,191,0.18)" : "rgba(251,191,36,0.2)";
      const previewText = dragMode === "restore" ? "#5eead4" : "#fcd34d";
      headerCtx.fillStyle = isPreview
        ? previewFill
        : isBad
        ? "rgba(248,113,113,0.18)"
        : isFocused
        ? "rgba(96,165,250,0.18)"
        : "rgba(255,255,255,0.02)";
      headerCtx.fillRect(x0, 0, epochWidth, headerHeight);
      headerCtx.fillStyle = isPreview
        ? previewText
        : isBad
        ? "#fca5a5"
        : isFocused
        ? "#93c5fd"
        : "#cbd5e1";
      headerCtx.fillText(`Epoch ${epoch.epoch_index + 1}`, x0 + 8, headerHeight / 2);

      if ("setLineDash" in bodyCtx) bodyCtx.setLineDash([4, 4]);
      bodyCtx.strokeStyle = "rgba(226,232,240,0.5)";
      bodyCtx.lineWidth = 1.25;
      bodyCtx.beginPath();
      bodyCtx.moveTo(x0 + 0.5, 0);
      bodyCtx.lineTo(x0 + 0.5, traceHeight);
      bodyCtx.stroke();
      if ("setLineDash" in bodyCtx) bodyCtx.setLineDash([]);
      bodyCtx.lineWidth = 1;

      footerCtx.fillStyle = "#94a3b8";
      footerCtx.fillText(`${epoch.start_time_seconds.toFixed(2)}s`, x0 + 8, footerHeight / 2);
    });

    const endBoundary = labelWidth + epochs.length * epochWidth;
    if ("setLineDash" in bodyCtx) bodyCtx.setLineDash([4, 4]);
    bodyCtx.strokeStyle = "rgba(226,232,240,0.5)";
    bodyCtx.lineWidth = 1.25;
    bodyCtx.beginPath();
    bodyCtx.moveTo(endBoundary + 0.5, 0);
    bodyCtx.lineTo(endBoundary + 0.5, traceHeight);
    bodyCtx.stroke();
    if ("setLineDash" in bodyCtx) bodyCtx.setLineDash([]);
    bodyCtx.lineWidth = 1;

    channels.forEach((channel, channelIndex) => {
      const y0 = channelIndex * channelHeight;
      const midY = y0 + channelHeight / 2;

      bodyCtx.fillStyle = focusedEpoch != null && epochWindow.epochs.some((epoch) => epoch.epoch_index === focusedEpoch)
        ? "rgba(255,255,255,0.01)"
        : "rgba(255,255,255,0)";
      bodyCtx.fillRect(0, y0, width, channelHeight);
      bodyCtx.strokeStyle = "rgba(255,255,255,0.06)";
      bodyCtx.beginPath();
      bodyCtx.moveTo(0, y0 + 0.5);
      bodyCtx.lineTo(width, y0 + 0.5);
      bodyCtx.stroke();

      bodyCtx.fillStyle = "#cbd5e1";
      bodyCtx.fillText(channel, 8, midY);

      epochs.forEach((epoch, epochIdx) => {
        const x0 = labelWidth + epochIdx * epochWidth;
        const values = epoch.traces_uv[channel] ?? [];
        if (!values.length) return;
        const xStep = epochWidth / Math.max(1, values.length - 1);
        const isBad = badEpochs.includes(epoch.epoch_index);
        const isPreview = previewEpochSet.has(epoch.epoch_index);
        bodyCtx.beginPath();
        values.forEach((value, sampleIdx) => {
          const x = x0 + sampleIdx * xStep;
          const normalized = value / Math.max(scaleUv, 1);
          const y = midY - normalized * (channelHeight * 0.38);
          if (sampleIdx === 0) bodyCtx.moveTo(x, y);
          else bodyCtx.lineTo(x, y);
        });
        bodyCtx.strokeStyle = isPreview
          ? dragMode === "restore"
            ? "#2dd4bf"
            : "#fbbf24"
          : isBad
          ? "#f87171"
          : focusedEpoch === epoch.epoch_index
          ? "#e2e8f0"
          : "#60a5fa";
        bodyCtx.lineWidth = isPreview ? 1.5 : focusedEpoch === epoch.epoch_index ? 1.35 : 1;
        bodyCtx.stroke();
      });
    });
  }, [badEpochs, channelHeight, dragMode, epochStart, epochWindow, focusedEpoch, frameWidth, manifest, previewEpochSet, scaleUv, visibleEpochCount]);

  const getEpochIndexFromPosition = (clientX: number, target: HTMLCanvasElement) => {
    if (!epochWindow) return null;
    const rect = target.getBoundingClientRect();
    const scaleX = rect.width > 0 ? canvasWidth / rect.width : 1;
    const localX = (clientX - rect.left) * scaleX;
    if (localX < labelWidth) return null;
    return Math.max(0, Math.min(visibleEpochs.length - 1, Math.floor((localX - labelWidth) / epochWidth)));
  };

  const getTopographyRequestFromPointer = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const epochIdx = getEpochIndexFromPosition(event.clientX, event.currentTarget);
    if (epochIdx == null || !manifest) return null;
    const rect = event.currentTarget.getBoundingClientRect();
    const scaleX = rect.width > 0 ? canvasWidth / rect.width : 1;
    const localX = (event.clientX - rect.left) * scaleX;
    const epoch = visibleEpochs[epochIdx];
    if (!epoch) return null;
    const relativeX = clamp(localX - labelWidth - epochIdx * epochWidth, 0, Math.max(epochWidth - 1, 1));
    const fraction = epochWidth > 0 ? relativeX / epochWidth : 0;
    const maxSampleIndex = Math.max(0, manifest.epoch_length_samples - 1);
    const sampleIndex = clamp(Math.round(fraction * maxSampleIndex), 0, maxSampleIndex);
    const latencyMs = sampleIndex / Math.max(manifest.sampling_rate, 1) * 1000;
    return {
      epochIndex: epoch.epoch_index,
      sampleIndex,
      latencyMs,
    };
  };

  const resetDragSelection = () => {
    dragAnchorRef.current = null;
    dragHoverRef.current = null;
    dragModeRef.current = null;
    setDragAnchorEpoch(null);
    setDragHoverEpoch(null);
    setDragMode(null);
    setIsDraggingEpochRange(false);
  };

  useEffect(() => resetDragSelection(), [epochStart, epochWindow]);

  useEffect(() => {
    const handleWindowBlur = () => resetDragSelection();
    window.addEventListener("blur", handleWindowBlur);
    return () => window.removeEventListener("blur", handleWindowBlur);
  }, []);

  if (!epochWindow || !manifest) {
    return <div className="py-16 text-center text-sm text-zinc-500">Loading EEG…</div>;
  }

  return (
    <div className="space-y-3">
      <div ref={frameRef} className="rounded-lg border border-border bg-surface-50/60 overflow-hidden">
        <div className="sticky top-0 z-10 border-b border-border bg-surface-100/95 backdrop-blur-sm">
          <canvas ref={headerCanvasRef} width={canvasWidth} height={headerHeight} className="block w-full" />
        </div>
        <div className="overflow-y-auto overflow-x-hidden max-h-[28rem]">
          <canvas
            ref={bodyCanvasRef}
            width={canvasWidth}
            height={bodyCanvasHeight}
            className="block w-full"
            onPointerDown={(event) => {
              if (event.button !== 0) return;
              const canvas = event.currentTarget;
              const epochIdx = getEpochIndexFromPosition(event.clientX, canvas);
              if (epochIdx == null) return;
              const epoch = visibleEpochs[epochIdx];
              if (!epoch) return;
              canvas.setPointerCapture(event.pointerId);
              onFocusEpoch(epoch.epoch_index);
              dragAnchorRef.current = epoch.epoch_index;
              dragHoverRef.current = epoch.epoch_index;
              dragModeRef.current = badEpochs.includes(epoch.epoch_index) ? "restore" : "reject";
              setDragAnchorEpoch(epoch.epoch_index);
              setDragHoverEpoch(epoch.epoch_index);
              setDragMode(dragModeRef.current);
              setIsDraggingEpochRange(true);
            }}
            onPointerMove={(event) => {
              if (!isDraggingEpochRange) return;
              const epochIdx = getEpochIndexFromPosition(event.clientX, event.currentTarget);
              if (epochIdx == null) return;
              const epoch = visibleEpochs[epochIdx];
              if (!epoch) return;
              dragHoverRef.current = epoch.epoch_index;
              setDragHoverEpoch((current) => (current === epoch.epoch_index ? current : epoch.epoch_index));
              onFocusEpoch(epoch.epoch_index);
            }}
            onPointerUp={(event) => {
              const anchorEpoch = dragAnchorRef.current;
              if (!isDraggingEpochRange || anchorEpoch == null) return;
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
              const epochIdx = getEpochIndexFromPosition(event.clientX, event.currentTarget);
              const epoch = epochIdx == null ? null : visibleEpochs[epochIdx];
              const endEpoch = epoch?.epoch_index ?? dragHoverRef.current ?? anchorEpoch;
              const nextMode = dragModeRef.current;
              if (endEpoch !== anchorEpoch && nextMode) {
                onApplyEpochRange(anchorEpoch, endEpoch, nextMode);
              } else {
                onFocusEpoch(endEpoch);
              }
              resetDragSelection();
            }}
            onPointerCancel={() => {
              resetDragSelection();
            }}
            onPointerLeave={() => {
              if (!isDraggingEpochRange) return;
              setDragHoverEpoch((current) => current ?? dragAnchorEpoch);
            }}
            onDoubleClick={(event) => {
              const epochIdx = getEpochIndexFromPosition(event.clientX, event.currentTarget);
              if (epochIdx == null) return;
              const epoch = visibleEpochs[epochIdx];
              if (epoch) onToggleEpoch(epoch.epoch_index);
            }}
            onContextMenu={(event) => {
              event.preventDefault();
              const request = getTopographyRequestFromPointer(event);
              if (!request) return;
              onFocusEpoch(request.epochIndex);
              onOpenTopography(request);
            }}
          />
        </div>
        <div className="sticky bottom-0 z-10 border-t border-border bg-surface-100/95 backdrop-blur-sm">
          <canvas ref={footerCanvasRef} width={canvasWidth} height={footerHeight} className="block w-full" />
        </div>
      </div>
      <div className="rounded border border-border bg-surface-100/60 px-3 py-2">
        <div className="flex items-center justify-between gap-3 text-[11px] text-zinc-400">
          <span>File position</span>
          <span>
            Epoch {epochStart + 1}
            {maxStart > 0 ? ` of ${maxStart + 1}` : ""}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, maxStart)}
          step={1}
          value={epochStart}
          onChange={(event) => onEpochStartChange(Number(event.target.value))}
          className="mt-2 w-full"
          aria-label="EEG file position"
        />
      </div>
      <div className="flex items-center justify-between text-[11px] text-zinc-500">
        <span>
          Use the bottom scrollbar for file position and scroll vertically for channels. Click to focus, drag across epochs to reject or
          restore a range, double-click or press Space for one-off changes, and right-click for a scalp map at that latency.
        </span>
        <span>{manifest.n_channels} channels · {visibleEpochs.length} visible epochs</span>
      </div>
    </div>
  );
}

function TopographyModal({
  open,
  loading,
  error,
  data,
  request,
  onClose,
}: {
  open: boolean;
  loading: boolean;
  error: string | null;
  data: ExcludeEpochTopographyResponse | null;
  request: TopographyRequest | null;
  onClose: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div
        className="w-full max-w-3xl rounded-xl border border-border bg-surface-200 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div>
            <h3 className="text-base font-semibold text-zinc-100">Epoch scalp topography</h3>
            <p className="mt-1 text-sm text-zinc-400">
              {data
                ? `Epoch ${data.epoch_index + 1} at ${data.latency_ms.toFixed(1)} ms`
                : request
                ? `Epoch ${request.epochIndex + 1} at ${request.latencyMs.toFixed(1)} ms`
                : "Loading selected latency"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md border border-border p-2 text-zinc-400 hover:bg-surface-100 hover:text-zinc-200"
            aria-label="Close topography popup"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-4 px-5 py-5">
          <div className="rounded-xl border border-border bg-[#0b1220] p-4 min-h-[26rem]">
            {loading ? (
              <div className="flex h-[24rem] items-center justify-center gap-3 text-sm text-zinc-400">
                <Loader2 className="h-5 w-5 animate-spin" />
                Rendering topography...
              </div>
            ) : error ? (
              <div className="flex h-[24rem] items-center justify-center text-sm text-red-300">{error}</div>
            ) : data ? (
              <img
                src={`data:image/png;base64,${data.image_png_base64}`}
                alt={`Epoch ${data.epoch_index + 1} scalp topography at ${data.latency_ms.toFixed(1)} milliseconds`}
                className="mx-auto max-h-[32rem] w-auto max-w-full rounded-md"
              />
            ) : null}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-zinc-500">
            <p>Right-click any trace location to inspect the scalp map at that exact latency.</p>
            {data ? <p>{data.channels_used.length} EEG channels used</p> : null}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ExcludePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedRoute = searchParams.get("route") || "";
  const taskFilter = searchParams.get("task") || "";
  const montageFilter = searchParams.get("montage") || "";
  const [files, setFiles] = useState<ExcludeFileSummary[]>([]);
  const [exportsRoot, setExportsRoot] = useState("");
  const [workspaceDir, setWorkspaceDir] = useState("");
  const [listError, setListError] = useState<string | null>(null);
  const [listLoading, setListLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [detail, setDetail] = useState<ExcludeFileDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [tab, setTab] = useState<TabKey>("eeg");
  const [manifest, setManifest] = useState<EpochManifest | null>(null);
  const [epochWindow, setEpochWindow] = useState<EpochWindowResponse | null>(null);
  const [epochWindowLoading, setEpochWindowLoading] = useState(false);
  const [epochStart, setEpochStart] = useState(0);
  const [focusedEpoch, setFocusedEpoch] = useState<number | null>(null);
  const [badEpochs, setBadEpochs] = useState<number[]>([]);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [notes, setNotes] = useState("");
  const [status, setStatus] = useState("UNSET");
  const [manualBadChannels, setManualBadChannels] = useState<string[]>([]);
  const [manualRejectedIca, setManualRejectedIca] = useState<number[]>([]);
  const [channelDraft, setChannelDraft] = useState("");
  const [icaDraft, setIcaDraft] = useState("");
  const [icaSummary, setIcaSummary] = useState<ExcludeIcaSummaryResponse | null>(null);
  const [visibleEpochCount, setVisibleEpochCount] = useState(10);
  const [scaleUv, setScaleUv] = useState(50);
  const [channelHeight, setChannelHeight] = useState(8);
  const [reprocessJobId, setReprocessJobId] = useState<string | null>(null);
  const [reprocessStatus, setReprocessStatus] = useState<string | null>(null);
  const [reprocessMessage, setReprocessMessage] = useState<string | null>(null);
  const [qaExportMessage, setQaExportMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [topographyRequest, setTopographyRequest] = useState<TopographyRequest | null>(null);
  const [topographyLoading, setTopographyLoading] = useState(false);
  const [topographyError, setTopographyError] = useState<string | null>(null);
  const [topographyData, setTopographyData] = useState<ExcludeEpochTopographyResponse | null>(null);
  const [showEegHelp, setShowEegHelp] = useState(false);
  const { data: routes } = usePolling<RouteSpec[]>(api.getRoutes, 30000);
  const epochSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const notesSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastEpochSaveRef = useRef<{ fileKey: string; badEpochs: number[] } | null>(null);
  const lastNotesSaveRef = useRef<{ fileKey: string; notes: string; status: string } | null>(null);
  const loadedManualBadChannelsRef = useRef<string[]>([]);
  const loadedManualRejectedIcaRef = useRef<number[]>([]);
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
    if (key !== "route") next.delete("route");
    setSearchParams(next, { replace: true });
    setSelectedKey(null);
  };

  function applySelectedFileData(
    fileDetail: ExcludeFileDetail,
    epochManifest: EpochManifest,
    summary: ExcludeIcaSummaryResponse | null,
    options?: { resetViewport?: boolean },
  ) {
    const resetViewport = options?.resetViewport ?? true;
    setDetail(fileDetail);
    setNotes(fileDetail.notes);
    setStatus(fileDetail.status);
    setManualBadChannels(fileDetail.manual_bad_channels);
    setManualRejectedIca(fileDetail.manual_rejected_ica);
    loadedManualBadChannelsRef.current = fileDetail.manual_bad_channels;
    loadedManualRejectedIcaRef.current = fileDetail.manual_rejected_ica;
    setManifest(epochManifest);
    if (resetViewport) {
      setVisibleEpochCount(Math.min(10, Math.max(1, epochManifest.n_epochs)));
      setScaleUv(50);
      setChannelHeight(8);
      setEpochStart(0);
      setEpochWindow(null);
    } else {
      setVisibleEpochCount((current) => clamp(current, 1, Math.max(1, epochManifest.n_epochs)));
      setEpochStart((current) => Math.min(current, Math.max(0, epochManifest.n_epochs - visibleEpochCount)));
    }
    setBadEpochs(fileDetail.epoch_review.bad_epoch_indices);
    setFocusedEpoch(fileDetail.epoch_review.bad_epoch_indices[0] ?? 0);
    setIcaSummary(summary);
  }

  const setVisibleEpochCountBounded = (value: number) => {
    const maxEpochs = Math.max(1, manifest?.n_epochs ?? 1);
    const next = clamp(Math.round(value), 1, maxEpochs);
    setVisibleEpochCount(next);
    setEpochStart((current) => {
      if (!manifest) return current;
      return Math.min(current, Math.max(0, manifest.n_epochs - next));
    });
  };

  const setScaleUvBounded = (value: number) => {
    setScaleUv(clamp(Math.round(value), 1, 1000));
  };

  const setChannelHeightBounded = (value: number) => {
    setChannelHeight(clamp(Math.round(value), 4, 56));
  };

  async function loadFiles(preserveSelection = true) {
    if (!selectedRoute) {
      setFiles([]);
      setExportsRoot("");
      setDetail(null);
      setManifest(null);
      setEpochWindow(null);
      setIcaSummary(null);
      setSelectedKey(null);
      const statusData = await api.getStatus().catch(() => null as DashboardStatus | null);
      setWorkspaceDir(statusData?.workspace_dir ?? "");
      return;
    }
    const [data, statusData] = await Promise.all([
      api.getExcludeFiles(selectedRoute || undefined),
      api.getStatus().catch(() => null as DashboardStatus | null),
    ]);
    setFiles(data.files);
    setExportsRoot(data.exports_root);
    setWorkspaceDir(statusData?.workspace_dir ?? "");
    if (!preserveSelection || !selectedKey || !data.files.some((file) => file.file_key === selectedKey)) {
      setSelectedKey(data.files[0]?.file_key ?? null);
    }
  }

  async function flushEpochSave() {
    const pending = lastEpochSaveRef.current;
    if (!pending) return;
    if (epochSaveTimer.current) {
      clearTimeout(epochSaveTimer.current);
      epochSaveTimer.current = null;
    }
    setSaveState("saving");
    try {
      const result = await api.saveExcludeEpochReview(pending.fileKey, pending.badEpochs, selectedRoute || undefined);
      setActionError(typeof result.warning === "string" && result.warning ? result.warning : null);
      setSaveState("saved");
      lastEpochSaveRef.current = null;
    } catch {
      setSaveState("error");
      setActionError("Could not save epoch review changes.");
    }
  }

  async function flushNotesSave() {
    const pending = lastNotesSaveRef.current;
    if (!pending) return;
    if (notesSaveTimer.current) {
      clearTimeout(notesSaveTimer.current);
      notesSaveTimer.current = null;
    }
    setSaveState("saving");
    try {
      await api.saveExcludeNotes(pending.fileKey, pending.notes, pending.status, selectedRoute || undefined);
      setSaveState("saved");
      lastNotesSaveRef.current = null;
    } catch {
      setSaveState("error");
      setActionError("Could not save review notes.");
    }
  }

  useEffect(() => {
    if (!selectedRoute) {
      setListLoading(false);
      setListError(null);
      return;
    }
    let cancelled = false;
    setListLoading(true);
    loadFiles()
      .then(() => {
        if (cancelled) return;
        setListError(null);
      })
      .catch((err: unknown) => {
        if (!cancelled) setListError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setListLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedKey, selectedRoute]);

  useEffect(() => {
    if (!selectedRoute) return;
    const id = setInterval(() => {
      loadFiles(true).catch(() => {});
    }, 15000);
    return () => clearInterval(id);
  }, [selectedKey, selectedRoute]);

  useEffect(() => {
    if (!selectedKey || !selectedRoute) return;
    let cancelled = false;
    setDetailLoading(true);
    setDetailError(null);
    Promise.all([
      api.getExcludeFile(selectedKey, selectedRoute || undefined),
      api.getExcludeEpochManifest(selectedKey, selectedRoute || undefined),
      api.getExcludeIcaSummary(selectedKey, selectedRoute || undefined).catch(() => null),
    ])
      .then(([fileDetail, epochManifest, summary]) => {
        if (cancelled) return;
        setActionError(null);
        setTopographyRequest(null);
        setTopographyData(null);
        setTopographyError(null);
        applySelectedFileData(fileDetail, epochManifest, summary, { resetViewport: true });
        setReprocessJobId(null);
        setReprocessStatus(null);
        setReprocessMessage(null);
      })
      .catch((err: unknown) => {
        if (!cancelled) setDetailError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedKey, selectedRoute]);

  useEffect(() => {
    if (!selectedKey || !manifest) return;
    let cancelled = false;
    setEpochWindowLoading(true);
    api.getExcludeEpochWindow(selectedKey, 0, manifest.n_epochs, manifest.channel_names, selectedRoute || undefined)
      .then((window) => {
        if (cancelled) return;
        setEpochWindow(window);
        setFocusedEpoch((current) => {
          if (current != null && window.epochs.some((epoch) => epoch.epoch_index === current)) {
            return current;
          }
          return window.epochs[epochStart]?.epoch_index ?? window.epochs[0]?.epoch_index ?? null;
        });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setActionError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setEpochWindowLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [manifest, selectedKey, selectedRoute]);

  useEffect(() => {
    if (!selectedKey || !topographyRequest || tab !== "eeg") return;
    let cancelled = false;
    setTopographyLoading(true);
    setTopographyError(null);
    api.getExcludeEpochTopography(selectedKey, topographyRequest.epochIndex, topographyRequest.sampleIndex, selectedRoute || undefined)
      .then((payload) => {
        if (cancelled) return;
        setTopographyData(payload);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setTopographyError(err instanceof Error ? err.message : String(err));
        setTopographyData(null);
      })
      .finally(() => {
        if (!cancelled) setTopographyLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedKey, tab, topographyRequest, selectedRoute]);

  useEffect(() => {
    return () => {
      void flushEpochSave();
      void flushNotesSave();
    };
  }, []);

  useEffect(() => {
    const onPageHide = () => {
      void flushEpochSave();
      void flushNotesSave();
    };
    window.addEventListener("pagehide", onPageHide);
    return () => window.removeEventListener("pagehide", onPageHide);
  }, []);

  useEffect(() => {
    void flushEpochSave();
    void flushNotesSave();
  }, [selectedKey, tab]);

  useEffect(() => {
    if (!reprocessJobId) return;
    let cancelled = false;
    const id = setInterval(() => {
      api.getExcludeReprocessStatus(reprocessJobId)
        .then((statusData) => {
          if (cancelled) return;
          setReprocessStatus(String(statusData.status ?? "unknown"));
          setReprocessMessage(String(statusData.message ?? ""));
          if (statusData.running === false) {
            clearInterval(id);
            if (selectedKey) {
              Promise.all([
                api.getExcludeFiles(selectedRoute || undefined),
                api.getExcludeFile(selectedKey, selectedRoute || undefined),
                api.getExcludeEpochManifest(selectedKey, selectedRoute || undefined),
                api.getExcludeIcaSummary(selectedKey, selectedRoute || undefined).catch(() => null),
              ])
                .then(([fileList, fileDetail, epochManifest, summary]) => {
                  if (cancelled) return;
                setFiles(fileList.files);
                setExportsRoot(fileList.exports_root);
                applySelectedFileData(fileDetail, epochManifest, summary, { resetViewport: true });
                })
                .catch(() => {});
            }
          }
        })
        .catch(() => {
          if (!cancelled) {
            setReprocessStatus("failed");
            setReprocessMessage("Could not retrieve reprocess status.");
          }
          clearInterval(id);
        });
    }, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [reprocessJobId, selectedKey, selectedRoute]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (tab !== "eeg" || !manifest || focusedEpoch == null) return;
      if (event.key === "ArrowRight") {
        event.preventDefault();
        if (!epochWindow) return;
        const focusIndex = epochWindow.epochs.findIndex((epoch) => epoch.epoch_index === focusedEpoch);
        if (focusIndex >= 0 && focusIndex < epochWindow.epochs.length - 1) {
          setFocusedEpoch(epochWindow.epochs[focusIndex + 1]?.epoch_index ?? focusedEpoch);
          return;
        }
        const maxStart = Math.max(0, manifest.n_epochs - visibleEpochCount);
        const nextStart = Math.min(maxStart, epochStart + visibleEpochCount);
        if (nextStart !== epochStart) setEpochStart(nextStart);
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        if (!epochWindow) return;
        const focusIndex = epochWindow.epochs.findIndex((epoch) => epoch.epoch_index === focusedEpoch);
        if (focusIndex > 0) {
          setFocusedEpoch(epochWindow.epochs[focusIndex - 1]?.epoch_index ?? focusedEpoch);
          return;
        }
        const prevStart = Math.max(0, epochStart - visibleEpochCount);
        if (prevStart !== epochStart) setEpochStart(prevStart);
      }
      if (event.key === " ") {
        event.preventDefault();
        toggleEpoch(focusedEpoch);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [tab, manifest, focusedEpoch, epochWindow, epochStart, visibleEpochCount]);

  const filteredFiles = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return files;
    return files.filter((file) => file.name.toLowerCase().includes(q) || file.relative_path.toLowerCase().includes(q));
  }, [files, query]);

  function scheduleEpochSave(nextBadEpochs: number[]) {
    if (!selectedKey) return;
    setSaveState("saving");
    lastEpochSaveRef.current = { fileKey: selectedKey, badEpochs: nextBadEpochs };
    if (epochSaveTimer.current) clearTimeout(epochSaveTimer.current);
    epochSaveTimer.current = setTimeout(() => {
      api.saveExcludeEpochReview(selectedKey, nextBadEpochs, selectedRoute || undefined)
        .then((result) => {
          setActionError(typeof result.warning === "string" && result.warning ? result.warning : null);
          setSaveState("saved");
          lastEpochSaveRef.current = null;
        })
        .catch(() => {
          setSaveState("error");
          setActionError("Could not save epoch review changes.");
        });
    }, 400);
  }

  function toggleEpoch(epochIndex: number) {
    const next = badEpochs.includes(epochIndex)
      ? badEpochs.filter((v) => v !== epochIndex)
      : [...badEpochs, epochIndex].sort((a, b) => a - b);
    setBadEpochs(next);
    scheduleEpochSave(next);
  }

  function applyEpochRangeAction(startEpoch: number, endEpoch: number, mode: DragMode) {
    const epochRange = buildEpochRange(startEpoch, endEpoch);
    const next = mode === "reject"
      ? Array.from(new Set([...badEpochs, ...epochRange])).sort((a, b) => a - b)
      : badEpochs.filter((epochIndex) => !epochRange.includes(epochIndex));
    setFocusedEpoch(endEpoch);
    setBadEpochs(next);
    scheduleEpochSave(next);
  }

  function scheduleNotesSave(nextNotes: string, nextStatus: string) {
    if (!selectedKey) return;
    setSaveState("saving");
    lastNotesSaveRef.current = { fileKey: selectedKey, notes: nextNotes, status: nextStatus };
    if (notesSaveTimer.current) clearTimeout(notesSaveTimer.current);
    notesSaveTimer.current = setTimeout(() => {
      api.saveExcludeNotes(selectedKey, nextNotes, nextStatus, selectedRoute || undefined)
        .then(() => {
          setSaveState("saved");
          lastNotesSaveRef.current = null;
        })
        .catch(() => {
          setSaveState("error");
          setActionError("Could not save review notes.");
        });
    }, 500);
  }

  async function saveOverrides() {
    if (!selectedKey) return;
    if (invalidCombinedOverrideChange) {
      setActionError("Change either bad channels or ICA in this run, not both. Epoch edits can still be combined with channel edits.");
      return;
    }
    setActionError(null);
    setSaveState("saving");
    try {
      await api.saveExcludeOverrides(selectedKey, manualBadChannels, manualRejectedIca, selectedRoute || undefined);
      loadedManualBadChannelsRef.current = manualBadChannels;
      loadedManualRejectedIcaRef.current = manualRejectedIca;
      setSaveState("saved");
    } catch {
      setSaveState("error");
      setActionError("Could not save manual overrides.");
    }
  }

  async function startReprocess() {
    if (!selectedKey) return;
    if (invalidCombinedOverrideChange) {
      setActionError("Change either bad channels or ICA in this run, not both. Epoch edits can still be combined with channel edits.");
      return;
    }
    const confirmed = window.confirm(
      `Reprocess ${detail?.name ?? selectedKey}?\n\n` +
      `Bad channels: ${manualBadChannels.length}\n` +
      `ICA components: ${manualRejectedIca.length}\n` +
      `Saved bad epochs: ${badEpochs.length}\n\n` +
      "This will rerun the pipeline and replace task outputs with a backed-up copy of the previous results.",
    );
    if (!confirmed) return;
    setActionError(null);
    await flushEpochSave();
    await flushNotesSave();
    setSaveState("saving");
    try {
      const result = await api.startExcludeReprocess(selectedKey, manualBadChannels, manualRejectedIca, selectedRoute || undefined);
      setReprocessJobId(result.job_id);
      setReprocessStatus(result.status);
      setReprocessMessage(result.message);
      setSaveState("saved");
    } catch (err) {
      setSaveState("error");
      setReprocessStatus("failed");
      setReprocessMessage(err instanceof Error ? err.message : String(err));
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  async function exportQa() {
    if (!selectedKey) return;
    const confirmed = window.confirm(
      `Export ${detail?.name ?? selectedKey} to QA?\n\n` +
      `Saved bad epochs: ${badEpochs.length}\n\n` +
      "This will write or overwrite the QA export for the selected file.",
    );
    if (!confirmed) return;
    setActionError(null);
    await flushEpochSave();
    await flushNotesSave();
    setSaveState("saving");
    try {
      const result = await api.exportExcludeQa([selectedKey]);
      const errorCount = result.errors.length;
      setQaExportMessage(
        errorCount
          ? `QA export finished with ${errorCount} error${errorCount === 1 ? "" : "s"}.`
          : result.exported > 0
          ? "QA export complete."
          : "QA export unchanged."
      );
      if (selectedKey) {
        const refreshed = await api.getExcludeFile(selectedKey, selectedRoute || undefined);
        setDetail(refreshed);
      }
      setSaveState("saved");
    } catch (err) {
      setQaExportMessage(err instanceof Error ? err.message : String(err));
      setSaveState("error");
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  const metrics = (detail?.metrics ?? {}) as {
    data_retained?: string;
    channels_retained?: string;
    channels_original?: string;
  };
  const artifact = (detail?.artifacts ?? {}) as Record<string, string | null>;
  const visibleEpochsInView = epochWindow?.epochs.slice(epochStart, epochStart + visibleEpochCount) ?? [];
  const validChannelSet = new Set((detail?.valid_channels ?? []).map((value) => String(value).toUpperCase()));
  const channelOverridesDirty = !arraysEqual(manualBadChannels, loadedManualBadChannelsRef.current);
  const icaOverridesDirty = !arraysEqual(manualRejectedIca, loadedManualRejectedIcaRef.current);
  const invalidCombinedOverrideChange = channelOverridesDirty && icaOverridesDirty;

  function openTopography(request: TopographyRequest) {
    setTopographyRequest(request);
    setTopographyData(null);
    setTopographyError(null);
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl">
          <h2 className="text-xl font-semibold text-zinc-100">Exclude</h2>
          <p className="mt-1 text-sm text-zinc-400">
            Route-scoped review for manual epoch cleanup, notes, and reruns.
          </p>
        </div>
        <div className="flex flex-col gap-2 text-xs text-zinc-400 sm:flex-row sm:flex-wrap sm:items-center sm:justify-start lg:justify-end">
          <div className="min-w-0 rounded-full border border-border bg-surface-100/70 px-3 py-1.5">
            <span className="mr-2 uppercase tracking-wider text-zinc-500">Workspace</span>
            <span className="font-mono text-zinc-300" title={workspaceDir}>
              {workspaceDir || "Loading..."}
            </span>
          </div>
          <div className="min-w-0 rounded-full border border-border bg-surface-100/70 px-3 py-1.5">
            <span className="mr-2 uppercase tracking-wider text-zinc-500">Exports</span>
            <span className="font-mono text-zinc-300" title={exportsRoot}>
              {exportsRoot || "Loading..."}
            </span>
          </div>
        </div>
      </div>

      <div className="grid gap-3 rounded-lg border border-border bg-surface-100 p-4 xl:grid-cols-[minmax(0,1fr)_14rem_14rem_18rem]">
        <div>
          <p className="text-xs font-medium text-zinc-300">
            {selectedRoute ? `Showing Exclude files for route '${selectedRoute}'` : "Choose a route to scope the Exclude workspace"}
          </p>
          <p className="mt-1 text-xs text-zinc-500">
            Exclude should follow route ownership. Use task and montage filters to narrow the route list when many routes are present.
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

      {listError && <ErrorBanner message={listError} />}
      {actionError && <ErrorBanner message={actionError} />}

      <div className="text-xs text-zinc-400">
        {selectedRoute
          ? `${files.length} file${files.length === 1 ? "" : "s"} in the selected route`
          : "Select a route to load Exclude files"}
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[clamp(15rem,20vw,18rem)_minmax(0,1fr)] xl:items-stretch">
        <aside className="flex h-full min-h-[clamp(34rem,68vh,52rem)] w-full shrink-0 flex-col overflow-hidden rounded-lg border border-border bg-surface-100">
          <div className="p-3 border-b border-border">
            <div className="flex items-center gap-2 rounded-md border border-border bg-surface-50 px-2.5 py-2">
              <Search className="w-4 h-4 text-zinc-500" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search files"
                className="w-full bg-transparent text-sm text-zinc-100 placeholder:text-zinc-600 outline-none"
              />
            </div>
          </div>
          <div className="flex-1 overflow-auto">
            {listLoading ? (
              <div className="flex items-center justify-center gap-2 py-10 text-zinc-500 text-sm">
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading files...
              </div>
            ) : !selectedRoute ? (
              <div className="py-10 text-center text-sm text-zinc-600">
                Select a route to load Exclude files
              </div>
            ) : filteredFiles.length === 0 ? (
              <div className="py-10 text-center text-sm text-zinc-600">
                {files.length === 0 ? "No export files found in the current exports folder" : "No files match the current search"}
              </div>
            ) : (
              filteredFiles.map((file) => (
                <button
                  key={file.file_key}
                  onClick={() => setSelectedKey(file.file_key)}
                  className={[
                    "w-full border-b border-border-subtle px-3 py-2.5 text-left transition-colors hover:bg-surface-50/40",
                    selectedKey === file.file_key ? "border-l-2 border-l-brand bg-brand/10 pl-[0.625rem]" : "border-l-2 border-l-transparent",
                  ].join(" ")}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-zinc-100 truncate">{file.name}</p>
                      <p className="text-[11px] text-zinc-400 truncate">{file.relative_path}</p>
                    </div>
                    <div className="flex flex-col items-end gap-1 text-[10px]">
                      {file.bad_epochs_count > 0 && (
                        <span className="rounded bg-red-500/15 px-1.5 py-0.5 text-red-400">
                          {file.bad_epochs_count} bad
                        </span>
                      )}
                      {file.notes_present && (
                        <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-amber-400">note</span>
                      )}
                      {file.has_overrides && (
                        <span className="rounded bg-brand/15 px-1.5 py-0.5 text-brand">override</span>
                      )}
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className="min-w-0 flex-1 overflow-hidden rounded-lg border border-border bg-surface-100">
          <div className="px-4 py-3 border-b border-border">
            <div className="flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-semibold text-zinc-100 truncate">{detail?.name ?? "Select a file"}</p>
                <p className="text-[11px] text-zinc-400 truncate">{detail?.relative_path ?? "No file selected"}</p>
              </div>
              <span
                className={[
                  "text-[11px] font-medium px-2 py-1 rounded",
                  saveState === "saving"
                    ? "bg-amber-500/15 text-amber-400"
                    : saveState === "error"
                    ? "bg-red-500/15 text-red-400"
                    : "bg-emerald-500/15 text-emerald-400",
                ].join(" ")}
              >
                {saveState === "saving" ? "Saving…" : saveState === "error" ? "Save failed" : "Saved"}
              </span>
            </div>

            <div className="flex items-center gap-1 mt-3 overflow-x-auto">
              <TabButton active={tab === "eeg"} label="EEG" icon={Waves} onClick={() => setTab("eeg")} />
              <TabButton active={tab === "psd"} label="PSD" icon={Activity} onClick={() => setTab("psd")} />
              <TabButton active={tab === "report"} label="Run Report" icon={FileText} onClick={() => setTab("report")} />
              <TabButton active={tab === "ica"} label="ICA" icon={Brain} onClick={() => setTab("ica")} />
            </div>
          </div>

          <div className="min-h-[clamp(34rem,68vh,52rem)] p-4">
            {detailLoading ? (
              <div className="flex items-center justify-center gap-2 py-20 text-zinc-500">
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading review workspace...
              </div>
            ) : detailError ? (
              <ErrorBanner message={detailError} />
            ) : !detail ? (
              <div className="py-20 text-center text-zinc-600">Select a file to review</div>
            ) : tab === "eeg" ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-zinc-400">
                  <div className="flex flex-wrap items-center gap-2">
                    <span>
                      {manifest ? `${manifest.n_epochs} epochs · ${manifest.n_channels} channels · ${manifest.sampling_rate.toFixed(0)} Hz` : "Loading EEG…"}
                    </span>
                    <button
                      type="button"
                      onClick={() => setShowEegHelp((value) => !value)}
                      className="inline-flex items-center gap-2 rounded-full border border-border bg-surface-50/70 px-3 py-1 text-[11px] font-medium text-zinc-200 hover:bg-surface-50 hover:text-zinc-100"
                      aria-label={showEegHelp ? "Hide EEG help" : "Show EEG help"}
                      title={showEegHelp ? "Hide EEG help" : "Show EEG help"}
                    >
                      <span className="flex h-4 w-4 items-center justify-center rounded-full border border-border text-[10px] font-semibold">?</span>
                      Help &amp; Color Key
                    </button>
                  </div>
                  <div className="flex flex-wrap items-center justify-end gap-2">
                    {artifact.postedit ? (
                      <a href={artifact.postedit} target="_blank" rel="noreferrer" className="text-brand hover:underline">
                        Open postedit export
                      </a>
                    ) : null}
                  </div>
                </div>

                {showEegHelp ? (
                  <div className="rounded-lg border border-border bg-surface-50/60 p-3">
                    <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
                      <div className="space-y-2">
                        <p className="text-[11px] uppercase tracking-wider text-zinc-500">Directions</p>
                        <div className="grid gap-2 text-xs text-zinc-300 sm:grid-cols-2">
                          <p>Click a trace to focus an epoch.</p>
                          <p>Double-click or press Space to reject or restore the focused epoch.</p>
                          <p>Use the bottom scrollbar to move through the full file.</p>
                          <p>Right-click a trace to open a scalp map for that exact latency.</p>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <p className="text-[11px] uppercase tracking-wider text-zinc-500">Color Key</p>
                        <div className="grid gap-2 text-xs text-zinc-300">
                          <div className="flex items-center gap-2">
                            <span className="h-3 w-3 rounded-full bg-[#60a5fa]" />
                            <span>Blue trace and header tint: focused epoch</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="h-3 w-3 rounded-full bg-[#f87171]" />
                            <span>Red trace and header tint: rejected epoch</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="h-3 w-3 rounded-full bg-[#e2e8f0]" />
                            <span>Light dotted vertical lines: epoch boundaries</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null}

                <div className="rounded-md border border-border bg-surface-50/40 px-3 py-2 text-xs text-zinc-300">
                  Click to focus. Drag across epochs to reject or restore a contiguous range. Double-click or press <span className="font-medium text-zinc-100">Space</span> for one-off changes. Right-click for a scalp map at the clicked latency.
                </div>

                <div className="space-y-3">
                  <div className="relative">
                    <div className={epochWindowLoading ? "pointer-events-none opacity-40 blur-[1px]" : ""}>
                      <EegBrowser
                        epochWindow={epochWindow}
                        manifest={manifest}
                        badEpochs={badEpochs}
                        focusedEpoch={focusedEpoch}
                        scaleUv={scaleUv}
                        channelHeight={channelHeight}
                        visibleEpochCount={visibleEpochCount}
                        epochStart={epochStart}
                        onFocusEpoch={setFocusedEpoch}
                        onToggleEpoch={(epochIndex) => {
                          setFocusedEpoch(epochIndex);
                          toggleEpoch(epochIndex);
                        }}
                        onApplyEpochRange={applyEpochRangeAction}
                        onEpochStartChange={setEpochStart}
                        onOpenTopography={openTopography}
                      />
                    </div>
                    {epochWindowLoading ? (
                      <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-surface-200/35 backdrop-blur-[1px]">
                        <div className="flex items-center gap-2 rounded-full border border-border bg-surface-100/90 px-4 py-2 text-sm text-zinc-200 shadow-lg">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Loading EEG window...
                        </div>
                      </div>
                    ) : null}
                  </div>
                  <div className="rounded-lg border border-border bg-surface-50/50 p-3">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => {
                            const prevStart = Math.max(0, epochStart - visibleEpochCount);
                            setEpochStart(prevStart);
                          }}
                          className="rounded border border-border px-2 py-1 text-zinc-300 hover:bg-surface-50"
                        >
                          Previous
                        </button>
                        <button
                          onClick={() => {
                            if (!manifest) return;
                            const maxStart = Math.max(0, manifest.n_epochs - visibleEpochCount);
                            const nextStart = Math.min(maxStart, epochStart + visibleEpochCount);
                            setEpochStart(nextStart);
                          }}
                          className="rounded border border-border px-2 py-1 text-zinc-300 hover:bg-surface-50"
                        >
                          Next
                        </button>
                      </div>
                      <div className="flex flex-wrap gap-4 text-xs text-zinc-300">
                        <p>Leftmost epoch: {epochStart + 1}</p>
                        <p>Focused epoch: {focusedEpoch != null ? focusedEpoch + 1 : "None"}</p>
                        <p>Rejected shown: {visibleEpochsInView.filter((epoch) => badEpochs.includes(epoch.epoch_index)).length}</p>
                        <p>Total rejected: {badEpochs.length}</p>
                      </div>
                    </div>
                    <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-3">
                      <label className="block text-[11px] uppercase tracking-wider text-zinc-600">
                        Visible Epochs
                        <input
                          type="number"
                          min={1}
                          max={Math.max(1, manifest?.n_epochs ?? 1)}
                          step={1}
                          value={visibleEpochCount}
                          onChange={(event) => setVisibleEpochCountBounded(Number(event.target.value) || 1)}
                          className="mt-2 w-full rounded border border-border bg-surface-100 px-2 py-1.5 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-brand/50"
                        />
                        <input
                          type="range"
                          min={1}
                          max={Math.max(1, manifest?.n_epochs ?? 1)}
                          step={1}
                          value={visibleEpochCount}
                          onChange={(event) => setVisibleEpochCountBounded(Number(event.target.value))}
                          className="mt-2 w-full"
                        />
                        <div className="mt-1 text-xs text-zinc-300">
                          {visibleEpochCount} epochs visible at once. More epochs = wider view.
                        </div>
                      </label>
                      <label className="block text-[11px] uppercase tracking-wider text-zinc-600">
                        Scale
                        <input
                          type="number"
                          min={1}
                          max={1000}
                          step={1}
                          value={scaleUv}
                          onChange={(event) => setScaleUvBounded(Number(event.target.value) || 1)}
                          className="mt-2 w-full rounded border border-border bg-surface-100 px-2 py-1.5 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-brand/50"
                        />
                        <input
                          type="range"
                          min={1}
                          max={1000}
                          step={1}
                          value={scaleUv}
                          onChange={(event) => setScaleUvBounded(Number(event.target.value))}
                          className="mt-2 w-full"
                        />
                        <div className="mt-1 text-xs text-zinc-400">±{scaleUv} uV</div>
                      </label>
                      <label className="block text-[11px] uppercase tracking-wider text-zinc-600">
                        Channel Height
                        <input
                          type="number"
                          min={4}
                          max={56}
                          step={1}
                          value={channelHeight}
                          onChange={(event) => setChannelHeightBounded(Number(event.target.value) || 4)}
                          className="mt-2 w-full rounded border border-border bg-surface-100 px-2 py-1.5 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-brand/50"
                        />
                        <input
                          type="range"
                          min={4}
                          max={56}
                          step={1}
                          value={channelHeight}
                          onChange={(event) => setChannelHeightBounded(Number(event.target.value))}
                          className="mt-2 w-full"
                        />
                        <div className="mt-1 text-xs text-zinc-400">{channelHeight}px rows</div>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            ) : tab === "psd" ? (
              artifact.psd ? (
                <div className="space-y-3">
                  <div className="flex justify-end">
                    <a href={artifact.psd} target="_blank" rel="noreferrer" className="text-xs text-brand hover:underline">
                      Open PSD in new tab
                    </a>
                  </div>
                  <img src={artifact.psd} alt="PSD overview" className="w-full rounded-lg border border-border" />
                </div>
              ) : (
                <div className="py-20 text-center text-zinc-600">PSD overview not available</div>
              )
            ) : tab === "report" ? (
              artifact.run_report ? (
                <div className="space-y-3">
                  <div className="flex justify-end">
                    <a href={artifact.run_report} target="_blank" rel="noreferrer" className="text-xs text-brand hover:underline">
                      Open report in new tab
                    </a>
                  </div>
                  <iframe title="Run report" src={artifact.run_report} className="w-full min-h-[38rem] rounded-lg border border-border bg-white" />
                </div>
              ) : (
                <div className="py-20 text-center text-zinc-600">Run report not available</div>
              )
            ) : artifact.ica_report ? (
              <div className="space-y-4">
                <div className="flex flex-wrap justify-end gap-3">
                  {typeof icaSummary?.structure?.topo_grid_page === "number" ? (
                    <a
                      href={`${artifact.ica_report}#page=${Number(icaSummary.structure.topo_grid_page) + 1}`}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs text-brand hover:underline"
                    >
                      Open topography overview
                    </a>
                  ) : null}
                  <a href={artifact.ica_report} target="_blank" rel="noreferrer" className="text-xs text-brand hover:underline">
                    Open ICA PDF in new tab
                  </a>
                </div>
                <iframe title="ICA report" src={artifact.ica_report} className="w-full min-h-[28rem] rounded-lg border border-border bg-white" />
                <div className="rounded-lg border border-border bg-surface-50 p-4">
                  <h3 className="text-sm font-semibold text-zinc-100 mb-3">ICA Override Context</h3>
                  <div className="text-xs text-zinc-400 space-y-1">
                    <p>Baseline rejected ICA: {detail.baseline_rejected_ica.length ? detail.baseline_rejected_ica.join(", ") : "None"}</p>
                    <p>Manual rejected ICA: {manualRejectedIca.length ? manualRejectedIca.join(", ") : "None"}</p>
                  </div>
                  {icaSummary?.components?.length ? (
                    <div className="mt-4 overflow-auto rounded border border-border">
                      <table className="w-full text-xs">
                        <thead className="bg-surface-100 text-zinc-500">
                          <tr>
                            <th className="px-2 py-1.5 text-left">Component</th>
                            <th className="px-2 py-1.5 text-left">Type</th>
                            <th className="px-2 py-1.5 text-left">Confidence</th>
                            <th className="px-2 py-1.5 text-left">Rejected</th>
                          </tr>
                        </thead>
                        <tbody>
                          {icaSummary.components.slice(0, 16).map((component) => (
                            <tr key={component.component} className="border-t border-border-subtle">
                              <td className="px-2 py-1.5 text-zinc-200">{component.component}</td>
                              <td className="px-2 py-1.5 text-zinc-400">{component.type}</td>
                              <td className="px-2 py-1.5 text-zinc-400">{component.confidence.toFixed(3)}</td>
                              <td className="px-2 py-1.5">
                                <div className="flex items-center gap-2">
                                  <span className={component.rejected ? "text-red-400" : "text-emerald-400"}>
                                    {component.rejected ? "Yes" : "No"}
                                  </span>
                                  {typeof icaSummary.structure?.detail_page_map === "object" &&
                                  component.component in (icaSummary.structure.detail_page_map as Record<string, number>) ? (
                                    <a
                                      href={`${artifact.ica_report}#page=${Number((icaSummary.structure.detail_page_map as Record<string, number>)[component.component]) + 1}`}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="text-brand hover:underline"
                                    >
                                      Page
                                    </a>
                                  ) : null}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}
                </div>
              </div>
            ) : (
              <div className="py-20 text-center text-zinc-600">ICA report not available</div>
            )}
          </div>
        </section>
      </div>

      <section className="space-y-4 rounded-lg border border-border bg-surface-100 p-4">
        <div className="flex flex-col gap-1">
          <h3 className="text-sm font-semibold text-zinc-100">Review Details</h3>
          <p className="text-xs text-zinc-400">
            Capture file notes, inspect summary metrics, edit overrides, then run reprocess or QA export.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(16rem,0.9fr)_minmax(15rem,0.8fr)_minmax(22rem,1.35fr)]">
          <div className="space-y-5">
          <section className="space-y-3 rounded-lg border border-border bg-surface-50/60 p-4">
            <div className="flex items-center gap-2">
              <StickyNote className="w-4 h-4 text-zinc-500" />
              <h3 className="text-sm font-semibold text-zinc-100">Review State</h3>
            </div>
            <div className="space-y-2">
              <label className="block text-[11px] uppercase tracking-wider text-zinc-500">Status</label>
              <select
                value={status}
                onChange={(e) => {
                  const next = e.target.value;
                  setStatus(next);
                  scheduleNotesSave(notes, next);
                }}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
              >
                <option value="UNSET">Unset</option>
                <option value="PASS">Pass</option>
                <option value="FAIL">Fail</option>
                <option value="REVIEW">Review</option>
              </select>
              <textarea
                value={notes}
                onChange={(e) => {
                  const next = e.target.value;
                  setNotes(next);
                  scheduleNotesSave(next, status);
                }}
                placeholder="Add reviewer notes..."
                className="w-full min-h-28 rounded-md border border-border bg-surface-100 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 outline-none"
              />
            </div>
          </section>
          </div>

          <div className="space-y-5">
          <section className="space-y-3 rounded-lg border border-border bg-surface-50/50 p-4">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-zinc-500" />
              <h3 className="text-sm font-semibold text-zinc-100">Metrics</h3>
            </div>
            <div className="space-y-2 p-1">
              <MetricRow label="Data retained" value={String(metrics.data_retained ?? "—")} />
              <MetricRow label="Channels retained" value={`${metrics.channels_retained ?? "—"} / ${metrics.channels_original ?? "—"}`} />
              <MetricRow label="Epochs reviewed" value={`${detail?.epoch_review.bad_epochs_count ?? 0} bad`} />
              <MetricRow label="Baseline ICA" value={detail?.baseline_rejected_ica.length ? detail.baseline_rejected_ica.join(", ") : "None"} />
              <MetricRow label="Reprocess fix type" value={detail?.reprocess.fix_type || "None"} />
              <MetricRow label="QA export" value={detail?.qa_export.timestamp || "Not exported"} />
            </div>
          </section>
          </div>

          <div className="space-y-5">
          <section className="space-y-4 p-1">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-zinc-500" />
              <h3 className="text-sm font-semibold text-zinc-100">Manual Overrides</h3>
            </div>

            <div className="space-y-4 rounded-lg border border-border bg-surface-100/70 p-4">
              <div className="space-y-1">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">Edit Overrides</h4>
                <p className="text-xs text-zinc-400">Adjust channels or ICA decisions before running a new pass.</p>
              </div>

              <div className="space-y-2">
                <DiffChips label="Bad Channels" baseline={detail?.baseline_bad_channels ?? []} manual={manualBadChannels} />
                <div className="flex gap-2">
                  <input
                    value={channelDraft}
                    onChange={(e) => setChannelDraft(e.target.value)}
                    placeholder="e.g. E8 or 8"
                    className="flex-1 rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
                  />
                  <button
                    onClick={() => {
                      const next = normalizeBadChannelInput(channelDraft);
                      if (!next) return;
                      if (!validChannelSet.has(next)) {
                        setActionError(`Invalid bad channel override: ${next}`);
                        return;
                      }
                      setActionError(null);
                      if (!manualBadChannels.includes(next)) setManualBadChannels([...manualBadChannels, next].sort());
                      setChannelDraft("");
                    }}
                    className="rounded-md bg-brand px-3 py-2 text-sm font-medium text-brand-900"
                  >
                    Add
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {manualBadChannels.map((channel) => (
                    <button
                      key={channel}
                      onClick={() => setManualBadChannels(manualBadChannels.filter((v) => v !== channel))}
                      className="rounded-full border border-border px-2 py-1 text-xs text-zinc-200 hover:bg-surface-50"
                    >
                      {channel} ×
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <DiffChips label="Rejected ICA Components" baseline={detail?.baseline_rejected_ica ?? []} manual={manualRejectedIca} prefix="IC " />
                <div className="flex gap-2">
                  <input
                    value={icaDraft}
                    onChange={(e) => setIcaDraft(e.target.value)}
                    placeholder="e.g. 3"
                    className="flex-1 rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 outline-none"
                  />
                  <button
                    onClick={() => {
                      const parsed = Number.parseInt(icaDraft, 10);
                      if (Number.isNaN(parsed)) return;
                      if (!isValidIcaComponent(parsed, detail?.max_components ?? 0)) {
                        const maxText = (detail?.max_components ?? 0) > 0 ? `0-${(detail?.max_components ?? 1) - 1}` : "available range";
                        setActionError(`Invalid ICA override: IC ${parsed}. Valid range is ${maxText}.`);
                        return;
                      }
                      setActionError(null);
                      if (!manualRejectedIca.includes(parsed)) {
                        setManualRejectedIca([...manualRejectedIca, parsed].sort((a, b) => a - b));
                      }
                      setIcaDraft("");
                    }}
                    className="rounded-md bg-brand px-3 py-2 text-sm font-medium text-brand-900"
                  >
                    Add
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {manualRejectedIca.map((component) => (
                    <button
                      key={component}
                      onClick={() => setManualRejectedIca(manualRejectedIca.filter((v) => v !== component))}
                      className="rounded-full border border-border px-2 py-1 text-xs text-zinc-200 hover:bg-surface-50"
                    >
                      IC {component} ×
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={saveOverrides}
                disabled={invalidCombinedOverrideChange}
                className="w-full rounded-md border border-brand/40 bg-brand/10 px-3 py-2 text-sm font-medium text-brand hover:bg-brand/20 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Save Overrides
              </button>

              <p className="text-[11px] text-zinc-400">
                Baseline bad channels: {detail?.baseline_bad_channels.length ? detail.baseline_bad_channels.join(", ") : "None"}
              </p>
            </div>

            <div className="space-y-3 rounded-lg border border-border bg-surface-100/90 p-4">
              <div className="space-y-1">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">Run Actions</h4>
                <p className="text-xs text-zinc-400">Use these after notes, epochs, and overrides are in the state you want to preserve.</p>
              </div>

              <button
                onClick={startReprocess}
                disabled={invalidCombinedOverrideChange}
                className="w-full rounded-md bg-brand px-3 py-2 text-sm font-semibold text-brand-900 hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Reprocess with Overrides
              </button>

              {invalidCombinedOverrideChange ? (
                <p className="text-[11px] text-amber-400">
                  Change either bad channels or ICA in this run, not both. Channel overrides can carry forward into a later ICA run.
                </p>
              ) : null}

              <button
                onClick={exportQa}
                className="w-full rounded-md border border-border px-3 py-2 text-sm font-medium text-zinc-200 hover:bg-surface-50"
              >
                Export QA File
              </button>

              {reprocessStatus && (
                <div className="rounded-md border border-border bg-surface-50 p-3 text-xs">
                <p className="font-medium text-zinc-200">Reprocess: {reprocessStatus}</p>
                  {reprocessMessage && <p className="mt-1 text-zinc-400">{reprocessMessage}</p>}
                  {detail?.reprocess.timestamp ? <p className="mt-1 text-zinc-500">Last update: {detail.reprocess.timestamp}</p> : null}
                </div>
              )}

              {qaExportMessage && (
                <div className="rounded-md border border-border bg-surface-50 p-3 text-xs">
                  <p className="font-medium text-zinc-200">QA Export</p>
                  <p className="mt-1 text-zinc-400">{qaExportMessage}</p>
                  <div className="mt-2 flex flex-wrap gap-3">
                    {detail?.qa_export.path ? (
                      <a href={api.getExcludeQaLogUrl()} className="inline-block text-brand hover:underline">
                        Open QA preprocessing log
                      </a>
                    ) : null}
                    {artifact.postedit ? (
                      <a href={artifact.postedit} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-brand hover:underline">
                        <MonitorDown className="h-3.5 w-3.5" />
                        Open postedit export
                      </a>
                    ) : null}
                  </div>
                </div>
              )}
            </div>
          </section>
          </div>
        </div>
      </section>
      <TopographyModal
        open={topographyRequest != null}
        loading={topographyLoading}
        error={topographyError}
        data={topographyData}
        request={topographyRequest}
        onClose={() => {
          setTopographyRequest(null);
          setTopographyData(null);
          setTopographyError(null);
        }}
      />
    </div>
  );
}
