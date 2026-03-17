import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, Brain, FileText, Loader2, MonitorDown, Search, SlidersHorizontal, StickyNote, Waves } from "lucide-react";
import { api } from "../lib/api";
import type {
  DashboardStatus,
  EpochManifest,
  EpochWindowResponse,
  ExcludeFileDetail,
  ExcludeFileSummary,
  ExcludeIcaSummaryResponse,
} from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";

type TabKey = "eeg" | "psd" | "report" | "ica";

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
  timeZoom,
  onFocusEpoch,
  onToggleEpoch,
}: {
  epochWindow: EpochWindowResponse | null;
  manifest: EpochManifest | null;
  badEpochs: number[];
  focusedEpoch: number | null;
  scaleUv: number;
  channelHeight: number;
  timeZoom: number;
  onFocusEpoch: (epochIndex: number) => void;
  onToggleEpoch: (epochIndex: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const headerHeight = 34;
  const labelWidth = 88;
  const [frameWidth, setFrameWidth] = useState(1200);

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
    const canvas = canvasRef.current;
    if (!canvas || !epochWindow) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const channels = epochWindow.channel_names;
    const epochs = epochWindow.epochs;
    const samplesPerEpoch = epochs[0]?.traces_uv[channels[0] ?? ""]?.length ?? 0;
    const availableWidth = Math.max(360, frameWidth - 2);
    const traceWidth = Math.max(220, availableWidth - labelWidth);
    const epochWidth = traceWidth / Math.max(1, epochs.length);
    const width = availableWidth;
    const height = headerHeight + channels.length * channelHeight;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, labelWidth, height);
    ctx.fillStyle = "#111827";
    ctx.fillRect(labelWidth, 0, width - labelWidth, headerHeight);

    ctx.font = "12px monospace";
    ctx.textBaseline = "middle";

    epochs.forEach((epoch, epochIdx) => {
      const x0 = labelWidth + epochIdx * epochWidth;
      const isBad = badEpochs.includes(epoch.epoch_index);
      const isFocused = focusedEpoch === epoch.epoch_index;
      ctx.fillStyle = isBad ? "rgba(248,113,113,0.18)" : isFocused ? "rgba(96,165,250,0.18)" : "rgba(255,255,255,0.02)";
      ctx.fillRect(x0, 0, epochWidth, headerHeight);
      ctx.strokeStyle = isFocused ? "#60a5fa" : "rgba(255,255,255,0.08)";
      ctx.strokeRect(x0 + 0.5, 0.5, epochWidth - 1, headerHeight - 1);
      ctx.fillStyle = isBad ? "#fca5a5" : isFocused ? "#93c5fd" : "#cbd5e1";
      ctx.fillText(`Epoch ${epoch.epoch_index + 1}`, x0 + 8, headerHeight / 2);
      ctx.fillStyle = "#64748b";
      const suffix = epoch.event_code ? `E${epoch.event_code}` : `${epoch.start_time_seconds.toFixed(2)}s`;
      ctx.fillText(suffix, x0 + Math.max(78, epochWidth - 68), headerHeight / 2);
    });

    channels.forEach((channel, channelIndex) => {
      const y0 = headerHeight + channelIndex * channelHeight;
      const midY = y0 + channelHeight / 2;

      ctx.fillStyle = focusedEpoch != null && epochWindow.epochs.some((epoch) => epoch.epoch_index === focusedEpoch)
        ? "rgba(255,255,255,0.01)"
        : "rgba(255,255,255,0)";
      ctx.fillRect(0, y0, width, channelHeight);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.beginPath();
      ctx.moveTo(0, y0 + 0.5);
      ctx.lineTo(width, y0 + 0.5);
      ctx.stroke();

      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(channel, 8, midY);
      ctx.fillStyle = "#64748b";
      ctx.fillText(`±${scaleUv}uV`, labelWidth - 54, midY);

      epochs.forEach((epoch, epochIdx) => {
        const x0 = labelWidth + epochIdx * epochWidth;
        const values = epoch.traces_uv[channel] ?? [];
        if (!values.length) return;
        const xStep = epochWidth / Math.max(1, values.length - 1);
        const isBad = badEpochs.includes(epoch.epoch_index);
        ctx.beginPath();
        values.forEach((value, sampleIdx) => {
          const x = x0 + sampleIdx * xStep;
          const normalized = value / Math.max(scaleUv, 1);
          const y = midY - normalized * (channelHeight * 0.38);
          if (sampleIdx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = isBad ? "#f87171" : focusedEpoch === epoch.epoch_index ? "#e2e8f0" : "#60a5fa";
        ctx.lineWidth = focusedEpoch === epoch.epoch_index ? 1.35 : 1;
        ctx.stroke();
      });
    });
  }, [badEpochs, channelHeight, epochWindow, focusedEpoch, frameWidth, manifest, onFocusEpoch, scaleUv, timeZoom]);

  if (!epochWindow || !manifest) {
    return <div className="py-16 text-center text-sm text-zinc-500">Loading EEG…</div>;
  }

  const channels = epochWindow.channel_names;
  const traceWidth = Math.max(220, frameWidth - labelWidth - 2);
  const epochWidth = traceWidth / Math.max(1, epochWindow.epochs.length);
  const canvasWidth = Math.max(360, frameWidth);
  const canvasHeight = headerHeight + channels.length * channelHeight;

  const getEpochIndexFromPointer = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const scaleX = rect.width > 0 ? canvasWidth / rect.width : 1;
    const localX = (event.clientX - rect.left) * scaleX;
    if (localX < labelWidth) return null;
    return Math.max(0, Math.min(epochWindow.epochs.length - 1, Math.floor((localX - labelWidth) / epochWidth)));
  };

  return (
    <div className="space-y-3">
      <div ref={frameRef} className="rounded-lg border border-border bg-surface-50/60 overflow-y-auto overflow-x-hidden max-h-[36rem]">
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          className="block w-full"
          onClick={(event) => {
            const epochIdx = getEpochIndexFromPointer(event);
            if (epochIdx == null) return;
            const epoch = epochWindow.epochs[epochIdx];
            if (epoch) onFocusEpoch(epoch.epoch_index);
          }}
          onDoubleClick={(event) => {
            const epochIdx = getEpochIndexFromPointer(event);
            if (epochIdx == null) return;
            const epoch = epochWindow.epochs[epochIdx];
            if (epoch) onToggleEpoch(epoch.epoch_index);
          }}
        />
      </div>
      <div className="flex items-center justify-between text-[11px] text-zinc-500">
        <span>Scroll vertically for channels. Click a trace block to focus an epoch. Double-click or press Space to reject or restore it.</span>
        <span>{manifest.n_channels} channels · {epochWindow.epochs.length} visible epochs</span>
      </div>
    </div>
  );
}

export default function ExcludePage() {
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
  const [showFileList, setShowFileList] = useState(true);
  const [reprocessJobId, setReprocessJobId] = useState<string | null>(null);
  const [reprocessStatus, setReprocessStatus] = useState<string | null>(null);
  const [reprocessMessage, setReprocessMessage] = useState<string | null>(null);
  const [qaExportMessage, setQaExportMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const epochSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const notesSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastEpochSaveRef = useRef<{ fileKey: string; badEpochs: number[] } | null>(null);
  const lastNotesSaveRef = useRef<{ fileKey: string; notes: string; status: string } | null>(null);
  const loadedManualBadChannelsRef = useRef<string[]>([]);
  const loadedManualRejectedIcaRef = useRef<number[]>([]);

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
    setScaleUv(clamp(Math.round(value), 1, 150));
  };

  const setChannelHeightBounded = (value: number) => {
    setChannelHeight(clamp(Math.round(value), 4, 56));
  };

  async function loadFiles(preserveSelection = true) {
    const [data, statusData] = await Promise.all([
      api.getExcludeFiles(),
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
      const result = await api.saveExcludeEpochReview(pending.fileKey, pending.badEpochs);
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
      await api.saveExcludeNotes(pending.fileKey, pending.notes, pending.status);
      setSaveState("saved");
      lastNotesSaveRef.current = null;
    } catch {
      setSaveState("error");
      setActionError("Could not save review notes.");
    }
  }

  useEffect(() => {
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
  }, [selectedKey]);

  useEffect(() => {
    const id = setInterval(() => {
      loadFiles(true).catch(() => {});
    }, 15000);
    return () => clearInterval(id);
  }, [selectedKey]);

  useEffect(() => {
    if (!selectedKey) return;
    let cancelled = false;
    setDetailLoading(true);
    setDetailError(null);
    Promise.all([
      api.getExcludeFile(selectedKey),
      api.getExcludeEpochManifest(selectedKey),
      api.getExcludeIcaSummary(selectedKey).catch(() => null),
    ])
      .then(([fileDetail, epochManifest, summary]) => {
        if (cancelled) return;
        setActionError(null);
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
  }, [selectedKey]);

  useEffect(() => {
    if (!selectedKey || !manifest) return;
    let cancelled = false;
    setEpochWindowLoading(true);
    api.getExcludeEpochWindow(selectedKey, epochStart, visibleEpochCount, manifest.channel_names)
      .then((window) => {
        if (cancelled) return;
        setEpochWindow(window);
        setFocusedEpoch((current) => {
          if (current != null && window.epochs.some((epoch) => epoch.epoch_index === current)) {
            return current;
          }
          return window.epochs[0]?.epoch_index ?? null;
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
  }, [epochStart, manifest, selectedKey, visibleEpochCount]);

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
                api.getExcludeFiles(),
                api.getExcludeFile(selectedKey),
                api.getExcludeEpochManifest(selectedKey),
                api.getExcludeIcaSummary(selectedKey).catch(() => null),
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
  }, [reprocessJobId]);

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
      api.saveExcludeEpochReview(selectedKey, nextBadEpochs)
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

  function scheduleNotesSave(nextNotes: string, nextStatus: string) {
    if (!selectedKey) return;
    setSaveState("saving");
    lastNotesSaveRef.current = { fileKey: selectedKey, notes: nextNotes, status: nextStatus };
    if (notesSaveTimer.current) clearTimeout(notesSaveTimer.current);
    notesSaveTimer.current = setTimeout(() => {
      api.saveExcludeNotes(selectedKey, nextNotes, nextStatus)
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
      await api.saveExcludeOverrides(selectedKey, manualBadChannels, manualRejectedIca);
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
      const result = await api.startExcludeReprocess(selectedKey, manualBadChannels, manualRejectedIca);
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
        const refreshed = await api.getExcludeFile(selectedKey);
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
  const validChannelSet = new Set((detail?.valid_channels ?? []).map((value) => String(value).toUpperCase()));
  const channelOverridesDirty = !arraysEqual(manualBadChannels, loadedManualBadChannelsRef.current);
  const icaOverridesDirty = !arraysEqual(manualRejectedIca, loadedManualRejectedIcaRef.current);
  const invalidCombinedOverrideChange = channelOverridesDirty && icaOverridesDirty;

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">Exclude</h2>
          <p className="text-sm text-zinc-500 mt-1">
            Browser-native review for epoch rejection, notes, manual overrides, and report inspection.
          </p>
        </div>
        <div className="text-left xl:text-right">
          <p className="text-[11px] uppercase tracking-wider text-zinc-600">Workspace</p>
          <p className="text-xs text-zinc-400 font-mono max-w-[32rem] truncate" title={workspaceDir}>
            {workspaceDir || "Loading..."}
          </p>
          <p className="text-[11px] uppercase tracking-wider text-zinc-600">Exports Root</p>
          <p className="text-xs text-zinc-400 font-mono max-w-[32rem] truncate" title={exportsRoot}>
            {exportsRoot || "Loading..."}
          </p>
        </div>
      </div>

      {listError && <ErrorBanner message={listError} />}
      {actionError && <ErrorBanner message={actionError} />}

      <div className="flex items-center justify-between gap-3">
        <button
          onClick={() => setShowFileList((value) => !value)}
          className="inline-flex items-center gap-2 rounded-md border border-border bg-surface-100 px-3 py-2 text-sm text-zinc-200 hover:bg-surface-50"
        >
          <Search className="h-4 w-4 text-zinc-500" />
          {showFileList ? "Hide Files" : "Show Files"}
        </button>
        <div className="text-xs text-zinc-500">
          {files.length} file{files.length === 1 ? "" : "s"} in Exclude workspace
        </div>
      </div>

      <div className="flex flex-col gap-5 xl:flex-row">
        {showFileList ? (
          <aside className="w-full shrink-0 rounded-lg border border-border bg-surface-100 overflow-hidden xl:w-[18rem]">
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
          <div className="max-h-[70vh] overflow-auto">
            {listLoading ? (
              <div className="flex items-center justify-center gap-2 py-10 text-zinc-500 text-sm">
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading files...
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
                    "w-full px-3 py-3 text-left border-b border-border-subtle hover:bg-surface-50/40 transition-colors",
                    selectedKey === file.file_key ? "bg-brand/10" : "",
                  ].join(" ")}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-zinc-100 truncate">{file.name}</p>
                      <p className="text-[11px] text-zinc-500 truncate">{file.relative_path}</p>
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
        ) : null}

        <section className="min-w-0 flex-1 rounded-lg border border-border bg-surface-100 overflow-hidden">
          <div className="px-4 py-3 border-b border-border">
            <div className="flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-semibold text-zinc-100 truncate">{detail?.name ?? "Select a file"}</p>
                <p className="text-[11px] text-zinc-500 truncate">{detail?.relative_path ?? "No file selected"}</p>
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

          <div className="p-4 min-h-[42rem]">
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
                <div className="flex items-center justify-between gap-4 text-xs text-zinc-500">
                  <div>
                    {manifest ? `${manifest.n_epochs} epochs · ${manifest.n_channels} channels · ${manifest.sampling_rate.toFixed(0)} Hz` : "Loading EEG…"}
                  </div>
                  <div className="flex flex-wrap items-center justify-end gap-2">
                    {artifact.postedit ? (
                      <a href={artifact.postedit} target="_blank" rel="noreferrer" className="text-brand hover:underline">
                        Open postedit export
                      </a>
                    ) : null}
                  </div>
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
                        timeZoom={1}
                        onFocusEpoch={setFocusedEpoch}
                        onToggleEpoch={(epochIndex) => {
                          setFocusedEpoch(epochIndex);
                          toggleEpoch(epochIndex);
                        }}
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
                      <div className="flex flex-wrap gap-4 text-xs text-zinc-400">
                        <p>Focused epoch: {focusedEpoch != null ? focusedEpoch + 1 : "None"}</p>
                        <p>Rejected in view: {epochWindow?.epochs.filter((epoch) => badEpochs.includes(epoch.epoch_index)).length ?? 0}</p>
                        <p>Total rejected: {badEpochs.length}</p>
                      </div>
                    </div>
                    <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
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
                        <div className="mt-1 text-xs text-zinc-400">{visibleEpochCount} epochs per page</div>
                      </label>
                      <label className="block text-[11px] uppercase tracking-wider text-zinc-600">
                        Scale
                        <input
                          type="number"
                          min={1}
                          max={150}
                          step={1}
                          value={scaleUv}
                          onChange={(event) => setScaleUvBounded(Number(event.target.value) || 1)}
                          className="mt-2 w-full rounded border border-border bg-surface-100 px-2 py-1.5 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-brand/50"
                        />
                        <input
                          type="range"
                          min={1}
                          max={150}
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

      <section className="rounded-lg border border-border bg-surface-100 p-4 space-y-5">
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
          <div className="space-y-5">
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <StickyNote className="w-4 h-4 text-zinc-500" />
              <h3 className="text-sm font-semibold text-zinc-100">Review State</h3>
            </div>
            <div className="space-y-2">
              <label className="block text-[11px] uppercase tracking-wider text-zinc-600">Status</label>
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
                className="w-full min-h-28 rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none"
              />
            </div>
          </section>
          </div>

          <div className="space-y-5">
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-zinc-500" />
              <h3 className="text-sm font-semibold text-zinc-100">Metrics</h3>
            </div>
            <div className="space-y-2 rounded-md border border-border bg-surface-50 p-3">
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
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-zinc-500" />
              <h3 className="text-sm font-semibold text-zinc-100">Manual Overrides</h3>
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
                  className="rounded-md bg-brand px-3 py-2 text-sm font-medium text-surface-500"
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
                  className="rounded-md bg-brand px-3 py-2 text-sm font-medium text-surface-500"
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

            <button
              onClick={startReprocess}
              disabled={invalidCombinedOverrideChange}
              className="w-full rounded-md bg-brand px-3 py-2 text-sm font-semibold text-surface-500 hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-50"
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
                {reprocessMessage && <p className="text-zinc-500 mt-1">{reprocessMessage}</p>}
                {detail?.reprocess.timestamp ? <p className="text-zinc-600 mt-1">Last update: {detail.reprocess.timestamp}</p> : null}
              </div>
            )}

            {qaExportMessage && (
              <div className="rounded-md border border-border bg-surface-50 p-3 text-xs">
                <p className="font-medium text-zinc-200">QA Export</p>
                <p className="text-zinc-500 mt-1">{qaExportMessage}</p>
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

            <p className="text-[11px] text-zinc-600">
              Baseline bad channels: {detail?.baseline_bad_channels.length ? detail.baseline_bad_channels.join(", ") : "None"}
            </p>
          </section>
          </div>
        </div>
      </section>
    </div>
  );
}
