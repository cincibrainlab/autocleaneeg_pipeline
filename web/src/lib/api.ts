type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API ${response.status}: ${text}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return (await response.text()) as T;
  }
  return (await response.json()) as T;
}

function json<T>(path: string, method: HttpMethod = "GET", body?: unknown): Promise<T> {
  return request<T>(path, {
    method,
    body: body == null ? undefined : JSON.stringify(body),
  });
}

export interface DashboardStatus {
  configured: boolean;
  workspace_dir?: string;
  output_dir?: string;
  mode: "test" | "live" | string;
  queue: { pending: number; processing: number; processed: number; failed: number; total: number };
  routes: { total: number; active?: number; archived: number };
  config: { errors: string[]; warnings?: string[]; needs_deploy?: boolean; source?: string };
  service: { running: boolean; pid?: number | null; uptime_seconds?: number | null };
  operational_state?: "setup_incomplete" | "blocked" | "needs_apply" | "ui_only" | "ready" | string;
  processing_state?: "idle" | "queued" | "processing" | "attention" | string;
  next_step?: string | null;
}

export interface QueueEntry {
  path: string;
  status: "pending" | "processing" | "processed" | "failed" | string;
  route_id?: string | null;
  ingestion_root?: string | null;
  added_at?: string | null;
  processed_at?: string | null;
  failed_at?: string | null;
  last_error?: string | null;
}

export interface QueueStats {
  pending: number;
  processing: number;
  processed: number;
  failed: number;
  total: number;
}

export interface QueueEntriesResponse {
  entries: QueueEntry[];
  total: number;
  filters?: Record<string, unknown>;
}

export interface RecentWorkspace {
  path: string;
  name?: string;
  has_routes?: boolean;
  n_routes?: number;
  has_runtime_test?: boolean;
  has_runtime_live?: boolean;
  is_current?: boolean;
  exists?: boolean;
}

export interface RouteSpec {
  id: string;
  enabled: boolean;
  archived?: boolean;
  modes: string[];
  priority?: number;
  taskfile: string;
  montage: string;
  ingestion_folders: string[];
  file_globs: string[];
  recursive?: boolean;
  sentinel_ext?: string;
  output_dir?: string;
  output_folder?: string;
  version?: string | null;
  [key: string]: unknown;
}

export interface RouteFormData {
  id: string;
  enabled: boolean;
  modes: string[];
  ingestion_folders: string[];
  file_globs: string[];
  recursive: boolean;
  sentinel_ext?: string;
  taskfile: string;
  montage: string;
  priority: number;
  version?: string;
  [key: string]: unknown;
}

export interface TaskOption {
  name: string;
  path?: string;
  label?: string;
  description?: string;
}

export interface MontageOption {
  name: string;
  path?: string;
  label?: string;
  description?: string;
}

export interface ServiceStartSettings {
  routes?: string[];
  poll_interval?: number;
  workers?: number;
  open_browser?: boolean;
  [key: string]: unknown;
}

export interface ValidationResponse {
  valid: boolean;
  errors: string[];
  warnings: string[];
  message?: string;
}

export interface TunnelStatus {
  active: boolean;
  url?: string | null;
  mode?: "quick" | "named" | string;
  password?: string | null;
}

export interface FolderEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface BrowseResponse {
  path: string;
  parent?: string | null;
  entries: FolderEntry[];
}

export interface TutorialSuggestedRoute {
  route_id?: string;
  route_name?: string;
  workspace_dir?: string;
  output_dir?: string;
  sample_file?: string;
  id?: string;
  taskfile?: string;
  montage?: string;
  ingestion_folders?: string[];
  file_globs?: string[];
  modes?: string[];
  enabled?: boolean;
  recursive?: boolean;
  priority?: number;
}

export interface MontageInfo {
  name: string;
  category: string;
  n_channels: number;
  compatible_tasks: string[];
  description?: string;
}

export interface MontageDetail extends MontageInfo {
  channels: Array<{ name: string; x?: number; y?: number; z?: number; pos?: number[] }>;
  channel_names?: string[];
  preview_svg?: string;
  topomap_png?: string;
  landmarks?: Record<string, number[]>;
}

export interface MontageListResponse {
  montages: MontageInfo[];
  total: number;
}

export interface ManagedTask {
  name: string;
  display_name?: string;
  category: string;
  source: string;
  version?: string;
  installed?: boolean;
  update_available?: boolean;
  montage?: string;
  sample_rate?: number | null;
  ica_method?: string | null;
  description: string;
  sync_status: "installed" | "modified" | "not_installed" | "workspace_only";
  config?: Record<string, any>;
  pipeline: string[];
  source_code?: string;
  [key: string]: unknown;
}

export type TaskSyncStatus = "installed" | "modified" | "not_installed" | "workspace_only";

export interface TaskManagerResponse {
  tasks: ManagedTask[];
  sync?: Record<string, unknown>;
  registry_status?: {
    synced_at?: string;
    task_count?: number;
    [key: string]: unknown;
  };
}

export interface TaskActionResponse {
  success: boolean;
  message: string;
}

export interface RunSummary {
  run_id: string;
  filename: string;
  task: string;
  status: string;
  success?: boolean;
  created_at: string;
  output_dir?: string;
  decision?: string | null;
  notes?: string | null;
  route_id?: string | null;
}

export interface RunDetail {
  run_id: string;
  filename: string;
  task?: string;
  status?: string;
  success?: boolean;
  created_at?: string;
  route_id?: string | null;
  metrics: {
    channels_original: number;
    channels_retained: number;
    bad_channels: Array<{ channel: string; reason?: string }>;
    epochs_total: number | null;
    epochs_kept: number | null;
    ica_n_components: number | null;
    ica_removed: number[];
    ica_method?: string;
    duration_raw?: number | null;
    duration_post?: number | null;
    filter_low?: number | null;
    filter_high?: number | null;
    notch_freqs?: number[];
    sample_rate?: number | null;
  };
  assets: Record<string, string | null>;
  error?: string | null;
}

export interface ResultsListResponse {
  runs: RunSummary[];
  total: number;
}

export interface IcaComponent {
  component: string;
  type: string;
  confidence: number;
  rejected: boolean;
}

export interface IcaSummaryResponse {
  components: IcaComponent[];
  total_pages?: number;
  topo_grid_page?: number | null;
  structure: {
    topo_grid_page: number | null;
    detail_page_map: Record<string, number>;
    [key: string]: unknown;
  };
}

export interface EventsResponse {
  recording_type?: string;
  classification?: string;
  event_count: number;
  events_per_min?: number | null;
  filename?: string;
  n_channels?: number;
  sfreq?: number;
  duration_sec?: number | null;
  file_info?: {
    duration: number;
    filename: string;
    n_channels: number;
    sfreq: number;
  };
  metadata?: { duration?: number; filename?: string; n_channels?: number; sfreq?: number; [key: string]: unknown };
  event_types: Array<{
    event_id?: string | number;
    event_type?: string;
    label?: string;
    code?: string | number;
    count: number;
    percentage?: number;
    first_onset?: number | null;
    last_onset?: number | null;
    mean_isi?: number | null;
    median_isi?: number | null;
  }>;
  unique_type_count: number;
  transitions?: Array<{ from: string; to: string; count: number }>;
  inter_event_intervals?: { min?: number; max?: number; mean?: number; median?: number; std?: number; count?: number };
  isi_stats?: { min: number; max: number; mean: number; median: number; std: number; count: number };
  long_gaps?: Array<{ start: number; end: number; duration: number }>;
  burst_groups?: Array<{ start: number; end: number; count: number }>;
  events?: Array<Record<string, unknown>>;
  summary?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ExcludeFileSummary {
  file_key: string;
  relative_path: string;
  name: string;
  notes_present: boolean;
  epochs_reviewed: boolean;
  bad_epochs_count: number;
  has_overrides: boolean;
  status: string;
}

export interface ExcludeFileDetail {
  file_key: string;
  relative_path: string;
  name: string;
  exports_root: string;
  status: string;
  notes: string;
  metrics: Record<string, unknown>;
  baseline_bad_channels: string[];
  baseline_rejected_ica: number[];
  valid_channels: string[];
  max_components: number;
  manual_bad_channels: string[];
  manual_rejected_ica: number[];
  epoch_review: {
    epochs_reviewed: boolean;
    bad_epochs_count: number;
    bad_epoch_indices: number[];
    bad_epoch_times: string[];
    bad_epoch_events: string[];
    total_epochs: number;
    epoch_rejection_rate: number;
  };
  qa_export: {
    hash: string;
    timestamp: string;
    path: string;
  };
  reprocess: {
    modified: boolean;
    fix_type: string;
    timestamp: string;
  };
  artifacts: Record<string, string | null>;
}

export interface ExcludeFilesResponse {
  exports_root: string;
  files: ExcludeFileSummary[];
}

export interface EpochManifest {
  file_key: string;
  relative_path: string;
  mode: string;
  sampling_rate: number;
  channel_names: string[];
  n_channels: number;
  n_epochs: number;
  epoch_length_samples: number;
  epoch_duration_seconds: number;
  existing_bad_epoch_indices: number[];
  default_scaling_uv: number;
  visible_epoch_count: number;
}

export interface EpochWindowResponse {
  file_key: string;
  start_epoch: number;
  count: number;
  channel_names: string[];
  sampling_rate: number;
  epoch_duration_seconds: number;
  epochs: Array<{
    epoch_index: number;
    event_code?: string | null;
    start_time_seconds: number;
    is_bad: boolean;
    traces_uv: Record<string, number[]>;
  }>;
}

export interface ExcludeIcaSummaryResponse {
  components: Array<{
    component: string;
    type: string;
    confidence: number;
    rejected: boolean;
  }>;
  structure?: Record<string, unknown>;
}

export interface ExcludeEpochTopographyResponse {
  file_key: string;
  epoch_index: number;
  sample_index: number;
  latency_ms: number;
  image_png_base64: string;
  channels_used: string[];
}

export const api = {
  getHealth: () => json<Record<string, any>>("/health"),
  getStatus: () => json<DashboardStatus>("/api/status"),
  switchMode: (mode: "test" | "live") => json<Record<string, any>>("/api/mode/switch", "POST", { mode }),

  getQueueStats: () => json<QueueStats>("/api/queue/stats"),
  getQueueEntries: (routeId?: string) =>
    json<QueueEntriesResponse>(`/api/queue/entries${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`),
  retryFailed: () => json<{ retried: number }>("/api/queue/retry", "POST", {}),
  clearProcessed: () => json<{ cleared: number }>("/api/queue/processed", "DELETE"),
  removeEntry: (path: string) => json<{ cleared?: number }>(`/api/queue/entry/${encodeURIComponent(path)}`, "DELETE"),

  getRecentWorkspaces: () => json<{ workspaces: RecentWorkspace[]; current?: string | null }>("/api/workspaces/recent"),
  setupWorkspace: (path: string, createNew: boolean) => json<Record<string, any>>("/api/setup/workspace", "POST", { path, create_new: createNew }),

  getRoutes: () => json<RouteSpec[]>("/api/routes"),
  getTaskOptions: () => json<TaskOption[]>("/api/routes/discovery/tasks"),
  getMontageOptions: () => json<MontageOption[]>("/api/routes/discovery/montages"),
  createRoute: (body: RouteFormData) => json<Record<string, any>>("/api/routes", "POST", body),
  deleteRoute: (id: string) => json<Record<string, any>>(`/api/routes/${encodeURIComponent(id)}`, "DELETE"),
  promoteRoute: (id: string) => json<Record<string, any>>(`/api/routes/${encodeURIComponent(id)}/promote`, "POST", {}),
  archiveRoute: (id: string) => json<Record<string, any>>(`/api/routes/${encodeURIComponent(id)}/archive`, "POST", {}),
  unarchiveRoute: (id: string) => json<Record<string, any>>(`/api/routes/${encodeURIComponent(id)}/unarchive`, "POST", {}),
  enableRoute: (id: string) => json<Record<string, any>>(`/api/routes/${encodeURIComponent(id)}/enable`, "POST", {}),
  disableRoute: (id: string) => json<Record<string, any>>(`/api/routes/${encodeURIComponent(id)}/disable`, "POST", {}),
  syncRoutes: () => json<Record<string, any>>("/api/routes/sync", "POST", {}),

  getConfig: () => json<Record<string, any>>("/api/config"),
  getConfigYaml: () => json<{ content: string }>("/api/config/yaml"),
  validateConfig: () => json<ValidationResponse>("/api/config/validate", "POST", {}),
  deployConfig: () => json<{ success: boolean; message: string }>("/api/config/deploy", "POST", {}),

  getServiceStatus: () => json<Record<string, any>>("/api/service/status"),
  getServiceLogs: () => json<{ lines: string[]; total: number }>("/api/service/logs"),
  startService: (settings: ServiceStartSettings) => json<Record<string, any>>("/api/service/start", "POST", settings),
  stopService: () => json<Record<string, any>>("/api/service/stop", "POST", {}),

  getTunnelStatus: () => json<TunnelStatus>("/api/tunnel/status"),
  startTunnel: () => json<Record<string, any>>("/api/tunnel/start", "POST", {}),
  stopTunnel: () => json<Record<string, any>>("/api/tunnel/stop", "POST", {}),
  getTunnelConfig: () => json<Record<string, any>>("/api/tunnel/config"),
  setTunnelConfig: (token: string, url: string) => json<Record<string, any>>("/api/tunnel/config", "PUT", { token, url }),
  clearTunnelConfig: () => json<Record<string, any>>("/api/tunnel/config", "DELETE"),

  browseFolders: (path?: string) => json<BrowseResponse>(`/api/filesystem/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`),

  getTaskManager: () => json<TaskManagerResponse>("/api/task-manager"),
  installTask: (name: string) => json<TaskActionResponse>("/api/task-manager/install", "POST", { name }),
  updateTask: (name: string) => json<TaskActionResponse>(`/api/task-manager/${encodeURIComponent(name)}/update`, "POST", {}),
  removeTask: (name: string) => json<TaskActionResponse>(`/api/task-manager/${encodeURIComponent(name)}`, "DELETE"),
  createTask: (name: string) => json<TaskActionResponse>("/api/task-manager/create", "POST", { name }),
  refreshLibrary: () => json<TaskActionResponse>("/api/task-manager/refresh-library", "POST", {}),

  getMontages: () => json<MontageListResponse>("/api/montages"),
  getMontageDetail: (name: string) => json<MontageDetail>(`/api/montages/${encodeURIComponent(name)}`),
  getResults: (routeId?: string) =>
    json<ResultsListResponse>(`/api/results${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`),
  getRun: (runId: string) => json<RunDetail>(`/api/results/${encodeURIComponent(runId)}`),
  getRunDetail: (runId: string) => json<RunDetail>(`/api/results/${encodeURIComponent(runId)}`),
  getRunReportUrl: (runId: string) => `/api/results/${encodeURIComponent(runId)}/report`,
  getRunPsdUrl: (runId: string) => `/api/results/${encodeURIComponent(runId)}/psd`,
  getRunOverlayUrl: (runId: string) => `/api/results/${encodeURIComponent(runId)}/overlay`,
  getRunMetadata: (runId: string) => json<Record<string, unknown>>(`/api/results/${encodeURIComponent(runId)}/metadata`),
  getRunChannels: (runId: string) => request<string>(`/api/results/${encodeURIComponent(runId)}/channels`),
  getRunEvents: (runId: string) => json<EventsResponse>(`/api/results/${encodeURIComponent(runId)}/events`),
  getIcaSummary: (runId: string) => json<IcaSummaryResponse>(`/api/results/${encodeURIComponent(runId)}/ica/summary`),
  getIcaPageUrl: (runId: string, pageNum: number) => `/api/results/${encodeURIComponent(runId)}/ica/page/${pageNum}`,
  getRunDownloadUrl: (runId: string) => `/api/results/${encodeURIComponent(runId)}/download`,
  getResultsCsvUrl: () => "/api/results/export/csv",
  getDecisions: () => json<{ decisions: Record<string, { decision: string; notes: string }> }>("/api/results/decisions"),
  setDecision: (runId: string, decision: string, notes = "") => json<Record<string, any>>(`/api/results/${encodeURIComponent(runId)}/decision`, "PUT", { decision, notes }),

  analyzeEvents: (filePath: string) => json<EventsResponse>("/api/events/analyze", "POST", { file_path: filePath }),

  tutorialSetup: () => json<{ success?: boolean; sample_file?: string; suggested_route?: TutorialSuggestedRoute }>("/api/tutorial/setup", "POST", {}),
  tutorialCleanup: () => json<Record<string, any>>("/api/tutorial/cleanup", "POST", {}),

  getExcludeRoot: (routeId?: string) =>
    json<{ exports_root: string }>(`/api/exclude/root${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`),
  getExcludeFiles: (routeId?: string) =>
    json<ExcludeFilesResponse>(`/api/exclude/files${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`),
  getExcludeFile: (fileKey: string, routeId?: string) =>
    json<ExcludeFileDetail>(`/api/exclude/files/${fileKey}${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`),
  getExcludeIcaSummary: (fileKey: string, routeId?: string) =>
    json<ExcludeIcaSummaryResponse>(`/api/exclude/files/${fileKey}/ica-summary${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`),
  getExcludeEpochManifest: (fileKey: string, routeId?: string) =>
    json<EpochManifest>(`/api/exclude/files/${fileKey}/eeg/manifest${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`),
  getExcludeEpochWindow: (fileKey: string, start = 0, count = 10, channels?: string[], routeId?: string) => {
    const params = new URLSearchParams({ start: String(start), count: String(count) });
    if (channels?.length) params.set("channels", channels.join(","));
    if (routeId) params.set("route_id", routeId);
    return json<EpochWindowResponse>(`/api/exclude/files/${fileKey}/eeg/epochs?${params.toString()}`);
  },
  getExcludeEpochTopography: (fileKey: string, epochIndex: number, sampleIndex: number, routeId?: string) =>
    json<ExcludeEpochTopographyResponse>(
      `/api/exclude/files/${fileKey}/eeg/topography?epoch_index=${epochIndex}&sample_index=${sampleIndex}${routeId ? `&route_id=${encodeURIComponent(routeId)}` : ""}`,
    ),
  getExcludeEpochReview: (fileKey: string, routeId?: string) =>
    json<Record<string, any>>(`/api/exclude/files/${fileKey}/epoch-review${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`),
  saveExcludeEpochReview: (fileKey: string, badEpochIndices: number[], routeId?: string) =>
    json<Record<string, any>>(`/api/exclude/files/${fileKey}/epoch-review${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`, "PUT", { bad_epoch_indices: badEpochIndices }),
  saveExcludeNotes: (fileKey: string, notes: string, status?: string, routeId?: string) =>
    json<Record<string, any>>(`/api/exclude/files/${fileKey}/notes${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`, "PUT", { notes, status }),
  saveExcludeOverrides: (fileKey: string, manualBadChannels: string[], manualRejectedIca: number[], routeId?: string) =>
    json<Record<string, any>>(`/api/exclude/files/${fileKey}/overrides${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`, "PUT", {
      manual_bad_channels: manualBadChannels,
      manual_rejected_ica: manualRejectedIca,
    }),
  startExcludeReprocess: (fileKey: string, manualBadChannels: string[], manualRejectedIca: number[], routeId?: string) =>
    json<{ job_id: string; status: string; message: string }>(`/api/exclude/files/${fileKey}/reprocess${routeId ? `?route_id=${encodeURIComponent(routeId)}` : ""}`, "POST", {
      manual_bad_channels: manualBadChannels,
      manual_rejected_ica: manualRejectedIca,
    }),
  getExcludeReprocessStatus: (jobId: string) => json<Record<string, any>>(`/api/exclude/reprocess/${jobId}`),
  exportExcludeQa: (fileKeys?: string[]) =>
    json<{ exported: number; skipped: number; errors: Array<{ file_key: string; error: string }>; qa_log_path?: string | null }>(
      "/api/exclude/qa/export",
      "POST",
      { file_keys: fileKeys ?? [] },
    ),
  getExcludeQaLogUrl: () => "/api/exclude/qa/preprocessing-log",
};
