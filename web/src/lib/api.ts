// ---- Type definitions ----

export interface HealthResponse {
  status: string;
  workspace_configured: boolean;
  mode: string;
  pipeline_version: string;
}

export interface ServiceStatus {
  running: boolean;
  pid: number | null;
  mode: string;
  uptime_seconds: number | null;
}

export interface RouteSpec {
  id: string;
  modes: string[];
  enabled: boolean;
  archived: boolean;
  priority: number;
  taskfile: string;
  montage: string;
  ingestion_folders: string[];
  file_globs: string[];
  recursive: boolean;
  output_folder: string;
  [key: string]: unknown;
}

export interface RouteFormData {
  id: string;
  taskfile: string;
  montage: string;
  ingestion_folders: string[];
  file_globs: string[];
  modes: string[];
  enabled: boolean;
  recursive: boolean;
  priority: number;
}

export interface QueueStats {
  pending: number;
  processing: number;
  processed: number;
  failed: number;
  total: number;
}

export interface QueueEntry {
  path: string;
  status: string;
  route_id: string | null;
  ingestion_root: string | null;
  added_at: string | null;
  processed_at: string | null;
  failed_at: string | null;
  last_error: string | null;
}

export interface QueueEntriesResponse {
  entries: QueueEntry[];
  total: number;
}

export interface ConfigYamlResponse {
  content: string;
}

export interface ValidationResponse {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface DeployResponse {
  success: boolean;
  message: string;
}

export interface ActionResponse {
  success: boolean;
  message: string;
}

export interface ServiceLogsResponse {
  lines: string[];
  total: number;
}

export interface ServiceStartSettings {
  max_cycles?: number;
  idle_limit?: number;
  sleep_seconds?: number;
  no_watch?: boolean;
  no_sentinel?: boolean;
}

export interface RetryResponse {
  retried: number;
}

export interface ClearResponse {
  cleared: number;
}

export interface TaskOption {
  name: string;
  source: string;
  description: string;
}

export interface MontageOption {
  name: string;
  description: string;
}

export interface TunnelStatus {
  active: boolean;
  url: string | null;
  password: string | null;
  mode: string | null;
}

export interface TunnelStartResponse {
  success: boolean;
  url: string | null;
  password: string | null;
  message: string;
  mode: string | null;
}

export interface TunnelConfig {
  configured: boolean;
  url: string;
  has_token: boolean;
}

export interface TaskDetailConfig {
  montage: string;
  sample_rate: number | null;
  filter_low: number | null;
  filter_high: number | null;
  notch_freqs: number[];
  ica_method: string;
  ica_threshold: number | null;
  epoch_tmin: number | null;
  epoch_tmax: number | null;
  event_id: Record<string, number> | null;
}

export interface TaskDetail {
  name: string;
  description: string;
  source: string;
  category: string;
  config: TaskDetailConfig;
  pipeline: string[];
  source_code: string;
}

export interface TutorialSuggestedRoute {
  id: string;
  taskfile: string;
  montage: string;
  ingestion_folders: string[];
  file_globs: string[];
  modes: string[];
  enabled: boolean;
  recursive: boolean;
  priority: number;
}

export interface TutorialSetupResponse {
  success: boolean;
  sample_file: string;
  suggested_route: TutorialSuggestedRoute;
}

export interface TutorialCleanupResponse {
  success: boolean;
  message: string;
}

export interface FolderEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface BrowseResponse {
  path: string;
  parent: string | null;
  entries: FolderEntry[];
}

export type TaskSyncStatus = "installed" | "modified" | "not_installed" | "workspace_only";

export interface ManagedTask {
  name: string;
  description: string;
  category: string;
  source: string;
  sync_status: TaskSyncStatus;
  workspace_path: string | null;
  config: TaskDetailConfig | null;
  pipeline: string[];
  source_code: string;
}

export interface RegistryInfo {
  commit: string | null;
  synced_at: string | null;
  task_count: number;
}

export interface TaskManagerResponse {
  tasks: ManagedTask[];
  registry_status: RegistryInfo;
  workspace_dir: string;
}

export interface TaskActionResponse {
  success: boolean;
  message: string;
  task_name: string;
  path: string | null;
}

export interface MontageInfo {
  name: string;
  n_channels: number;
  category: string;
  description: string;
  channel_names: string[];
  compatible_tasks: string[];
}

export interface MontageListResponse {
  montages: MontageInfo[];
  total: number;
}

export interface ChannelPosition {
  name: string;
  x: number;
  y: number;
  z: number;
}

export interface MontageDetail {
  name: string;
  n_channels: number;
  category: string;
  description: string;
  channels: ChannelPosition[];
  topomap_png: string;
  compatible_tasks: string[];
  landmarks: Record<string, number[]>;
}

export interface RunSummary {
  run_id: string;
  created_at: string;
  task: string;
  filename: string;
  status: string;
  success: boolean;
  automation_dir: string;
}

export interface ProcessingMetrics {
  channels_original: number;
  channels_retained: number;
  bad_channels: Array<{ channel: string; reason: string }>;
  epochs_total: number | null;
  epochs_kept: number | null;
  ica_n_components: number | null;
  ica_removed: number[];
  ica_method: string;
  duration_raw: number | null;
  duration_post: number | null;
  filter_low: number | null;
  filter_high: number | null;
  notch_freqs: number[];
  sample_rate: number | null;
}

export interface AssetAvailability {
  report: boolean;
  ica_report: boolean;
  psd: boolean;
  overlay: boolean;
  metadata: boolean;
  channels: boolean;
}

export interface RunDetail {
  run_id: string;
  created_at: string;
  task: string;
  filename: string;
  status: string;
  success: boolean;
  error: string | null;
  metrics: ProcessingMetrics;
  assets: AssetAvailability;
  user_context: Record<string, unknown> | null;
}

export interface EventType {
  label: string;
  code: number;
  count: number;
  first_onset: number | null;
  last_onset: number | null;
  mean_isi: number | null;
  median_isi: number | null;
}

export interface IsiStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  count: number;
}

export interface EventsResponse {
  has_events: boolean;
  event_count: number;
  event_types: EventType[];
  unique_type_count: number;
  timeline?: Array<{ onset: number; duration: number; trial_type: string; value: string }>;
  isi_stats: IsiStats | null;
  recording_type: string;
  epoch_info?: Record<string, unknown>;
  long_gaps: Array<{ start: number; end: number; duration: number }>;
  transitions: Array<{ from: string; to: string; count: number }>;
  duration_sec: number | null;
  events_per_min: number | null;
  file_info?: {
    filename: string;
    n_channels: number;
    sfreq: number;
    duration: number;
  };
}

export interface IcaComponent {
  component: string;
  type: string;
  confidence: number;
  rejected: boolean;
}

export interface IcaStructure {
  total_pages: number;
  summary_pages: number[];
  topo_grid_page: number | null;
  detail_start_page: number | null;
  n_detail_pages: number;
  detail_page_map: Record<string, number>;
}

export interface IcaSummaryResponse {
  components: IcaComponent[];
  structure: IcaStructure;
}

export interface DashboardStatus {
  configured: boolean;
  mode: string;
  workspace_dir: string;
  output_dir: string;
  routes: {
    total: number;
    active: number;
    archived: number;
  };
  queue: QueueStats;
  config: {
    valid: boolean;
    errors: string[];
    needs_deploy: boolean;
    source: string;
  };
  service: ServiceStatus;
}

export interface RecentWorkspace {
  path: string;
  name: string;
  has_routes: boolean;
  n_routes: number;
  has_runtime_test: boolean;
  has_runtime_live: boolean;
  is_current: boolean;
}

export interface RecentWorkspacesResponse {
  workspaces: RecentWorkspace[];
  current: string | null;
}

// ---- Fetch helper ----

const BASE = "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {};
  if (options?.body) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { ...headers, ...(options?.headers as Record<string, string>) },
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }

  return res.json() as Promise<T>;
}

// ---- API client ----

export const api = {
  // Dashboard
  getStatus: () => request<DashboardStatus>("/api/status"),

  // Routes
  getRoutes: () => request<RouteSpec[]>("/api/routes"),
  createRoute: (data: RouteFormData) =>
    request<ActionResponse>("/api/routes", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  deleteRoute: (id: string) =>
    request<ActionResponse>(`/api/routes/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  promoteRoute: (id: string) =>
    request<ActionResponse>(`/api/routes/${encodeURIComponent(id)}/promote`, {
      method: "POST",
    }),
  archiveRoute: (id: string) =>
    request<ActionResponse>(`/api/routes/${encodeURIComponent(id)}/archive`, {
      method: "POST",
    }),
  unarchiveRoute: (id: string) =>
    request<ActionResponse>(`/api/routes/${encodeURIComponent(id)}/unarchive`, {
      method: "POST",
    }),
  enableRoute: (id: string) =>
    request<ActionResponse>(`/api/routes/${encodeURIComponent(id)}/enable`, {
      method: "POST",
    }),
  disableRoute: (id: string) =>
    request<ActionResponse>(`/api/routes/${encodeURIComponent(id)}/disable`, {
      method: "POST",
    }),
  syncRoutes: () =>
    request<ActionResponse>("/api/routes/sync", { method: "POST" }),

  // Route discovery
  getTaskOptions: () => request<TaskOption[]>("/api/routes/discovery/tasks"),
  getMontageOptions: () =>
    request<MontageOption[]>("/api/routes/discovery/montages"),

  // Queue
  getQueueStats: () => request<QueueStats>("/api/queue/stats"),
  getQueueEntries: (params?: { limit?: number; offset?: number }) => {
    const qs = params
      ? "?" + new URLSearchParams(
          Object.entries(params).map(([k, v]) => [k, String(v)])
        ).toString()
      : "";
    return request<QueueEntriesResponse>(`/api/queue/entries${qs}`);
  },
  retryFailed: () =>
    request<RetryResponse>("/api/queue/retry", { method: "POST", body: JSON.stringify({}) }),
  clearProcessed: () =>
    request<ClearResponse>("/api/queue/processed", { method: "DELETE" }),
  removeEntry: (path: string) =>
    request<ClearResponse>(`/api/queue/entry/${encodeURIComponent(path)}`, {
      method: "DELETE",
    }),

  // Config
  getConfigYaml: () => request<ConfigYamlResponse>("/api/config/yaml"),
  validateConfig: () =>
    request<ValidationResponse>("/api/config/validate", { method: "POST" }),
  deployConfig: () =>
    request<DeployResponse>("/api/config/deploy", { method: "POST" }),

  // Service
  getServiceStatus: () => request<ServiceStatus>("/api/service/status"),
  startService: (settings?: ServiceStartSettings) =>
    request<ActionResponse>("/api/service/start", {
      method: "POST",
      body: JSON.stringify(settings ?? {}),
    }),
  stopService: () =>
    request<ActionResponse>("/api/service/stop", { method: "POST" }),
  getServiceLogs: () =>
    request<ServiceLogsResponse>("/api/service/logs"),

  // Tunnel
  getTunnelStatus: () => request<TunnelStatus>("/api/tunnel/status"),
  startTunnel: () =>
    request<TunnelStartResponse>("/api/tunnel/start", { method: "POST" }),
  stopTunnel: () =>
    request<ActionResponse>("/api/tunnel/stop", { method: "POST" }),
  getTunnelConfig: () => request<TunnelConfig>("/api/tunnel/config"),
  setTunnelConfig: (token: string, url: string) =>
    request<ActionResponse>("/api/tunnel/config", {
      method: "PUT",
      body: JSON.stringify({ token, url }),
    }),
  clearTunnelConfig: () =>
    request<ActionResponse>("/api/tunnel/config", { method: "DELETE" }),

  // Tutorial
  tutorialSetup: () =>
    request<TutorialSetupResponse>("/api/tutorial/setup", { method: "POST" }),
  tutorialCleanup: () =>
    request<TutorialCleanupResponse>("/api/tutorial/cleanup", { method: "POST" }),

  // Filesystem browser
  browseFolders: (path?: string) =>
    request<BrowseResponse>(
      `/api/filesystem/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`
    ),

  // Task browser
  getTasks: () => request<TaskDetail[]>("/api/tasks"),
  getTask: (name: string) => request<TaskDetail>(`/api/tasks/${encodeURIComponent(name)}`),

  // Task manager
  getTaskManager: () => request<TaskManagerResponse>("/api/task-manager"),
  installTask: (name: string) =>
    request<TaskActionResponse>("/api/task-manager/install", {
      method: "POST",
      body: JSON.stringify({ task_name: name }),
    }),
  createTask: (className: string) =>
    request<TaskActionResponse>("/api/task-manager/create", {
      method: "POST",
      body: JSON.stringify({ class_name: className }),
    }),
  refreshLibrary: () =>
    request<TaskActionResponse>("/api/task-manager/refresh-library", {
      method: "POST",
    }),
  updateTask: (name: string) =>
    request<TaskActionResponse>(`/api/task-manager/${encodeURIComponent(name)}/update`, {
      method: "POST",
    }),
  removeTask: (name: string) =>
    request<TaskActionResponse>(`/api/task-manager/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),

  // Montage browser
  getMontages: () => request<MontageListResponse>("/api/montages"),
  getMontageDetail: (name: string) =>
    request<MontageDetail>(`/api/montages/${encodeURIComponent(name)}`),

  // Results viewer
  getResults: () =>
    request<{ runs: RunSummary[]; total: number }>("/api/results"),
  getRunDetail: (runId: string) =>
    request<RunDetail>(`/api/results/${encodeURIComponent(runId)}`),
  getRunMetadata: (runId: string) =>
    request<Record<string, unknown>>(`/api/results/${encodeURIComponent(runId)}/metadata`),
  getIcaSummary: (runId: string) =>
    request<IcaSummaryResponse>(`/api/results/${encodeURIComponent(runId)}/ica/summary`),
  getRunEvents: (runId: string) =>
    request<EventsResponse>(`/api/results/${encodeURIComponent(runId)}/events`),

  // Downloads
  getResultsCsvUrl: () => `${BASE}/api/results/export/csv`,
  getRunDownloadUrl: (runId: string) =>
    `${BASE}/api/results/${encodeURIComponent(runId)}/download`,
  getRunReportUrl: (runId: string) =>
    `${BASE}/api/results/${encodeURIComponent(runId)}/report`,
  getRunIcaReportUrl: (runId: string) =>
    `${BASE}/api/results/${encodeURIComponent(runId)}/ica-report`,
  getRunPsdUrl: (runId: string) =>
    `${BASE}/api/results/${encodeURIComponent(runId)}/psd`,
  getRunOverlayUrl: (runId: string) =>
    `${BASE}/api/results/${encodeURIComponent(runId)}/overlay`,

  // Decisions
  getDecisions: () =>
    request<{
      decisions: Record<string, { run_id: string; decision: string; notes: string; decided_at: string; filename: string }>;
      total: number;
    }>("/api/results/decisions"),
  setDecision: (runId: string, decision: "pass" | "fail" | "review" | "clear", notes: string = "") =>
    request<{ success: boolean; run_id: string; decision: string }>(
      `/api/results/${encodeURIComponent(runId)}/decision`,
      {
        method: "PUT",
        body: JSON.stringify({ decision, notes }),
      }
    ),
  getDecisionsCsvUrl: () => `${BASE}/api/results/decisions/export/csv`,

  // Event analyzer
  analyzeEvents: (filePath: string) =>
    request<EventsResponse>("/api/events/analyze", {
      method: "POST",
      body: JSON.stringify({ file_path: filePath }),
    }),

  // Mode
  switchMode: (mode: "test" | "live") =>
    request<{ success: boolean; mode: string; message: string }>("/api/mode/switch", {
      method: "POST",
      body: JSON.stringify({ mode }),
    }),

  // Health
  getHealth: () => request<HealthResponse>("/health"),

  // Workspace picker
  setupWorkspace: (path: string, createNew = false) =>
    request<{ success: boolean; workspace_dir: string; message: string }>(
      "/api/setup/workspace",
      {
        method: "POST",
        body: JSON.stringify({ path, create_new: createNew }),
      }
    ),
  getRecentWorkspaces: () =>
    request<RecentWorkspacesResponse>("/api/workspaces/recent"),
};
