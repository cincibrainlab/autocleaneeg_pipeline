import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
  getExcludeFiles: vi.fn(),
  getStatus: vi.fn(),
  getExcludeFile: vi.fn(),
  getExcludeEpochManifest: vi.fn(),
  getExcludeEpochWindow: vi.fn(),
  getExcludeIcaSummary: vi.fn(),
  saveExcludeEpochReview: vi.fn(),
  saveExcludeNotes: vi.fn(),
  saveExcludeOverrides: vi.fn(),
  startExcludeReprocess: vi.fn(),
  getExcludeReprocessStatus: vi.fn(),
  exportExcludeQa: vi.fn(),
  getExcludeQaLogUrl: vi.fn(() => "/api/exclude/qa/preprocessing-log"),
  },
}));

vi.mock("../lib/api", () => ({ api }));

import ExcludePage from "./Exclude";

const files = {
  exports_root: "/workspace/task/exports",
  files: [
    {
      file_key: "subject01_comp_epo",
      relative_path: "subject01_comp_epo.set",
      name: "subject01_comp_epo.set",
      notes_present: true,
      epochs_reviewed: true,
      bad_epochs_count: 1,
      has_overrides: true,
      status: "REVIEW",
    },
    {
      file_key: "subject02_comp_epo",
      relative_path: "subject02_comp_epo.set",
      name: "subject02_comp_epo.set",
      notes_present: false,
      epochs_reviewed: false,
      bad_epochs_count: 0,
      has_overrides: false,
      status: "UNSET",
    },
  ],
};

function detail(fileKey: string) {
  return {
    file_key: fileKey,
    relative_path: `${fileKey}.set`,
    name: `${fileKey}.set`,
    exports_root: files.exports_root,
    status: "REVIEW",
    notes: `notes for ${fileKey}`,
    metrics: { data_retained: "100.0s", channels_retained: "62", channels_original: "64" },
    baseline_bad_channels: ["FP1"],
    baseline_rejected_ica: [1],
    valid_channels: ["FP1", "FP2"],
    max_components: 12,
    manual_bad_channels: ["FP1"],
    manual_rejected_ica: [1, 3],
    epoch_review: {
      epochs_reviewed: true,
      bad_epochs_count: 1,
      bad_epoch_indices: [0],
      bad_epoch_times: ["0.000"],
      bad_epoch_events: ["101"],
      total_epochs: 6,
      epoch_rejection_rate: 16.7,
    },
    qa_export: { hash: "", timestamp: "", path: "" },
    reprocess: { modified: true, fix_type: "both", timestamp: "2026-03-16 10:00:00" },
    artifacts: { run_report: null, ica_report: null, psd: null, metadata: null, postedit: null },
  };
}

beforeEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
  vi.spyOn(window, "confirm").mockReturnValue(true);
  api.getExcludeFiles.mockResolvedValue(files);
  api.getStatus.mockResolvedValue({ workspace_dir: "/workspace" });
  api.getExcludeFile.mockImplementation(async (fileKey: string) => detail(fileKey));
  api.getExcludeEpochManifest.mockResolvedValue({
    file_key: "subject01_comp_epo",
    relative_path: "subject01_comp_epo.set",
    mode: "epochs",
    sampling_rate: 250,
    channel_names: ["FP1", "FP2"],
    n_channels: 2,
    n_epochs: 6,
    epoch_length_samples: 100,
    epoch_duration_seconds: 1,
    existing_bad_epoch_indices: [0],
    default_scaling_uv: 25,
    visible_epoch_count: 6,
  });
  api.getExcludeEpochWindow.mockResolvedValue({
    file_key: "subject01_comp_epo",
    start_epoch: 0,
    count: 1,
    channel_names: ["FP1", "FP2"],
    sampling_rate: 250,
    epoch_duration_seconds: 1,
    epochs: [
      {
        epoch_index: 0,
        event_code: "101",
        start_time_seconds: 0,
        is_bad: true,
        traces_uv: { FP1: [0, 1, 0], FP2: [0, -1, 0] },
      },
    ],
  });
  api.getExcludeIcaSummary.mockResolvedValue({ components: [], structure: {} });
  api.saveExcludeEpochReview.mockResolvedValue({ saved: true });
  api.saveExcludeNotes.mockResolvedValue({ saved: true });
  api.saveExcludeOverrides.mockResolvedValue({ saved: true });
  api.startExcludeReprocess.mockResolvedValue({ job_id: "job-1", status: "running", message: "started" });
  api.getExcludeReprocessStatus.mockResolvedValue({ status: "completed", message: "done", running: false });
  api.exportExcludeQa.mockResolvedValue({ exported: 1, skipped: 0, errors: [], qa_log_path: "/workspace/task/qa/qa_preprocessing_log.csv" });
});

describe("ExcludePage", () => {
  it("renders workspace, exports root, and file list", async () => {
    render(<ExcludePage />);

    await screen.findAllByText("subject01_comp_epo.set");
    expect(screen.getByText("/workspace")).toBeInTheDocument();
    expect(screen.getByText("/workspace/task/exports")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show Files" }));
    expect(screen.getByText("note")).toBeInTheDocument();
  });

  it("loads detail for a newly selected file", async () => {
    render(<ExcludePage />);

    await screen.findAllByText("subject01_comp_epo.set");
    fireEvent.click(screen.getByRole("button", { name: "Show Files" }));
    fireEvent.click(screen.getAllByText("subject02_comp_epo.set")[0]!);

    await waitFor(() => {
      expect(api.getExcludeFile).toHaveBeenCalledWith("subject02_comp_epo");
    });
  });

  it("toggles an epoch with Space and autosaves it", async () => {
    render(<ExcludePage />);

    await screen.findByText("Epoch 1");
    fireEvent.keyDown(document, { key: " " });

    await waitFor(() => {
      expect(api.saveExcludeEpochReview).toHaveBeenCalled();
    }, { timeout: 1500 });
  });

  it("persists notes edits", async () => {
    render(<ExcludePage />);

    const textarea = await screen.findByPlaceholderText("Add reviewer notes...");
    fireEvent.change(textarea, { target: { value: "updated note" } });

    await waitFor(() => {
      expect(api.saveExcludeNotes).toHaveBeenCalledWith("subject01_comp_epo", "updated note", "UNSET");
    }, { timeout: 1500 });
  });

  it("adds and removes manual overrides, then saves them", async () => {
    render(<ExcludePage />);

    await screen.findByRole("button", { name: /FP1 ×/ });
    const channelInput = await screen.findByPlaceholderText("e.g. Fp1");
    fireEvent.change(channelInput, { target: { value: "Fp2" } });
    fireEvent.click(screen.getAllByText("Add")[0]!);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /FP2 ×/ })).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: /FP2 ×/ }));

    const icaInput = screen.getByPlaceholderText("e.g. 3");
    fireEvent.change(icaInput, { target: { value: "5" } });
    fireEvent.click(screen.getAllByText("Add")[1]!);
    expect(await screen.findByText("IC 5 ×")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Save Overrides"));

    await waitFor(() => {
      expect(api.saveExcludeOverrides).toHaveBeenCalledWith("subject01_comp_epo", ["FP1"], [1, 3, 5]);
    });
  });

  it("shows reprocess status updates", async () => {
    api.getExcludeReprocessStatus.mockResolvedValue({ status: "completed", message: "done", running: false });

    render(<ExcludePage />);

    await screen.findByRole("button", { name: /FP1 ×/ });
    fireEvent.click(screen.getByText("Reprocess with Overrides"));

    await waitFor(() => {
      expect(api.startExcludeReprocess).toHaveBeenCalledWith("subject01_comp_epo", ["FP1"], [1, 3]);
    });

    await waitFor(() => {
      expect(screen.getByText("Reprocess: completed")).toBeInTheDocument();
      expect(screen.getByText("done")).toBeInTheDocument();
    }, { timeout: 4000 });
  });

  it("runs the main review workflow in one page session", async () => {
    render(<ExcludePage />);

    await screen.findByRole("button", { name: /FP1 ×/ });

    fireEvent.keyDown(document, { key: " " });
    await waitFor(() => {
      expect(api.saveExcludeEpochReview).toHaveBeenCalled();
    }, { timeout: 1500 });

    const textarea = screen.getByPlaceholderText("Add reviewer notes...");
    fireEvent.change(textarea, { target: { value: "workflow note" } });
    await waitFor(() => {
      expect(api.saveExcludeNotes).toHaveBeenCalledWith("subject01_comp_epo", "workflow note", "REVIEW");
    }, { timeout: 1500 });

    fireEvent.click(screen.getByText("Save Overrides"));
    await waitFor(() => {
      expect(api.saveExcludeOverrides).toHaveBeenCalledWith("subject01_comp_epo", ["FP1"], [1, 3]);
    });

    fireEvent.click(screen.getByText("Export QA File"));
    await waitFor(() => {
      expect(api.exportExcludeQa).toHaveBeenCalledWith(["subject01_comp_epo"]);
    });
  });
});
