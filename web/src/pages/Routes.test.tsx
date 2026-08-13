import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getRoutes: vi.fn(),
    getTaskOptions: vi.fn(),
    getMontageOptions: vi.fn(),
    createRoute: vi.fn(),
    deleteRoute: vi.fn(),
    syncRoutes: vi.fn(),
    promoteRoute: vi.fn(),
    archiveRoute: vi.fn(),
    unarchiveRoute: vi.fn(),
    enableRoute: vi.fn(),
    disableRoute: vi.fn(),
    scanRouteMontageReview: vi.fn(),
    applyRouteMontageReview: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api }));
vi.mock("../contexts/TutorialContext", () => ({
  useTutorial: () => ({
    isActive: false,
    currentStep: 0,
    tutorialData: null,
    nextStep: vi.fn(),
  }),
}));
vi.mock("../hooks/useTutorialTarget", () => ({
  useTutorialTarget: () => null,
}));

import RoutesPage from "./Routes";

const baseRoute = {
  id: "route-1",
  enabled: true,
  archived: false,
  modes: ["test"],
  priority: 100,
  taskfile: "RestingEyesOpen",
  montage: "GSN-HydroCel-129",
  ingestion_folders: ["/input"],
  file_globs: ["*.set"],
  recursive: true,
};

function renderPage() {
  return render(
    <MemoryRouter>
      <RoutesPage />
    </MemoryRouter>,
  );
}

describe("Routes page actions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getRoutes.mockResolvedValue([{ ...baseRoute }]);
    api.getTaskOptions.mockResolvedValue([]);
    api.getMontageOptions.mockResolvedValue([]);
    api.syncRoutes.mockResolvedValue({ success: true });
    api.deleteRoute.mockResolvedValue({ success: true });
    api.promoteRoute.mockResolvedValue({ success: true });
    api.archiveRoute.mockResolvedValue({ success: true });
    api.unarchiveRoute.mockResolvedValue({ success: true });
    api.enableRoute.mockResolvedValue({ success: true });
    api.disableRoute.mockResolvedValue({ success: true });
    const reviewResponse = {
      route_id: "route-1",
      mode: "test",
      workspace_dir: "/workspace",
      taskfile: "RestingEyesOpen",
      task_path: "/workspace/tasks/RestingEyesOpen.py",
      configured_route_montage: "GSN-HydroCel-129",
      expected_task_montage: "GSN-HydroCel-129",
      input_paths: ["/input"],
      split_output_root: "/workspace/montage-preflight/test/route-1",
      groups: [
        {
          detected_montage: "GSN-HydroCel-128",
          status: "mismatch",
          file_count: 2,
          total_size_bytes: 4096,
          examples: ["sub-001.set"],
          supported: true,
          suggested_route_id: "route-1-gsn-hydrocel-128",
          suggested_taskfile: "tasks/RestingEyesOpen_GSN_HydroCel_128.py",
          suggested_workspace_name: "RestingEyesOpen_GSN-HydroCel-128",
          suggested_ingestion_folder: "/workspace/montage-preflight/test/route-1/GSN-HydroCel-128",
        },
        {
          detected_montage: "unknown",
          status: "unknown",
          file_count: 1,
          total_size_bytes: 100,
          examples: ["notes.txt"],
          supported: false,
          suggested_route_id: null,
          suggested_taskfile: null,
          suggested_workspace_name: null,
          suggested_ingestion_folder: null,
        },
      ],
      files: [],
      unknown_files: ["/input/notes.txt"],
      copy_estimate: {
        split_output_root: "/workspace/montage-preflight/test/route-1",
        actionable_file_count: 2,
        skipped_file_count: 1,
        required_bytes: 4096,
        free_bytes_before: 100000,
        free_bytes_after_estimate: 95904,
      },
      can_apply: true,
      warnings: ["1 file(s) are unknown or unsupported and will not be routed."],
    };
    api.scanRouteMontageReview.mockResolvedValue(reviewResponse);
    api.applyRouteMontageReview.mockResolvedValue({
      success: true,
      message: "Applied",
      review: reviewResponse,
      copied_files: [{ destination: "/workspace/montage-preflight/test/route-1/GSN-HydroCel-128/sub-001.set" }],
      skipped_files: ["/input/notes.txt"],
      enqueued: 1,
      updated_queue_entries: 1,
      route_actions: [],
      cloned_tasks: [],
      warnings: [],
    });
  });

  async function openActionMenu(routeId = "route-1") {
    const button = await screen.findByRole("button", {
      name: `Open actions for route ${routeId}`,
    });
    fireEvent.click(button);
  }

  it("requires confirmation for promote and then shows Apply guidance", async () => {
    renderPage();

    await openActionMenu();
    fireEvent.click(await screen.findByRole("button", { name: "Go Live" }));

    expect(
      await screen.findByText("Enable 'route-1' for Live processing?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Apply the latest config in Settings before live processing changes take effect/i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Enable Live" }));

    await waitFor(() => {
      expect(api.promoteRoute).toHaveBeenCalledWith("route-1");
      expect(api.syncRoutes).toHaveBeenCalled();
    });

    expect(
      screen.getByText("Route 'route-1' promoted to Live. Open Settings and click Apply to publish this change for processing."),
    ).toBeInTheDocument();
  });

  it("requires confirmation for archive and explains Apply semantics", async () => {
    renderPage();

    await openActionMenu();
    fireEvent.click(await screen.findByRole("button", { name: "Archive" }));

    expect(
      await screen.findByText("Archive route 'route-1'?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Apply the latest config in Settings before processing stops watching its input folders/i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Archive" }));

    await waitFor(() => {
      expect(api.archiveRoute).toHaveBeenCalledWith("route-1");
      expect(api.syncRoutes).toHaveBeenCalled();
    });

    expect(
      screen.getByText("Route 'route-1' archived. Open Settings and click Apply to publish this change for processing."),
    ).toBeInTheDocument();
  });

  it("requires confirmation for delete and explains Apply semantics", async () => {
    renderPage();

    await openActionMenu();
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));

    expect(
      await screen.findByText("Delete route 'route-1'?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Apply the latest config in Settings before processing stops using this route/i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete Route" }));

    await waitFor(() => {
      expect(api.deleteRoute).toHaveBeenCalledWith("route-1");
      expect(api.syncRoutes).toHaveBeenCalled();
    });

    expect(
      screen.getByText("Route 'route-1' deleted. Open Settings and click Apply to publish this change for processing."),
    ).toBeInTheDocument();
  });

  it("runs immediate disable action and still requires Apply afterward", async () => {
    renderPage();

    await openActionMenu();
    fireEvent.click(await screen.findByRole("button", { name: "Disable" }));

    await waitFor(() => {
      expect(api.disableRoute).toHaveBeenCalledWith("route-1");
      expect(api.syncRoutes).toHaveBeenCalled();
    });

    expect(
      screen.getByText("Route 'route-1' disabled. Open Settings and click Apply to publish this change for processing."),
    ).toBeInTheDocument();
  });

  it("scans and confirms montage review copy mode", async () => {
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", {
        name: "Review montage preflight for route route-1",
      }),
    );

    await waitFor(() => {
      expect(api.scanRouteMontageReview).toHaveBeenCalledWith("route-1");
    });
    expect(await screen.findByText("Montage Review: route-1")).toBeInTheDocument();
    expect(screen.getByText("route-1-gsn-hydrocel-128")).toBeInTheDocument();
    expect(screen.getAllByText("not routed").length).toBeGreaterThan(0);

    fireEvent.click(screen.getAllByRole("button", { name: "Apply Copy" })[0]!);
    expect(await screen.findByText("Apply montage review copy?")).toBeInTheDocument();
    const applyButtons = screen.getAllByRole("button", { name: "Apply Copy" });
    fireEvent.click(applyButtons[applyButtons.length - 1]!);

    await waitFor(() => {
      expect(api.applyRouteMontageReview).toHaveBeenCalledWith("route-1", {
        confirm: true,
        mode: "copy",
      });
    });
    expect(
      screen.getByText("Montage review copied 1 file(s), queued 1, and skipped 1."),
    ).toBeInTheDocument();
  });

  it("shows action errors instead of success notice when an action fails", async () => {
    api.disableRoute.mockRejectedValue(new Error("API 500: disable failed"));
    renderPage();

    await openActionMenu();
    fireEvent.click(await screen.findByRole("button", { name: "Disable" }));

    expect(
      await screen.findByText("API 500: disable failed"),
    ).toBeInTheDocument();
    expect(api.syncRoutes).not.toHaveBeenCalled();
    expect(
      screen.queryByText(/Open Settings and click Apply to publish this change for processing/i),
    ).not.toBeInTheDocument();
  });

  it("allows typing a custom montage even when montage suggestions are available", async () => {
    api.getRoutes.mockResolvedValue([]);
    api.getTaskOptions.mockResolvedValue([{ name: "RestingEyesOpen", source: "/tasks/RestingEyesOpen.py", description: "" }]);
    api.getMontageOptions.mockResolvedValue([
      { name: "GSN-HydroCel-129", description: "EGI 129-channel net" },
      { name: "standard_1020", description: "Standard 10-20" },
    ]);
    api.createRoute.mockResolvedValue({ success: true });

    renderPage();

    fireEvent.click(await screen.findAllByRole("button", { name: "New Route" }).then((buttons) => buttons[0]!));

    const montageInput = await screen.findByRole("combobox", { name: "Montage" });
    fireEvent.focus(montageInput);
    expect(await screen.findByRole("listbox")).toBeInTheDocument();
    fireEvent.mouseDown(await screen.findByRole("option", { name: /GSN-HydroCel-129/i }));
    expect((montageInput as HTMLInputElement).value).toBe("GSN-HydroCel-129");

    fireEvent.change(await screen.findByLabelText("Route ID"), {
      target: { value: "custom-montage-route" },
    });
    fireEvent.change(screen.getByLabelText("Task"), {
      target: { value: "RestingEyesOpen" },
    });
    fireEvent.change(montageInput, {
      target: { value: "MyCustomCap-64" },
    });
    fireEvent.change(screen.getByPlaceholderText("/path/to/folder — press Enter to add"), {
      target: { value: "/input" },
    });
    fireEvent.keyDown(screen.getByPlaceholderText("/path/to/folder — press Enter to add"), {
      key: "Enter",
      code: "Enter",
    });

    fireEvent.click(screen.getByRole("button", { name: "Create Route" }));

    await waitFor(() => {
      expect(api.createRoute).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "custom-montage-route",
          taskfile: "RestingEyesOpen",
          montage: "MyCustomCap-64",
        }),
      );
    });
  });
});
