import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getWorkspaceUtilities: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api }));

import Utilities from "./Utilities";

describe("Utilities page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows workspace status details and doctor guidance", async () => {
    api.getWorkspaceUtilities.mockResolvedValue({
      configured: true,
      selected_workspace_path: "/tmp/workspace",
      bootstrapped_from_autoclean: true,
      workspace_details: {
        serve_test_exists: true,
        serve_live_exists: true,
        deploy_exists: true,
        runtimes_test_exists: true,
        runtimes_live_exists: true,
        test_runtime_ready: true,
        live_runtime_ready: false,
      },
      status_checks: [
        { label: "serve-test.yaml", ok: true, detail: "/tmp/workspace/serve-test.yaml" },
        { label: "live runtime ready", ok: false, detail: "/tmp/workspace/runtimes/live/.venv" },
      ],
      doctor: {
        ok: false,
        summary: "Found 1 blocking issue(s)",
        blocking_issues: [
          { label: "live runtime ready", detail: "/tmp/workspace/runtimes/live/.venv" },
        ],
        guidance: [
          "Re-run 'autocleaneeg-pipeline serve workspace --mode existing --path <dir>' to rebuild runtimes.",
        ],
      },
    });

    render(
      <MemoryRouter>
        <Utilities />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Workspace Status")).toBeInTheDocument();
    expect(screen.getByText("/tmp/workspace")).toBeInTheDocument();
    expect(screen.getByText("Bootstrapped from an AutoClean workspace")).toBeInTheDocument();
    expect(screen.getAllByText("serve-test.yaml").length).toBeGreaterThan(0);
    expect(screen.getAllByText("live runtime ready").length).toBeGreaterThan(0);
    expect(screen.getByText("Found 1 blocking issue(s)")).toBeInTheDocument();
    expect(
      screen.getByText(/serve workspace --mode existing --path <dir>/i),
    ).toBeInTheDocument();
  });
});
