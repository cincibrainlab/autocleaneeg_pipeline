import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getServiceStatus: vi.fn(),
    getStatus: vi.fn(),
    getServiceLogs: vi.fn(),
    startService: vi.fn(),
    stopService: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api }));
vi.mock("../contexts/TutorialContext", () => ({
  useTutorial: () => ({
    isActive: false,
    currentStep: 0,
    nextStep: vi.fn(),
  }),
}));
vi.mock("../hooks/useTutorialTarget", () => ({
  useTutorialTarget: () => null,
}));

import Service from "./Service";

describe("Service page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getServiceLogs.mockResolvedValue({ lines: [], total: 0 });
  });

  it("disables start and shows apply guidance when config is not deployed", async () => {
    api.getServiceStatus.mockResolvedValue({
      running: false,
      mode: "test",
      can_start: false,
      blocked_reason: "Apply the latest configuration changes before starting the service.",
    });
    api.getStatus.mockResolvedValue({
      configured: true,
      mode: "test",
      queue: { pending: 0, processing: 0, processed: 0, failed: 0, total: 0 },
      routes: { total: 1, active: 1, archived: 0 },
      config: { errors: [], needs_deploy: true, source: "operator" },
      service: { running: false },
      operational_state: "needs_apply",
      next_step: "Apply the latest configuration changes.",
    });

    render(
      <MemoryRouter>
        <Service />
      </MemoryRouter>,
    );

    expect(
      await screen.findAllByText("Apply the latest configuration changes before starting the service."),
    ).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Start Service" })).toBeDisabled();
  });
});
