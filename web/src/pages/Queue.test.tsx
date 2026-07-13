import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getQueueEntries: vi.fn(),
    getQueueStats: vi.fn(),
    getRoutes: vi.fn(),
    getStatus: vi.fn(),
    retryFailed: vi.fn(),
    clearProcessed: vi.fn(),
    removeEntry: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api }));
vi.mock("../contexts/TutorialContext", () => ({
  useTutorial: () => ({ isActive: false, currentStep: 0, nextStep: vi.fn() }),
}));
vi.mock("../hooks/useTutorialTarget", () => ({
  useTutorialTarget: () => null,
}));

import Queue from "./Queue";

const selectedPath = "C:\\EEG data\\selected file.set";
const otherPath = "C:\\EEG data\\other.set";

function renderQueue() {
  return render(
    <MemoryRouter>
      <Queue />
    </MemoryRouter>,
  );
}

describe("Queue retry scope", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getQueueEntries.mockResolvedValue({
      entries: [
        {
          path: selectedPath,
          status: "failed",
          route_id: "route-a",
          last_error: "failed",
        },
        {
          path: otherPath,
          status: "failed",
          route_id: "route-b",
          last_error: "failed",
        },
      ],
      total: 2,
    });
    api.getQueueStats.mockResolvedValue({
      pending: 0,
      processing: 0,
      processed: 0,
      failed: 2,
      total: 2,
    });
    api.getRoutes.mockResolvedValue([]);
    api.getStatus.mockResolvedValue({ service: { running: false } });
    api.retryFailed.mockResolvedValue({ retried: 1 });
  });

  it("retries only the selected failed entry from the detail panel", async () => {
    renderQueue();

    fireEvent.click(await screen.findByText("selected file.set"));
    const retryButtons = screen.getAllByRole("button", { name: "Retry Failed" });
    expect(retryButtons).toHaveLength(2);

    fireEvent.click(retryButtons[1]!);

    await waitFor(() => {
      expect(api.retryFailed).toHaveBeenCalledWith([selectedPath]);
    });
    expect(api.retryFailed).not.toHaveBeenCalledWith([otherPath]);
  });

  it("keeps the page-level retry action unscoped", async () => {
    renderQueue();

    const retryButton = await screen.findByRole("button", { name: "Retry Failed" });
    fireEvent.click(retryButton);

    await waitFor(() => {
      expect(api.retryFailed).toHaveBeenCalledWith(undefined);
    });
  });
});
