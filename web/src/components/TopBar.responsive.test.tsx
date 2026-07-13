import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getHealth: vi.fn(),
    getTunnelStatus: vi.fn(),
    startTunnel: vi.fn(),
    stopTunnel: vi.fn(),
    switchMode: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api }));
vi.mock("../hooks/usePolling", () => ({
  usePolling: (fetcher: unknown) => ({
    data:
      fetcher === api.getHealth
        ? { mode: "test", status: "healthy" }
        : { active: false },
    error: null,
    loading: false,
    refresh: vi.fn(),
  }),
}));
vi.mock("../contexts/ThemeContext", () => ({
  useTheme: () => ({ theme: "dark", toggle: vi.fn() }),
}));

import TopBar from "./TopBar";

function renderTopBar() {
  return render(
    <MemoryRouter>
      <TopBar />
    </MemoryRouter>,
  );
}

describe("TopBar responsive layout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.startTunnel.mockResolvedValue({ success: true });
  });

  it("keeps every essential mobile control reachable in the two-row grid", () => {
    const { container } = renderTopBar();
    const header = container.querySelector("header");
    const title = screen.getByRole("heading", { name: "Dashboard" });

    expect(header).toHaveClass("grid", "grid-cols-[minmax(0,1fr)_auto]", "md:flex");
    expect(title).toHaveClass("truncate");
    expect(screen.getByRole("button", { name: "Toggle sidebar" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Switch to light mode" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Share" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Test" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Live" })).toBeVisible();
    expect(screen.getByText("Healthy")).toBeVisible();
  });

  it("contains the Share popover within mobile gutters and preserves desktop anchoring", async () => {
    const { container } = renderTopBar();

    fireEvent.click(screen.getByRole("button", { name: "Share" }));
    await screen.findByText("Share Publicly");

    const popover = container.querySelector(".fixed.inset-x-3");
    expect(popover).toHaveClass(
      "top-24",
      "w-auto",
      "sm:absolute",
      "sm:right-0",
      "sm:w-80",
    );
  });
});
