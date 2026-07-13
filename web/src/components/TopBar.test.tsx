import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getHealth: vi.fn(),
    getTunnelStatus: vi.fn(),
    startTunnel: vi.fn(),
    stopTunnel: vi.fn(),
    getTunnelConfig: vi.fn(),
    setTunnelConfig: vi.fn(),
    clearTunnelConfig: vi.fn(),
    switchMode: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api }));
vi.mock("../hooks/usePolling", () => ({
  usePolling: (fetcher: unknown) => ({
    data:
      fetcher === api.getHealth
        ? { mode: "test" }
        : { active: false, url: null, password: null, mode: null },
    error: null,
    loading: false,
    refresh: vi.fn(),
  }),
}));
vi.mock("../contexts/ThemeContext", () => ({
  useTheme: () => ({ theme: "dark", toggle: vi.fn() }),
}));

import TopBar from "./TopBar";

async function openConfig() {
  render(
    <MemoryRouter>
      <TopBar />
    </MemoryRouter>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Share" }));
  await screen.findByText("Share Publicly");
  fireEvent.click(screen.getByRole("button", { name: "Configure permanent tunnel" }));
  await screen.findByText("Named Tunnel Setup");
}

describe("TopBar tunnel configuration errors", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.startTunnel.mockResolvedValue({ success: true });
    api.getTunnelConfig.mockResolvedValue({ url: "" });
  });

  it("shows a retryable error when saving tunnel configuration fails", async () => {
    api.setTunnelConfig.mockRejectedValue(new Error("Token was rejected"));
    await openConfig();

    fireEvent.change(screen.getByPlaceholderText("eyJhIjoi..."), {
      target: { value: "bad-token" },
    });
    fireEvent.change(screen.getByPlaceholderText("https://eeg-lab.example.com"), {
      target: { value: "https://eeg.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(
      await screen.findByText("Could not save tunnel configuration. Check the values and try again."),
    ).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Could not save tunnel configuration");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
    });
    expect(api.setTunnelConfig).toHaveBeenCalledWith(
      "bad-token",
      "https://eeg.example.com",
    );
  });

  it("shows a retryable error when clearing tunnel configuration fails", async () => {
    api.clearTunnelConfig.mockRejectedValue(new Error("Configuration could not be cleared"));
    await openConfig();

    fireEvent.click(screen.getByRole("button", { name: "Clear" }));

    expect(
      await screen.findByText("Could not clear tunnel configuration. Try again."),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Clear" })).toBeEnabled();
    });
    expect(api.clearTunnelConfig).toHaveBeenCalledTimes(1);
  });
});
