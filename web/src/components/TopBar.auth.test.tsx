import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getHealth: vi.fn(),
    getTunnelStatus: vi.fn(),
    logout: vi.fn(),
  },
}));

const { usePolling } = vi.hoisted(() => ({
  usePolling: vi.fn(),
}));

const { useAuth } = vi.hoisted(() => ({
  useAuth: vi.fn(),
}));

const { useTheme } = vi.hoisted(() => ({
  useTheme: vi.fn(),
}));

vi.mock("../lib/api", async () => {
  const actual = await vi.importActual<typeof import("../lib/api")>("../lib/api");
  return { ...actual, api };
});
vi.mock("../hooks/usePolling", () => ({ usePolling }));
vi.mock("../hooks/useAuth", () => ({ useAuth }));
vi.mock("../contexts/ThemeContext", () => ({ useTheme }));

import TopBar from "./TopBar";

describe("TopBar auth state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useTheme.mockReturnValue({ theme: "dark", toggle: vi.fn() });
    usePolling.mockImplementation((fn: unknown) => {
      if (fn === api.getHealth) return { data: { mode: "test" }, refresh: vi.fn() };
      return { data: { active: false }, refresh: vi.fn() };
    });
  });

  it("shows the auth-disabled warning and blocks share for non-admins", () => {
    useAuth.mockReturnValue({
      authStatus: { mode: "disabled", enabled: false },
      me: null,
      refresh: vi.fn(),
      hasPermission: () => false,
    });

    render(
      <MemoryRouter initialEntries={["/service"]}>
        <TopBar />
      </MemoryRouter>,
    );

    expect(screen.getByText("Auth Off")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Share/i })).toBeDisabled();
  });
});
