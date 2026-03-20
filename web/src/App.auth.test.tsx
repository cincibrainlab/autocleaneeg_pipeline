import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getAuthStatus: vi.fn(),
    getMe: vi.fn(),
    login: vi.fn(),
  },
}));

vi.mock("./lib/api", async () => {
  const actual = await vi.importActual<typeof import("./lib/api")>("./lib/api");
  return { ...actual, api };
});

vi.mock("./pages/Dashboard", () => ({ default: () => <div>Dashboard</div> }));
vi.mock("./pages/Routes", () => ({ default: () => <div>Routes</div> }));
vi.mock("./pages/Queue", () => ({ default: () => <div>Queue</div> }));
vi.mock("./pages/Service", () => ({ default: () => <div>Service</div> }));
vi.mock("./pages/Settings", () => ({ default: () => <div>Settings</div> }));
vi.mock("./pages/Tasks", () => ({ default: () => <div>Tasks</div> }));
vi.mock("./pages/Montages", () => ({ default: () => <div>Montages</div> }));
vi.mock("./pages/Results", () => ({ default: () => <div>Results</div> }));
vi.mock("./pages/EventAnalyzer", () => ({ default: () => <div>Events</div> }));
vi.mock("./pages/Setup", () => ({ default: () => <div>Setup</div> }));
vi.mock("./pages/Exclude", () => ({ default: () => <div>Exclude</div> }));
vi.mock("./components/Layout", () => ({ default: () => <div>Layout</div> }));
vi.mock("./components/tutorial/TutorialOverlay", () => ({ default: () => null }));

import App from "./App";

describe("App auth gate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows the login gate when auth is enabled and the user is signed out", async () => {
    api.getAuthStatus.mockResolvedValue({
      enabled: true,
      mode: "oauth",
      configured: true,
      authenticated: false,
      provider: "github",
      providers: { github: { configured: true, selected: true } },
    });

    render(<App />);

    expect(await screen.findByText("Sign in to AutoClean Serve")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue with GITHUB" })).toBeInTheDocument();
  });

  it("shows the unconfigured message when auth is enabled without provider config", async () => {
    api.getAuthStatus.mockResolvedValue({
      enabled: true,
      mode: "oauth",
      configured: false,
      authenticated: false,
      provider: "github",
      bootstrap_allowed: false,
      providers: { github: { configured: false, selected: true }, oidc: { configured: false, selected: false } },
    });

    render(<App />);

    expect(await screen.findByText(/configure at least one auth provider/)).toBeInTheDocument();
  });

  it("starts login when the GitHub button is clicked", async () => {
    api.getAuthStatus.mockResolvedValue({
      enabled: true,
      mode: "oauth",
      configured: true,
      authenticated: false,
      provider: "github",
      providers: { github: { configured: true, selected: true } },
    });
    api.login.mockResolvedValue({ login_url: "http://example.test/login" });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Continue with GITHUB" }));

    await waitFor(() => {
      expect(api.login).toHaveBeenCalledWith("github");
    });
  });

  it("renders bootstrap settings flow when auth is enabled but bootstrap is allowed locally", async () => {
    api.getAuthStatus.mockResolvedValue({
      enabled: true,
      mode: "oauth",
      configured: false,
      authenticated: false,
      provider: "oidc",
      bootstrap_allowed: true,
      providers: { github: { configured: false, selected: false }, oidc: { configured: false, selected: true } },
    });

    render(<App />);

    expect(await screen.findByText("Settings")).toBeInTheDocument();
  });

  it("renders login buttons for every configured provider", async () => {
    api.getAuthStatus.mockResolvedValue({
      enabled: true,
      mode: "oauth",
      configured: true,
      authenticated: false,
      provider: "github",
      providers: { github: { configured: true, selected: true }, oidc: { configured: true, selected: false } },
    });

    render(<App />);

    expect(await screen.findByRole("button", { name: "Continue with GITHUB" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue with OIDC" })).toBeInTheDocument();
  });
});
