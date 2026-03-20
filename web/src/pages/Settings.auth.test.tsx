import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getConfigYaml: vi.fn(),
    getHealth: vi.fn(),
    validateConfig: vi.fn(),
    deployConfig: vi.fn(),
    getAuthConfig: vi.fn(),
    saveAuthConfig: vi.fn(),
    getAdminUsers: vi.fn(),
    setUserRoles: vi.fn(),
    getNotificationsConfig: vi.fn(),
    saveNotificationsConfig: vi.fn(),
    sendTestEmail: vi.fn(),
    sendDailyDigest: vi.fn(),
  },
}));

const { usePolling } = vi.hoisted(() => ({
  usePolling: vi.fn(),
}));

const { useAuth } = vi.hoisted(() => ({
  useAuth: vi.fn(),
}));

vi.mock("../lib/api", async () => {
  const actual = await vi.importActual<typeof import("../lib/api")>("../lib/api");
  return { ...actual, api };
});
vi.mock("../hooks/usePolling", () => ({ usePolling }));
vi.mock("../hooks/useAuth", () => ({ useAuth }));
vi.mock("../contexts/TutorialContext", () => ({
  useTutorial: () => ({ isActive: false, currentStep: 0, nextStep: vi.fn() }),
}));
vi.mock("../hooks/useTutorialTarget", () => ({
  useTutorialTarget: () => ({ current: null }),
}));

import Settings from "./Settings";

describe("Settings auth and notifications", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getAuthConfig.mockResolvedValue({
      mode: "oauth",
      provider: "github",
      allow_disable_auth: true,
      session: { cookie_name: "autoclean_session", ttl_hours: 12, secure: null },
      github: {
        client_id: "client-id",
        client_secret: "***",
        redirect_uri: "http://localhost:8000/api/auth/callback/github",
        allowed_orgs: [],
        allowed_users: [],
      },
      bootstrap_admins: [],
    });
    api.getNotificationsConfig.mockResolvedValue({
      enabled: true,
      provider: "resend",
      app_base_url: "http://localhost:8000",
      resend: {
        api_key: "***",
        has_api_key: true,
        sender_email: "lab@example.edu",
        sender_name: "Lab Alerts",
        reply_to: "",
      },
      recipients: {
        service: ["ops@example.edu"],
        config_failure: [],
        queue_failure: [],
        job_failure: [],
      },
      route_recipients: {},
      daily_digest_recipients: ["digest@example.edu"],
      providers: {
        resend: { configured: true, selected: true },
      },
      cooldown_minutes: {
        service: 15,
        config_failure: 30,
        queue_failure: 60,
        job_failure: 60,
      },
    });
    api.getAdminUsers.mockResolvedValue({
      users: [
        {
          id: "github:1",
          login: "admin-user",
          provider: "github",
          roles: ["admin"],
          last_login_at: "2026-03-20T00:00:00+00:00",
          disabled: false,
        },
      ],
    });
    api.sendTestEmail.mockResolvedValue({ success: true, message_id: "email_123" });
    api.saveNotificationsConfig.mockResolvedValue({
      success: true,
      config: {
        enabled: true,
        provider: "resend",
        app_base_url: "http://localhost:8000",
        resend: {
          api_key: "***",
          has_api_key: true,
          sender_email: "lab@example.edu",
          sender_name: "Lab Alerts",
          reply_to: "",
        },
        recipients: {
          service: ["ops@example.edu"],
          config_failure: [],
          queue_failure: [],
          job_failure: [],
        },
        route_recipients: {},
        daily_digest_recipients: ["digest@example.edu"],
        providers: {
          resend: { configured: true, selected: true },
        },
        cooldown_minutes: {
          service: 15,
          config_failure: 30,
          queue_failure: 60,
          job_failure: 60,
        },
      },
    });
    api.sendDailyDigest.mockResolvedValue({ success: true, message_id: "digest_123" });
    usePolling.mockImplementation((fn: unknown) => {
      if (fn === api.getConfigYaml) {
        return {
          data: { content: "mode: test\nruntime: runtimes/test\n" },
          error: null,
          loading: false,
          refresh: vi.fn(),
        };
      }
      if (fn === api.getHealth) {
        return { data: { mode: "test" }, error: null, loading: false, refresh: vi.fn() };
      }
      return { data: null, error: null, loading: false, refresh: vi.fn() };
    });
  });

  it("shows the apply permission warning and disables Apply for non-editors", async () => {
    useAuth.mockReturnValue({
      hasPermission: (permission: string) => permission === "config.read",
    });

    render(<Settings />);

    expect(await screen.findByText(/Editor permission is required to apply configuration changes/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply" })).toBeDisabled();
  });

  it("shows auth policy lock when disabling auth is not allowed", async () => {
    api.getAuthConfig.mockResolvedValueOnce({
      mode: "oauth",
      provider: "github",
      allow_disable_auth: false,
      session: { cookie_name: "autoclean_session", ttl_hours: 12, secure: null },
      github: {
        client_id: "client-id",
        client_secret: "***",
        redirect_uri: "http://localhost:8000/api/auth/callback/github",
        allowed_orgs: [],
        allowed_users: [],
      },
      oidc: {
        issuer_url: "",
        client_id: "",
        client_secret: "",
        redirect_uri: "http://localhost:8000/api/auth/callback/oidc",
        scopes: ["openid", "profile", "email"],
        allowed_groups: [],
        allowed_users: [],
        username_claim: "preferred_username",
        groups_claim: "groups",
      },
      bootstrap_admins: [],
    });
    useAuth.mockReturnValue({
      hasPermission: () => true,
    });

    render(<Settings />);

    expect(await screen.findByText(/keep auth enabled/i)).toBeInTheDocument();
    const disabledOptions = screen.getAllByRole("option", { name: "Disabled" });
    expect(disabledOptions[0]).toBeDisabled();
  });

  it("allows admins to send a test email from settings", async () => {
    useAuth.mockReturnValue({
      hasPermission: () => true,
    });

    render(<Settings />);

    fireEvent.change(await screen.findByPlaceholderText("scientist@example.edu, lab@example.edu"), {
      target: { value: "scientist@example.edu" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send Test Email" }));

    await waitFor(() => {
      expect(api.sendTestEmail).toHaveBeenCalledWith(
        ["scientist@example.edu"],
        "AutoClean Serve test email",
        "This is a test email from AutoClean Serve notifications.",
      );
    });
  });

  it("saves notification settings for admins", async () => {
    useAuth.mockReturnValue({
      hasPermission: () => true,
    });

    render(<Settings />);
    fireEvent.click(await screen.findByRole("button", { name: "Save Notifications" }));

    await waitFor(() => {
      expect(api.saveNotificationsConfig).toHaveBeenCalled();
    });
  });

  it("can trigger a daily digest from settings", async () => {
    useAuth.mockReturnValue({
      hasPermission: () => true,
    });

    render(<Settings />);
    fireEvent.click(await screen.findByRole("button", { name: "Send Daily Digest" }));

    await waitFor(() => {
      expect(api.sendDailyDigest).toHaveBeenCalled();
    });
  });
});
