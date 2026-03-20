import { useState, useEffect, useRef } from "react";
import { CheckCircle2, AlertTriangle, AlertCircle, RefreshCw, Upload, ChevronDown, ChevronRight, FileText, X } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { api } from "../lib/api";
import type { AdminUsersResponse, AuthConfigResponse, NotificationsConfigResponse, ValidationResponse } from "../lib/api";
import ErrorBanner from "../components/ErrorBanner";
import CodeViewer from "../components/CodeViewer";
import { useTutorial } from "../contexts/TutorialContext";
import { useTutorialTarget } from "../hooks/useTutorialTarget";
import { useAuth } from "../hooks/useAuth";

export default function Settings() {
  const { hasPermission } = useAuth();
  const canDeployConfig = hasPermission("config.deploy");
  const {
    data: configData,
    error: configError,
    loading: configLoading,
    refresh: refreshConfig,
  } = usePolling(api.getConfigYaml, 30000);
  const { data: health } = usePolling(api.getHealth, 30000);

  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [validating, setValidating] = useState(false);
  const [deploying, setDeploying] = useState(false);
  const [notice, setNotice] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [errorsOpen, setErrorsOpen] = useState(true);
  const [warningsOpen, setWarningsOpen] = useState(true);
  const [authConfig, setAuthConfig] = useState<AuthConfigResponse | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authSaving, setAuthSaving] = useState(false);
  const [notificationsConfig, setNotificationsConfig] = useState<NotificationsConfigResponse | null>(null);
  const [notificationsLoading, setNotificationsLoading] = useState(true);
  const [notificationsSaving, setNotificationsSaving] = useState(false);
  const [testEmailTo, setTestEmailTo] = useState("");
  const [testEmailSending, setTestEmailSending] = useState(false);
  const [dailyDigestSending, setDailyDigestSending] = useState(false);
  const [routeRecipientsRouteId, setRouteRecipientsRouteId] = useState("");
  const [routeRecipientsValue, setRouteRecipientsValue] = useState("");
  const [adminUsers, setAdminUsers] = useState<AdminUsersResponse["users"]>([]);
  const [usersLoading, setUsersLoading] = useState(true);
  const [roleSaving, setRoleSaving] = useState<string | null>(null);

  // Tutorial integration
  const { isActive, currentStep, nextStep } = useTutorial();
  const applyButtonRef = useTutorialTarget("apply-button");

  useEffect(() => {
    handleValidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let active = true;
    async function loadAuthConfig() {
      try {
        const result = await api.getAuthConfig();
        if (active) setAuthConfig(result);
      } catch {
        if (active) setAuthConfig(null);
      } finally {
        if (active) setAuthLoading(false);
      }
    }
    loadAuthConfig();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    async function loadNotificationsConfig() {
      try {
        const result = await api.getNotificationsConfig();
        if (active) setNotificationsConfig(result);
      } catch {
        if (active) setNotificationsConfig(null);
      } finally {
        if (active) setNotificationsLoading(false);
      }
    }
    loadNotificationsConfig();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    async function loadUsers() {
      if (!hasPermission("users.admin")) {
        if (active) {
          setAdminUsers([]);
          setUsersLoading(false);
        }
        return;
      }
      try {
        const result = await api.getAdminUsers();
        if (active) setAdminUsers(result.users);
      } catch {
        if (active) setAdminUsers([]);
      } finally {
        if (active) setUsersLoading(false);
      }
    }
    loadUsers();
    return () => {
      active = false;
    };
  }, [hasPermission]);

  // Clear notice timer on unmount
  useEffect(() => {
    return () => {
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    };
  }, []);

  const handleValidate = async () => {
    setValidating(true);
    try {
      const res = await api.validateConfig();
      setValidation(res);
    } catch (err) {
      setValidation({
        valid: false,
        errors: [err instanceof Error ? err.message : String(err)],
        warnings: [],
      });
    } finally {
      setValidating(false);
    }
  };

  const handleDeploy = async () => {
    setDeploying(true);
    setNotice(null);
    try {
      const res = await api.deployConfig();
      if (res.success) {
        setNotice({ type: "success", text: res.message || "Configuration applied successfully" });
        // Advance tutorial from apply-config (step 4) to start-service (step 5)
        if (isActive && currentStep === 4) {
          nextStep();
        }
      } else {
        setNotice({ type: "error", text: res.message || "Deploy failed" });
      }
      handleValidate();
      refreshConfig();
    } catch (err) {
      setNotice({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setDeploying(false);
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = setTimeout(() => setNotice(null), 6000);
    }
  };

  const handleSaveAuthConfig = async () => {
    if (!authConfig) return;
    setAuthSaving(true);
    setNotice(null);
    try {
      const result = await api.saveAuthConfig(authConfig);
      if (result.success) {
        setNotice({ type: "success", text: "Authentication settings saved" });
        const refreshed = await api.getAuthConfig();
        setAuthConfig(refreshed);
      } else {
        setNotice({ type: "error", text: "Failed to save authentication settings" });
      }
    } catch (err) {
      setNotice({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setAuthSaving(false);
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = setTimeout(() => setNotice(null), 6000);
    }
  };

  const toggleRole = async (userId: string, currentRoles: string[], role: string) => {
    setRoleSaving(`${userId}:${role}`);
    setNotice(null);
    const nextRoles = currentRoles.includes(role)
      ? currentRoles.filter((existingRole) => existingRole !== role)
      : [...currentRoles, role];
    try {
      await api.setUserRoles(userId, nextRoles);
      const result = await api.getAdminUsers();
      setAdminUsers(result.users);
      setNotice({ type: "success", text: "User roles updated" });
    } catch (err) {
      setNotice({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setRoleSaving(null);
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = setTimeout(() => setNotice(null), 6000);
    }
  };

  const handleSaveNotificationsConfig = async () => {
    if (!notificationsConfig) return;
    setNotificationsSaving(true);
    setNotice(null);
    try {
      const result = await api.saveNotificationsConfig(notificationsConfig);
      setNotificationsConfig(result.config);
      setNotice({ type: "success", text: "Notification settings saved" });
    } catch (err) {
      setNotice({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setNotificationsSaving(false);
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = setTimeout(() => setNotice(null), 6000);
    }
  };

  const handleSendTestEmail = async () => {
    const recipients = testEmailTo.split(",").map((value) => value.trim()).filter(Boolean);
    if (recipients.length === 0) {
      setNotice({ type: "error", text: "Add at least one recipient for the test email." });
      return;
    }
    setTestEmailSending(true);
    setNotice(null);
    try {
      await api.sendTestEmail(recipients, "AutoClean Serve test email", "This is a test email from AutoClean Serve notifications.");
      setNotice({ type: "success", text: "Test email sent" });
    } catch (err) {
      setNotice({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setTestEmailSending(false);
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = setTimeout(() => setNotice(null), 6000);
    }
  };

  const handleSendDailyDigest = async () => {
    setDailyDigestSending(true);
    setNotice(null);
    try {
      await api.sendDailyDigest();
      setNotice({ type: "success", text: "Daily digest sent" });
    } catch (err) {
      setNotice({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setDailyDigestSending(false);
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = setTimeout(() => setNotice(null), 6000);
    }
  };

  const status = (() => {
    if (!validation) return "checking";
    if (validation.errors.length > 0) return "errors";
    if (validation.warnings.length > 0) return "warnings";
    return "valid";
  })();

  const hasErrors = validation && validation.errors.length > 0;
  const hasWarnings = validation && validation.warnings.length > 0;

  const yamlLines = (configData?.content || "").split("\n");
  const mode = health?.mode ?? "test";
  const configFile = `serve-${mode}.yaml`;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <h2 className="text-xl font-semibold text-zinc-100">Settings</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { handleValidate(); refreshConfig(); }}
            disabled={validating}
            className="rounded-md px-3 py-1.5 text-sm font-medium border border-border text-zinc-300 hover:bg-surface-50 transition-colors duration-150 flex items-center gap-2 disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${validating ? "animate-spin" : ""}`} />
            Validate
          </button>
          <button
            ref={applyButtonRef}
            onClick={handleDeploy}
            disabled={deploying || status === "errors" || !canDeployConfig}
            title={
              !canDeployConfig
                ? "Editor permission required"
                : status === "errors"
                  ? "Fix errors before applying"
                  : "Copy config to deploy/"
            }
            className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors duration-150 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Upload className="w-3.5 h-3.5" />
            {deploying ? "Applying..." : "Apply"}
          </button>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-surface-100 px-5 py-3 text-sm text-zinc-400">
        <span className="text-zinc-200 font-medium">Apply</span> copies the current
        <code className="mx-1 rounded bg-surface-50 px-1.5 py-0.5 text-xs text-zinc-200">{configFile}</code>
        into the deployed configuration used by processing. Routes can be created in the UI, but the service only uses the applied config.
      </div>

      {!canDeployConfig && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
          Editor permission is required to apply configuration changes.
        </div>
      )}

      {configError && <ErrorBanner message={configError} />}

      {/* Notice */}
      {notice && (
        <div
          className={`rounded-lg px-4 py-2 text-sm font-medium flex items-center justify-between ${
            notice.type === "success"
              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
              : "bg-red-500/10 text-red-400 border border-red-500/30"
          }`}
        >
          {notice.text}
          <button onClick={() => setNotice(null)} className="opacity-60 hover:opacity-100">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Validation issues */}
      {hasErrors && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/5 overflow-hidden">
          <button
            onClick={() => setErrorsOpen(!errorsOpen)}
            className="w-full px-4 py-3 flex items-center gap-2 text-left hover:bg-red-500/5 transition-colors"
          >
            {errorsOpen ? <ChevronDown className="w-3.5 h-3.5 text-red-400" /> : <ChevronRight className="w-3.5 h-3.5 text-red-400" />}
            <AlertCircle className="w-4 h-4 text-red-400" />
            <span className="text-sm font-semibold text-red-400">
              {validation!.errors.length} {validation!.errors.length === 1 ? "Error" : "Errors"}
            </span>
            <span className="text-xs text-red-400/60 ml-1">— must fix before applying</span>
          </button>
          {errorsOpen && (
            <ul className="px-4 pb-3 space-y-1.5 ml-6">
              {validation!.errors.map((err, i) => (
                <li key={i} className="text-sm text-red-400/80 flex items-start gap-2">
                  <span className="text-red-500/40 select-none">-</span>
                  <span>{err}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {hasWarnings && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 overflow-hidden">
          <button
            onClick={() => setWarningsOpen(!warningsOpen)}
            className="w-full px-4 py-3 flex items-center gap-2 text-left hover:bg-amber-500/5 transition-colors"
          >
            {warningsOpen ? <ChevronDown className="w-3.5 h-3.5 text-amber-400" /> : <ChevronRight className="w-3.5 h-3.5 text-amber-400" />}
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-semibold text-amber-400">
              {validation!.warnings.length} {validation!.warnings.length === 1 ? "Warning" : "Warnings"}
            </span>
          </button>
          {warningsOpen && (
            <ul className="px-4 pb-3 space-y-1.5 ml-6">
              {validation!.warnings.map((w, i) => (
                <li key={i} className="text-sm text-amber-400/80 flex items-start gap-2">
                  <span className="text-amber-500/40 select-none">-</span>
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="rounded-lg border border-border bg-surface-100 overflow-hidden">
        <div className="px-5 py-3 border-b border-border flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-zinc-100">Authentication</h3>
            <p className="mt-1 text-xs text-zinc-500">GitHub and generic OIDC are supported. Session cookies automatically use `Secure` on HTTPS and stay compatible with local HTTP development.</p>
          </div>
          <button
            onClick={handleSaveAuthConfig}
            disabled={!authConfig || authSaving || !hasPermission("auth.admin")}
            className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors duration-150 disabled:opacity-50"
            title={!hasPermission("auth.admin") ? "Admin permission required" : undefined}
          >
            {authSaving ? "Saving..." : "Save Auth"}
          </button>
        </div>

        {authLoading ? (
          <div className="p-5 text-sm text-zinc-500">Loading authentication settings...</div>
        ) : !authConfig ? (
          <div className="p-5 text-sm text-zinc-500">Authentication settings are unavailable for this session.</div>
        ) : (
          <div className="grid gap-4 p-5 md:grid-cols-2">
            <label className="space-y-1">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Mode</span>
              <select
                value={authConfig.mode}
                onChange={(e) => setAuthConfig({ ...authConfig, mode: e.target.value })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              >
                <option value="oauth">OAuth Required</option>
                <option value="disabled" disabled={!authConfig.allow_disable_auth}>Disabled</option>
              </select>
            </label>

            {!authConfig.allow_disable_auth && (
              <div className="md:col-span-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-300">
                This workspace is configured to keep auth enabled. Disable mode is locked out by policy.
              </div>
            )}

            <label className="space-y-1">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Provider</span>
              <select
                value={authConfig.provider}
                onChange={(e) => setAuthConfig({ ...authConfig, provider: e.target.value })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              >
                <option value="github">GitHub</option>
                <option value="oidc">Generic OIDC</option>
              </select>
            </label>

            {authConfig.providers && (
              <div className="md:col-span-2 rounded-lg border border-border bg-surface-50 px-4 py-3 text-xs text-zinc-400">
                {Object.entries(authConfig.providers).map(([name, provider]) => (
                  <div key={name}>
                    {name}: {provider.configured ? "configured" : "not configured"}{provider.selected ? " · selected" : ""}
                  </div>
                ))}
              </div>
            )}

            {authConfig.provider === "github" ? (
              <>
                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">GitHub Client ID</span>
                  <input
                    value={authConfig.github.client_id}
                    onChange={(e) => setAuthConfig({ ...authConfig, github: { ...authConfig.github, client_id: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>

                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">GitHub Client Secret</span>
                  <input
                    type="password"
                    value={authConfig.github.client_secret}
                    onChange={(e) => setAuthConfig({ ...authConfig, github: { ...authConfig.github, client_secret: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>

                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Redirect URI</span>
                  <input
                    value={authConfig.github.redirect_uri}
                    onChange={(e) => setAuthConfig({ ...authConfig, github: { ...authConfig.github, redirect_uri: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>

                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Allowed GitHub Orgs</span>
                  <input
                    value={authConfig.github.allowed_orgs.join(", ")}
                    onChange={(e) => setAuthConfig({
                      ...authConfig,
                      github: {
                        ...authConfig.github,
                        allowed_orgs: e.target.value.split(",").map((value) => value.trim()).filter(Boolean),
                      },
                    })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>

                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Allowed GitHub Users</span>
                  <input
                    value={authConfig.github.allowed_users.join(", ")}
                    onChange={(e) => setAuthConfig({
                      ...authConfig,
                      github: {
                        ...authConfig.github,
                        allowed_users: e.target.value.split(",").map((value) => value.trim()).filter(Boolean),
                      },
                    })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
              </>
            ) : (
              <>
                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">OIDC Issuer URL</span>
                  <input
                    value={authConfig.oidc.issuer_url}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, issuer_url: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">OIDC Client ID</span>
                  <input
                    value={authConfig.oidc.client_id}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, client_id: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">OIDC Client Secret</span>
                  <input
                    type="password"
                    value={authConfig.oidc.client_secret}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, client_secret: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">OIDC Redirect URI</span>
                  <input
                    value={authConfig.oidc.redirect_uri}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, redirect_uri: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">OIDC Scopes</span>
                  <input
                    value={authConfig.oidc.scopes.join(", ")}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, scopes: e.target.value.split(",").map((value) => value.trim()).filter(Boolean) } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Username Claim</span>
                  <input
                    value={authConfig.oidc.username_claim}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, username_claim: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Groups Claim</span>
                  <input
                    value={authConfig.oidc.groups_claim}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, groups_claim: e.target.value } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Allowed OIDC Groups</span>
                  <input
                    value={authConfig.oidc.allowed_groups.join(", ")}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, allowed_groups: e.target.value.split(",").map((value) => value.trim()).filter(Boolean) } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
                <label className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Allowed OIDC Users</span>
                  <input
                    value={authConfig.oidc.allowed_users.join(", ")}
                    onChange={(e) => setAuthConfig({ ...authConfig, oidc: { ...authConfig.oidc, allowed_users: e.target.value.split(",").map((value) => value.trim()).filter(Boolean) } })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
              </>
            )}

            <label className="space-y-1 md:col-span-2">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Bootstrap Admins</span>
              <input
                value={authConfig.bootstrap_admins.join(", ")}
                onChange={(e) => setAuthConfig({
                  ...authConfig,
                  bootstrap_admins: e.target.value.split(",").map((value) => value.trim()).filter(Boolean),
                })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              />
            </label>
          </div>
        )}
      </div>

      <div className="rounded-lg border border-border bg-surface-100 overflow-hidden">
        <div className="px-5 py-3 border-b border-border">
          <h3 className="text-sm font-semibold text-zinc-100">User Roles</h3>
          <p className="mt-1 text-xs text-zinc-500">Manage viewer, operator, editor, and admin access for this Serve workspace.</p>
        </div>
        {!hasPermission("users.admin") ? (
          <div className="p-5 text-sm text-zinc-500">Admin permission is required to manage users.</div>
        ) : usersLoading ? (
          <div className="p-5 text-sm text-zinc-500">Loading users...</div>
        ) : adminUsers.length === 0 ? (
          <div className="p-5 text-sm text-zinc-500">No authenticated users yet.</div>
        ) : (
          <div className="divide-y divide-border">
            {adminUsers.map((user) => (
              <div key={user.id} className="flex flex-col gap-3 px-5 py-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="text-sm font-medium text-zinc-100">{user.display_name || user.login}</div>
                  <div className="mt-1 text-xs text-zinc-500">
                    {user.provider} · {user.email || "no email"}
                  </div>
                  <div className="mt-1 text-[11px] text-zinc-600">
                    Last login: {user.last_login_at || "never"} · {user.disabled ? "disabled" : "active"}
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {["viewer", "operator", "editor", "admin"].map((role) => {
                    const active = user.roles.includes(role);
                    const savingThis = roleSaving === `${user.id}:${role}`;
                    return (
                      <button
                        key={role}
                        onClick={() => toggleRole(user.id, user.roles, role)}
                        disabled={Boolean(roleSaving)}
                        className={[
                          "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                          active
                            ? "border-brand/40 bg-brand/10 text-brand"
                            : "border-border text-zinc-400 hover:bg-surface-50 hover:text-zinc-200",
                        ].join(" ")}
                      >
                        {savingThis ? "Saving..." : role}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-lg border border-border bg-surface-100 overflow-hidden">
        <div className="px-5 py-3 border-b border-border flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-zinc-100">Email Notifications</h3>
            <p className="mt-1 text-xs text-zinc-500">Configure Resend so labs can receive service and failure alerts.</p>
          </div>
          <button
            onClick={handleSaveNotificationsConfig}
            disabled={!notificationsConfig || notificationsSaving || !hasPermission("auth.admin")}
            className="rounded-md px-3 py-1.5 text-sm font-medium bg-brand text-surface-500 hover:bg-brand-500 transition-colors duration-150 disabled:opacity-50"
            title={!hasPermission("auth.admin") ? "Admin permission required" : undefined}
          >
            {notificationsSaving ? "Saving..." : "Save Notifications"}
          </button>
        </div>
        {notificationsLoading ? (
          <div className="p-5 text-sm text-zinc-500">Loading notification settings...</div>
        ) : !notificationsConfig ? (
          <div className="p-5 text-sm text-zinc-500">Notification settings are unavailable for this session.</div>
        ) : !hasPermission("auth.admin") ? (
          <div className="p-5 text-sm text-zinc-500">Admin permission is required to manage notifications.</div>
        ) : (
          <div className="grid gap-4 p-5 md:grid-cols-2">
            <label className="space-y-1">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Enabled</span>
              <select
                value={notificationsConfig.enabled ? "enabled" : "disabled"}
                onChange={(e) => setNotificationsConfig({ ...notificationsConfig, enabled: e.target.value === "enabled" })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              >
                <option value="disabled">Disabled</option>
                <option value="enabled">Enabled</option>
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Provider</span>
              <input
                value={notificationsConfig.provider}
                onChange={(e) => setNotificationsConfig({ ...notificationsConfig, provider: e.target.value })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              />
            </label>

            {notificationsConfig.providers && (
              <div className="md:col-span-2 rounded-lg border border-border bg-surface-50 px-4 py-3 text-xs text-zinc-400">
                {Object.entries(notificationsConfig.providers).map(([name, provider]) => (
                  <div key={name}>
                    {name}: {provider.configured ? "configured" : "not configured"}{provider.selected ? " · selected" : ""}
                  </div>
                ))}
              </div>
            )}

            <label className="space-y-1 md:col-span-2">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Serve Base URL</span>
              <input
                value={notificationsConfig.app_base_url}
                onChange={(e) => setNotificationsConfig({
                  ...notificationsConfig,
                  app_base_url: e.target.value,
                })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              />
            </label>

            <label className="space-y-1 md:col-span-2">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Resend API Key</span>
              <input
                type="password"
                value={notificationsConfig.resend.api_key}
                onChange={(e) => setNotificationsConfig({
                  ...notificationsConfig,
                  resend: { ...notificationsConfig.resend, api_key: e.target.value },
                })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              />
            </label>

            <label className="space-y-1">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Sender Name</span>
              <input
                value={notificationsConfig.resend.sender_name}
                onChange={(e) => setNotificationsConfig({
                  ...notificationsConfig,
                  resend: { ...notificationsConfig.resend, sender_name: e.target.value },
                })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              />
            </label>

            <label className="space-y-1">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Sender Email</span>
              <input
                value={notificationsConfig.resend.sender_email}
                onChange={(e) => setNotificationsConfig({
                  ...notificationsConfig,
                  resend: { ...notificationsConfig.resend, sender_email: e.target.value },
                })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              />
            </label>

            <label className="space-y-1 md:col-span-2">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Reply-To</span>
              <input
                value={notificationsConfig.resend.reply_to}
                onChange={(e) => setNotificationsConfig({
                  ...notificationsConfig,
                  resend: { ...notificationsConfig.resend, reply_to: e.target.value },
                })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              />
            </label>

            {[
              ["service", "Service recipients"],
              ["config_failure", "Config failure recipients"],
              ["queue_failure", "Queue failure recipients"],
              ["job_failure", "Job failure recipients"],
            ].map(([key, label]) => {
              const recipientKey = key as keyof NotificationsConfigResponse["recipients"];
              return (
                <label key={key} className="space-y-1 md:col-span-2">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">{label}</span>
                  <input
                    value={notificationsConfig.recipients[recipientKey].join(", ")}
                    onChange={(e) => setNotificationsConfig({
                      ...notificationsConfig,
                      recipients: {
                        ...notificationsConfig.recipients,
                        [recipientKey]: e.target.value.split(",").map((value) => value.trim()).filter(Boolean),
                      },
                    })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
              );
            })}

            {[
              ["service", "Service cooldown (minutes)"],
              ["config_failure", "Config failure cooldown (minutes)"],
              ["queue_failure", "Queue failure cooldown (minutes)"],
              ["job_failure", "Job failure cooldown (minutes)"],
            ].map(([key, label]) => {
              const cooldownKey = key as keyof NotificationsConfigResponse["cooldown_minutes"];
              return (
                <label key={key} className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">{label}</span>
                  <input
                    type="number"
                    min={0}
                    value={notificationsConfig.cooldown_minutes[cooldownKey]}
                    onChange={(e) => setNotificationsConfig({
                      ...notificationsConfig,
                      cooldown_minutes: {
                        ...notificationsConfig.cooldown_minutes,
                        [cooldownKey]: Number(e.target.value || 0),
                      },
                    })}
                    className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
                  />
                </label>
              );
            })}

            <div className="space-y-2 md:col-span-2 rounded-lg border border-border bg-surface-50 p-4">
              <div className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Test Email</div>
              <input
                value={testEmailTo}
                onChange={(e) => setTestEmailTo(e.target.value)}
                placeholder="scientist@example.edu, lab@example.edu"
                className="w-full rounded-md border border-border bg-surface-100 px-3 py-2 text-sm text-zinc-200"
              />
              <button
                onClick={handleSendTestEmail}
                disabled={testEmailSending}
                className="rounded-md px-3 py-1.5 text-sm font-medium border border-border text-zinc-200 hover:bg-surface-100 disabled:opacity-50"
              >
                {testEmailSending ? "Sending..." : "Send Test Email"}
              </button>
            </div>

            <label className="space-y-1 md:col-span-2">
              <span className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Daily Digest Recipients</span>
              <input
                value={notificationsConfig.daily_digest_recipients.join(", ")}
                onChange={(e) => setNotificationsConfig({
                  ...notificationsConfig,
                  daily_digest_recipients: e.target.value.split(",").map((value) => value.trim()).filter(Boolean),
                })}
                className="w-full rounded-md border border-border bg-surface-50 px-3 py-2 text-sm text-zinc-200"
              />
            </label>

            <div className="space-y-2 md:col-span-2 rounded-lg border border-border bg-surface-50 p-4">
              <div className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Route-Specific Recipients</div>
              <input
                value={routeRecipientsRouteId}
                onChange={(e) => setRouteRecipientsRouteId(e.target.value)}
                placeholder="route-id"
                className="w-full rounded-md border border-border bg-surface-100 px-3 py-2 text-sm text-zinc-200"
              />
              <input
                value={routeRecipientsValue}
                onChange={(e) => setRouteRecipientsValue(e.target.value)}
                placeholder="scientist@example.edu, owner@example.edu"
                className="w-full rounded-md border border-border bg-surface-100 px-3 py-2 text-sm text-zinc-200"
              />
              <button
                onClick={() => {
                  if (!routeRecipientsRouteId.trim()) return;
                  setNotificationsConfig({
                    ...notificationsConfig,
                    route_recipients: {
                      ...notificationsConfig.route_recipients,
                      [routeRecipientsRouteId.trim()]: routeRecipientsValue.split(",").map((value) => value.trim()).filter(Boolean),
                    },
                  });
                  setRouteRecipientsRouteId("");
                  setRouteRecipientsValue("");
                }}
                className="rounded-md px-3 py-1.5 text-sm font-medium border border-border text-zinc-200 hover:bg-surface-100"
              >
                Save Route Recipients
              </button>
              <div className="space-y-1 text-xs text-zinc-400">
                {Object.entries(notificationsConfig.route_recipients).map(([routeId, recipients]) => (
                  <div key={routeId}>{routeId}: {recipients.join(", ") || "none"}</div>
                ))}
              </div>
            </div>

            <div className="space-y-2 md:col-span-2 rounded-lg border border-border bg-surface-50 p-4">
              <div className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">Daily Digest</div>
              <button
                onClick={handleSendDailyDigest}
                disabled={dailyDigestSending}
                className="rounded-md px-3 py-1.5 text-sm font-medium border border-border text-zinc-200 hover:bg-surface-100 disabled:opacity-50"
              >
                {dailyDigestSending ? "Sending..." : "Send Daily Digest"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* YAML viewer */}
      <div className="rounded-lg border border-border bg-surface-100 overflow-hidden">
        {/* Toolbar */}
        <div className="px-5 py-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-4 h-4 text-zinc-500" />
            <span className="text-sm font-semibold text-zinc-100">{configFile}</span>
            <span className="text-xs text-zinc-600">
              ({mode === "live" ? "Live" : "Testing"} lane)
            </span>
          </div>
          <div className="flex items-center gap-3">
            {/* Status indicator */}
            {status === "valid" && (
              <div className="flex items-center gap-1.5 text-xs text-brand">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Valid
              </div>
            )}
            {status === "errors" && (
              <div className="flex items-center gap-1.5 text-xs text-red-400">
                <AlertCircle className="w-3.5 h-3.5" />
                Invalid
              </div>
            )}
            {status === "warnings" && (
              <div className="flex items-center gap-1.5 text-xs text-amber-400">
                <AlertTriangle className="w-3.5 h-3.5" />
                Warnings
              </div>
            )}
            {status === "checking" && (
              <div className="flex items-center gap-1.5 text-xs text-zinc-500">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Checking
              </div>
            )}
            <span className="text-xs text-zinc-600">{yamlLines.length} lines</span>
          </div>
        </div>

        {/* Code area */}
        {configLoading && !configData ? (
          <div className="p-6 text-sm text-zinc-600 bg-[#0A0A0A]">Loading configuration...</div>
        ) : yamlLines.length === 0 || !configData?.content ? (
          <div className="p-6 text-center bg-[#0A0A0A]">
            <FileText className="w-8 h-8 text-zinc-700 mx-auto mb-2" />
            <p className="text-sm text-zinc-500">No configuration found</p>
            <p className="text-xs text-zinc-600 mt-1">Create routes first, then sync to generate the config</p>
          </div>
        ) : (
          <CodeViewer lines={yamlLines} colorize={colorizeYaml} />
        )}
      </div>
    </div>
  );
}

// Simple YAML syntax highlighting
function colorizeYaml(line: string): React.ReactNode {
  // Comment lines
  if (/^\s*#/.test(line)) {
    return <span className="text-zinc-600">{line}</span>;
  }
  // Key: value lines
  const match = line.match(/^(\s*)([\w_-]+)(:)(.*)/);
  if (match) {
    const [, indent, key, colon, rest] = match as RegExpMatchArray;
    return (
      <>
        {indent ?? ""}
        <span className="text-cyan-400">{key ?? ""}</span>
        <span className="text-zinc-500">{colon ?? ""}</span>
        {colorizeValue(rest ?? "")}
      </>
    );
  }
  // List items
  const listMatch = line.match(/^(\s*)(- )(.*)/);
  if (listMatch) {
    const [, indent, dash, rest] = listMatch as RegExpMatchArray;
    return (
      <>
        {indent}
        <span className="text-zinc-500">{dash}</span>
        <span className="text-zinc-300">{rest}</span>
      </>
    );
  }
  return line;
}

function colorizeValue(val: string): React.ReactNode {
  const trimmed = val.trim();
  if (!trimmed) return val;
  // Boolean
  if (/^(true|false)$/i.test(trimmed)) {
    return <span className="text-amber-400"> {trimmed}</span>;
  }
  // Number
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    return <span className="text-purple-400"> {trimmed}</span>;
  }
  // String values
  return <span className="text-emerald-400">{val}</span>;
}
