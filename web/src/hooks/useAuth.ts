import { useEffect, useMemo, useState } from "react";
import { usePolling } from "./usePolling";
import { api } from "../lib/api";
import type { AuthStatus, MeResponse } from "../lib/api";

export function useAuth() {
  const { data: authStatus, error, loading, refresh } = usePolling<AuthStatus>(
    api.getAuthStatus,
    10000
  );
  const [me, setMe] = useState<MeResponse | null>(null);

  useEffect(() => {
    let active = true;
    async function loadMe() {
      if (!authStatus?.enabled || !authStatus.authenticated) {
        if (active) setMe(null);
        return;
      }
      try {
        const current = await api.getMe();
        if (active) setMe(current);
      } catch {
        if (active) setMe(null);
      }
    }
    loadMe();
    return () => {
      active = false;
    };
  }, [authStatus?.authenticated, authStatus?.enabled]);

  const permissions = useMemo(
    () => new Set(me?.permissions ?? []),
    [me?.permissions]
  );

  return {
    authStatus,
    me,
    permissions,
    loading,
    error,
    refresh,
    hasPermission: (permission: string) =>
      !authStatus?.enabled || permissions.has(permission),
    isAuthenticated: !authStatus?.enabled || Boolean(authStatus?.authenticated),
  };
}
