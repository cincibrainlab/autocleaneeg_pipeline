import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import RoutesPage from "./pages/Routes";
import Queue from "./pages/Queue";
import Service from "./pages/Service";
import Settings from "./pages/Settings";
import TasksPage from "./pages/Tasks";
import MontagesPage from "./pages/Montages";
import ResultsPage from "./pages/Results";
import EventAnalyzerPage from "./pages/EventAnalyzer";
import Setup from "./pages/Setup";
import ExcludePage from "./pages/Exclude";
import { TutorialProvider } from "./contexts/TutorialContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import TutorialOverlay from "./components/tutorial/TutorialOverlay";
import { api } from "./lib/api";
import type { AuthStatus } from "./lib/api";

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="flex items-center justify-center h-screen bg-surface-500">
          <div className="text-center p-8 max-w-md">
            <h1 className="text-xl font-bold text-zinc-100 mb-2">Something went wrong</h1>
            <p className="text-sm text-zinc-400 mb-4">{this.state.error.message}</p>
            <button
              onClick={() => {
                this.setState({ error: null });
                window.location.reload();
              }}
              className="px-4 py-2 bg-brand text-surface-500 rounded font-medium text-sm"
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [authStatus, setAuthStatus] = React.useState<AuthStatus | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [authError, setAuthError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let active = true;
    async function bootstrap() {
      try {
        const status = await api.getAuthStatus();
        if (!active) return;
        setAuthStatus(status);
        if (status.enabled && status.authenticated) {
          await api.getMe();
        }
      } catch (error) {
        if (!active) return;
        setAuthError(error instanceof Error ? error.message : String(error));
      } finally {
        if (active) setLoading(false);
      }
    }
    bootstrap();
    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return <div className="flex h-screen items-center justify-center bg-surface-500 text-zinc-300">Loading Serve...</div>;
  }

  if (authError) {
    return (
      <div className="flex h-screen items-center justify-center bg-surface-500 px-6">
        <div className="max-w-md rounded-lg border border-red-500/30 bg-red-500/10 p-6 text-sm text-red-300">
          {authError}
        </div>
      </div>
    );
  }

  if (authStatus?.enabled && !authStatus.authenticated) {
    const availableProviders = Object.entries(authStatus.providers ?? {})
      .filter(([, status]) => status.configured)
      .map(([name]) => name);
    if (!authStatus.configured && authStatus.bootstrap_allowed) {
      return (
        <ErrorBoundary>
          <ThemeProvider>
            <BrowserRouter>
              <Routes>
                <Route path="*" element={<Settings />} />
              </Routes>
            </BrowserRouter>
          </ThemeProvider>
        </ErrorBoundary>
      );
    }
    return (
      <ErrorBoundary>
        <ThemeProvider>
          <div className="flex min-h-screen items-center justify-center bg-surface-500 px-6">
            <div className="w-full max-w-md rounded-2xl border border-border bg-surface-200 p-8">
              <h1 className="text-2xl font-semibold text-zinc-100">Sign in to AutoClean Serve</h1>
              <p className="mt-2 text-sm text-zinc-400">
                {authStatus.configured
                  ? "Authentication is enabled for this workspace."
                  : `Authentication is enabled, but ${authStatus.provider || "the selected provider"} is not configured yet.`}
              </p>
              {authStatus.configured ? (
                <div className="mt-6 space-y-3">
                  {availableProviders.map((provider) => (
                    <button
                      key={provider}
                      onClick={async () => {
                        const result = await api.login(provider);
                        window.location.href = result.login_url;
                      }}
                      className="w-full rounded-md bg-brand px-4 py-2 text-sm font-medium text-surface-500 hover:bg-brand-500"
                    >
                      Continue with {provider.toUpperCase()}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="mt-6 rounded-md border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
                  An admin needs to configure at least one auth provider before login can start.
                </div>
              )}
            </div>
          </div>
        </ThemeProvider>
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary>
      <ThemeProvider>
      <BrowserRouter>
        <TutorialProvider>
          <Routes>
            {/* Main app shell (includes workspace picker) */}
            <Route element={<Layout />}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/setup" element={<Setup />} />
              <Route path="/routes" element={<RoutesPage />} />
              <Route path="/queue" element={<Queue />} />
              <Route path="/service" element={<Service />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/tasks" element={<TasksPage />} />
              <Route path="/montages" element={<MontagesPage />} />
              <Route path="/results" element={<ResultsPage />} />
              <Route path="/exclude" element={<ExcludePage />} />
              <Route path="/events" element={<EventAnalyzerPage />} />
              <Route
                path="*"
                element={
                  <div className="flex flex-col items-center justify-center py-20 text-center">
                    <h2 className="text-xl font-bold text-zinc-100 mb-2">Page not found</h2>
                    <p className="text-sm text-zinc-500 mb-4">
                      The page you're looking for doesn't exist.
                    </p>
                    <a href="/" className="text-sm text-brand hover:underline">
                      Go to Dashboard
                    </a>
                  </div>
                }
              />
            </Route>
          </Routes>
          <TutorialOverlay />
        </TutorialProvider>
      </BrowserRouter>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
