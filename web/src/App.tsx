import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import RoutesPage from "./pages/Routes";
import Queue from "./pages/Queue";
import Service from "./pages/Service";
import Settings from "./pages/Settings";
import Utilities from "./pages/Utilities";
import TasksPage from "./pages/Tasks";
import MontagesPage from "./pages/Montages";
import ResultsPage from "./pages/Results";
import EventAnalyzerPage from "./pages/EventAnalyzer";
import Setup from "./pages/Setup";
import ExcludePage from "./pages/Exclude";
import { TutorialProvider } from "./contexts/TutorialContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import TutorialOverlay from "./components/tutorial/TutorialOverlay";

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
              className="px-4 py-2 bg-brand text-brand-900 rounded font-medium text-sm"
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
              <Route path="/utilities" element={<Utilities />} />
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
