import { useState, useCallback } from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import TopBar from "./TopBar";
import KeyboardShortcutsHelp from "./KeyboardShortcutsHelp";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  const closeSidebar = useCallback(() => setSidebarOpen(false), []);
  const toggleSidebar = useCallback(() => setSidebarOpen((v) => !v), []);
  const toggleShortcuts = useCallback(() => setShortcutsOpen((v) => !v), []);
  const closeShortcuts = useCallback(() => setShortcutsOpen(false), []);

  useKeyboardShortcuts(toggleShortcuts);

  return (
    <div className="flex h-screen">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          onClick={closeSidebar}
        />
      )}
      <Sidebar open={sidebarOpen} onClose={closeSidebar} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <TopBar onToggleSidebar={toggleSidebar} />
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
      <KeyboardShortcutsHelp open={shortcutsOpen} onClose={closeShortcuts} />
    </div>
  );
}
