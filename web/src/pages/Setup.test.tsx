import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getRecentWorkspaces: vi.fn(),
    setupWorkspace: vi.fn(),
    browseFolders: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api }));

import Setup from "./Setup";

describe("Setup page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getRecentWorkspaces.mockResolvedValue({ workspaces: [] });
  });

  it("explains when to open an existing workspace versus create a new one", async () => {
    render(
      <MemoryRouter>
        <Setup />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText(/existing Serve workspace, or for an existing AutoClean workspace/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/existing Serve workspace, or for an existing AutoClean workspace/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/only for a new, empty directory/i),
    ).toBeInTheDocument();
  });
});
