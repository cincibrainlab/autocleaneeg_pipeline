import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: {
    getConfigYaml: vi.fn(),
    getHealth: vi.fn(),
    validateConfig: vi.fn(),
    deployConfig: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api }));
vi.mock("../hooks/usePolling", () => ({
  usePolling: (fetcher: unknown) => ({
    data:
      fetcher === api.getConfigYaml
        ? { content: "tasks: {}" }
        : { mode: "test" },
    error: null,
    loading: false,
    refresh: vi.fn(),
  }),
}));
vi.mock("../contexts/TutorialContext", () => ({
  useTutorial: () => ({ isActive: false, currentStep: 0, nextStep: vi.fn() }),
}));
vi.mock("../hooks/useTutorialTarget", () => ({
  useTutorialTarget: () => null,
}));

import Settings from "./Settings";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

const validResult = {
  valid: true,
  errors: [],
  warnings: [],
};

describe("Settings Apply validation gate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("keeps Apply disabled until initial validation finishes", async () => {
    const validation = deferred<typeof validResult>();
    api.validateConfig.mockReturnValue(validation.promise);

    render(<Settings />);

    const applyButton = screen.getByRole("button", { name: "Apply" });
    expect(applyButton).toBeDisabled();
    expect(applyButton).toHaveAttribute("title", "Wait for validation to finish");

    validation.resolve(validResult);

    await waitFor(() => expect(applyButton).toBeEnabled());
  });

  it("disables Apply during revalidation and re-enables it for warnings", async () => {
    const revalidation = deferred<{
      valid: boolean;
      errors: string[];
      warnings: string[];
    }>();
    api.validateConfig
      .mockResolvedValueOnce(validResult)
      .mockReturnValueOnce(revalidation.promise);

    render(<Settings />);

    const applyButton = screen.getByRole("button", { name: "Apply" });
    await waitFor(() => expect(applyButton).toBeEnabled());

    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    expect(applyButton).toBeDisabled();

    revalidation.resolve({
      valid: true,
      errors: [],
      warnings: ["Review this setting"],
    });

    await waitFor(() => expect(applyButton).toBeEnabled());
  });
});
