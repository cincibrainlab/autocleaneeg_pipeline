import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: { setDecision: vi.fn() },
}));

vi.mock("../lib/api", () => ({ api }));

import { DecisionBar } from "./Results";

describe("DecisionBar", () => {
  beforeEach(() => {
    api.setDecision.mockReset();
  });

  it("reports a failed save and retries without updating the parent early", async () => {
    api.setDecision
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(undefined);
    const onDecisionChange = vi.fn();

    render(
      <DecisionBar
        runId="run-1"
        currentDecision={null}
        currentNotes="needs review"
        onDecisionChange={onDecisionChange}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /pass/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Decision could not be saved.",
    );
    expect(onDecisionChange).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => {
      expect(api.setDecision).toHaveBeenCalledTimes(2);
      expect(onDecisionChange).toHaveBeenCalledWith("pass", "needs review");
    });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("preserves a failed decision when notes trigger an implicit save", async () => {
    api.setDecision
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue(undefined);
    const onDecisionChange = vi.fn();

    render(
      <DecisionBar
        runId="run-1"
        currentDecision="review"
        currentNotes="old notes"
        onDecisionChange={onDecisionChange}
      />,
    );

    const notes = screen.getByRole("textbox");
    fireEvent.change(notes, { target: { value: "updated notes" } });
    fireEvent.click(screen.getByRole("button", { name: /pass/i }));

    expect(await screen.findByRole("alert")).toBeInTheDocument();

    fireEvent.blur(notes);
    fireEvent.focus(notes);
    fireEvent.keyDown(notes, { key: "Enter" });

    await waitFor(() => {
      expect(api.setDecision).toHaveBeenCalledTimes(1);
      expect(api.setDecision).toHaveBeenLastCalledWith(
        "run-1",
        "pass",
        "updated notes",
      );
      expect(onDecisionChange).not.toHaveBeenCalled();
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
  });

  it("does not carry a failed decision into another run", async () => {
    api.setDecision
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(undefined);
    const onDecisionChange = vi.fn();

    const { rerender } = render(
      <DecisionBar
        key="run-1"
        runId="run-1"
        currentDecision={null}
        currentNotes="old run notes"
        onDecisionChange={onDecisionChange}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /fail/i }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();

    rerender(
      <DecisionBar
        key="run-2"
        runId="run-2"
        currentDecision={null}
        currentNotes="new run notes"
        onDecisionChange={onDecisionChange}
      />,
    );

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /retry/i }),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /pass/i }));

    await waitFor(() => {
      expect(api.setDecision).toHaveBeenLastCalledWith(
        "run-2",
        "pass",
        "new run notes",
      );
      expect(onDecisionChange).toHaveBeenCalledWith("pass", "new run notes");
    });
  });
});
