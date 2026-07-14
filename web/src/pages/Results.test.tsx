import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const { api } = vi.hoisted(() => ({
  api: { setDecision: vi.fn() },
}));

vi.mock("../lib/api", () => ({ api }));

import { DecisionBar } from "./Results";

describe("DecisionBar", () => {
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
});
