import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import CodeViewer from "./CodeViewer";

describe("CodeViewer theme treatment", () => {
  it("uses theme-aware surfaces without changing its code-view contract", () => {
    const { container } = render(
      <CodeViewer
        lines={["first", "second"]}
        colorize={(line) => <span data-testid={`color-${line}`}>{line}</span>}
        maxHeight="320px"
      />,
    );

    const viewer = container.firstElementChild;
    expect(viewer).toHaveClass(
      "overflow-x-auto",
      "overflow-y-auto",
      "bg-surface-500",
    );
    expect(viewer).toHaveStyle({ maxHeight: "320px" });

    const rows = container.querySelectorAll("tbody tr");
    expect(rows).toHaveLength(2);
    rows.forEach((row) => {
      expect(row).toHaveClass("hover:bg-surface-50/30", "transition-colors");
    });

    const lineNumbers = container.querySelectorAll("tbody td:first-child");
    expect(lineNumbers).toHaveLength(2);
    lineNumbers.forEach((cell) => expect(cell).toHaveClass("text-zinc-400"));

    const codeCells = container.querySelectorAll("tbody td:last-child");
    expect(codeCells).toHaveLength(2);
    codeCells.forEach((cell) => {
      expect(cell).toHaveClass("text-zinc-300", "whitespace-pre");
    });

    expect(screen.getByTestId("color-first")).toHaveTextContent("first");
    expect(screen.getByTestId("color-second")).toHaveTextContent("second");
  });
});
