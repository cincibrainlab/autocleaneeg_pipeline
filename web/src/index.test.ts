// @ts-expect-error -- tests run in Node; the browser app intentionally omits Node types.
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const css = readFileSync("src/index.css", "utf8");

describe("light surface hierarchy", () => {
  it("keeps semantic surface levels in monotonic light-to-deep order", () => {
    const lightBlock = css.match(/html\.light\s*\{([^}]*)\}/)?.[1] ?? "";
    const values = [
      ...lightBlock.matchAll(
        /--color-surface-(\d+):\s*(#[0-9a-f]{6})/g,
      ),
    ].map(([, level, value]) => [Number(level), value]);

    expect(values).toEqual([
      [50, "#ffffff"],
      [100, "#fafafa"],
      [200, "#f5f5f5"],
      [300, "#f0f0f0"],
      [400, "#e8e8e8"],
      [500, "#e5e5e5"],
    ]);
    expect(lightBlock).toContain("--color-surface: #f0f0f0;");
  });
});
