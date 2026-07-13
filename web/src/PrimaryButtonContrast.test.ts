// @ts-expect-error -- tests run in Node; the browser app intentionally omits Node types.
import { readFileSync, readdirSync } from "node:fs";

import { describe, expect, it } from "vitest";

describe("primary button contrast", () => {
  it("uses a stable dark foreground for every on-brand action", () => {
    const sources = readdirSync("src", { recursive: true })
      .filter((path: string) => path.endsWith(".tsx"))
      .map((path: string) => readFileSync(`src/${path}`, "utf8"))
      .join("\n");

    expect(sources).not.toContain("text-surface-500");
    expect(
      sources.match(/bg-brand[^"\n]*text-brand-900/g),
    ).toHaveLength(21);
  });
});
