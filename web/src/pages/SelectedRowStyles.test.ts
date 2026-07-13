// @ts-expect-error -- tests run in Node; the browser app intentionally omits Node types.
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const selectedRowSources = [
  "Queue.tsx",
  "Tasks.tsx",
  "Montages.tsx",
  "Results.tsx",
];

describe("selected table row styles", () => {
  it.each(selectedRowSources)(
    "uses a clear dual-theme selection in %s",
    (file) => {
      const source = readFileSync(`src/pages/${file}`, "utf8");

      expect(source).toContain('"bg-brand/15 dark:bg-brand/10"');
      expect(source).not.toContain('"bg-brand/5"');
    },
  );
});
