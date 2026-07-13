// @ts-expect-error -- tests run in Node; the browser app intentionally omits Node types.
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const dialogSources = [
  "src/components/FolderBrowser.tsx",
  "src/components/KeyboardShortcutsHelp.tsx",
  "src/components/tutorial/TutorialWelcome.tsx",
  "src/components/tutorial/TutorialComplete.tsx",
  "src/pages/Routes.tsx",
];

describe("mobile dialog widths", () => {
  it.each(dialogSources)("keeps 16px viewport gutters in %s", (path) => {
    const source = readFileSync(path, "utf8");

    expect(source).toContain("w-[calc(100%-2rem)]");
    expect(source).not.toMatch(/w-full max-w-(?:md|lg|xl)[^"]*mx-4/);
  });
});
