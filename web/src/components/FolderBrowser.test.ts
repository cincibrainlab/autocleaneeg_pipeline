import { describe, expect, it } from "vitest";

import { pathSegments } from "./FolderBrowser";

describe("FolderBrowser pathSegments", () => {
  it("preserves native Windows drive paths", () => {
    expect(pathSegments("C:\\Users\\Researcher\\EEG data")).toEqual([
      { label: "C:\\", path: "C:\\" },
      { label: "Users", path: "C:\\Users" },
      { label: "Researcher", path: "C:\\Users\\Researcher" },
      { label: "EEG data", path: "C:\\Users\\Researcher\\EEG data" },
    ]);
  });

  it("preserves Windows drive paths that use forward slashes", () => {
    expect(pathSegments("D:/Data/Session 01")).toEqual([
      { label: "D:/", path: "D:/" },
      { label: "Data", path: "D:/Data" },
      { label: "Session 01", path: "D:/Data/Session 01" },
    ]);
  });

  it("preserves UNC server and share roots", () => {
    expect(pathSegments("\\\\server\\share\\study\\subject-01")).toEqual([
      { label: "\\\\server\\share", path: "\\\\server\\share" },
      { label: "study", path: "\\\\server\\share\\study" },
      {
        label: "subject-01",
        path: "\\\\server\\share\\study\\subject-01",
      },
    ]);
  });

  it("keeps POSIX breadcrumbs unchanged", () => {
    expect(pathSegments("/home/researcher/data")).toEqual([
      { label: "/", path: "/" },
      { label: "home", path: "/home" },
      { label: "researcher", path: "/home/researcher" },
      { label: "data", path: "/home/researcher/data" },
    ]);
  });
});
