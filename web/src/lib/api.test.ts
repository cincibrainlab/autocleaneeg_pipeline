import { describe, expect, it } from "vitest";

import { api } from "./api";

describe("results CSV URL", () => {
  it("includes the selected route", () => {
    expect(api.getResultsCsvUrl("route one/alpha")).toBe(
      "/api/results/export/csv?route_id=route%20one%2Falpha",
    );
  });

  it("keeps the unfiltered URL when no route is selected", () => {
    expect(api.getResultsCsvUrl()).toBe("/api/results/export/csv");
  });
});
