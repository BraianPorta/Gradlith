import { describe, expect, it } from "vitest";

describe("playground", () => {
  it("bundles the training worker entry", () => {
    const workerUrl = new URL("./training.worker.ts", import.meta.url);

    expect(workerUrl.pathname).toContain("training.worker.ts");
  });
});

