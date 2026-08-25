import { describe, expect, it } from "vitest";
import { getBackend, setBackend, Tensor, WebGPUBackend } from "../index";

describe("Backends", () => {
  it("keeps WebGPU fallback storage compatible with CPU tensor operations", () => {
    const gpu = new WebGPUBackend();
    const a = Tensor.from([
      [1, 2],
      [3, 4]
    ]);
    const b = Tensor.from([
      [2, 0],
      [1, 2]
    ]);
    const fromBackend = Tensor.fromStorage(gpu.matmul(a.storage, b.storage, { m: 2, k: 2, n: 2 }), [2, 2]);

    expect(fromBackend.toArray()).toEqual(a.matmul(b).toArray());
  });

  it("exposes support detection for browser parity tests", () => {
    expect(typeof WebGPUBackend.isSupported()).toBe("boolean");
  });

  it("can select the CPU backend through the public runtime API", async () => {
    await setBackend("cpu");

    expect(getBackend()).toBe("cpu");
  });
});
