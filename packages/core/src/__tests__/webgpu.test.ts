import { describe, expect, it } from "vitest";
import { CPUBackend, Tensor, WebGPUBackend } from "../index";

describe("Backends", () => {
  it("keeps synchronous WebGPU methods compatible through CPU fallback", () => {
    const gpu = new WebGPUBackend();
    const cpu = new CPUBackend();
    const a = Tensor.from([
      [1, 2],
      [3, 4]
    ]);
    const b = Tensor.from([
      [2, 0],
      [1, 2]
    ]);

    expect(gpu.matmul(a, b).toArray()).toEqual(cpu.matmul(a, b).toArray());
  });

  it("exposes support detection for browser parity tests", () => {
    expect(typeof WebGPUBackend.isSupported()).toBe("boolean");
  });
});

