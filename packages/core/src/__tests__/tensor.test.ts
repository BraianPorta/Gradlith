import { describe, expect, it } from "vitest";
import { Gradlith, noGradAsync, Tensor, tensorAt } from "../index";

describe("Tensor", () => {
  it("infers shape, strides and row-major offsets", () => {
    const tensor = Tensor.from([
      [1, 2, 3],
      [4, 5, 6]
    ]);

    expect(tensor.shape).toEqual([2, 3]);
    expect(tensor.strides).toEqual([3, 1]);
    expect(tensorAt(tensor, [1, 2])).toBe(6);
  });

  it("broadcasts vectors across matrix rows", () => {
    const matrix = Tensor.from([
      [1, 2, 3],
      [4, 5, 6]
    ]);
    const bias = Tensor.from([10, 20, 30]);

    expect(matrix.add(bias).toArray()).toEqual([
      [11, 22, 33],
      [14, 25, 36]
    ]);
  });

  it("broadcastTo expands data and reduces gradients back to the source shape", () => {
    const bias = Tensor.from([1, 2, 3], { requiresGrad: true });

    bias.broadcastTo([2, 3]).sum().backward();

    expect(bias.broadcastTo([2, 3]).toArray()).toEqual([
      [1, 2, 3],
      [1, 2, 3]
    ]);
    expect(Array.from(bias.grad?.data ?? [])).toEqual([2, 2, 2]);
  });

  it("computes max, min and argmax", () => {
    const tensor = Tensor.from([1, -3, 7, 2], { requiresGrad: true });

    tensor.max().backward();

    expect(tensor.max().item()).toBe(7);
    expect(tensor.min().item()).toBe(-3);
    expect(tensor.argmax()).toBe(2);
    expect(Array.from(tensor.grad?.data ?? [])).toEqual([0, 0, 1, 0]);
  });

  it("tracks the storage device and exposes CPU data", () => {
    const tensor = Tensor.from([1, 2, 3]);

    expect(tensor.device).toBe("cpu");
    expect(Array.from(tensor.data)).toEqual([1, 2, 3]);
  });

  it("uses the runtime seed for reproducible random tensors", () => {
    Gradlith.manualSeed(42);
    const first = Array.from(Tensor.rand([4]).data);
    Gradlith.manualSeed(42);
    const second = Array.from(Tensor.rand([4]).data);

    expect(second).toEqual(first);
  });

  it("supports async noGrad through the runtime", async () => {
    const input = Tensor.from([1, 2], { requiresGrad: true });
    const output = await noGradAsync(async () => input.mul(2));

    expect(output.requiresGrad).toBe(false);
  });
});
