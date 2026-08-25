import { describe, expect, it } from "vitest";
import { Tensor } from "../index";

function close(actual: number, expected: number, tolerance = 1e-4) {
  expect(Math.abs(actual - expected)).toBeLessThan(tolerance);
}

describe("Autograd", () => {
  it("differentiates a scalar polynomial", () => {
    const x = Tensor.scalar(3, { requiresGrad: true });
    const y = x.pow(2).add(x.mul(4)).add(7);

    y.backward();

    close(x.grad?.item() ?? NaN, 10);
  });

  it("accumulates gradients through multiple paths", () => {
    const x = Tensor.scalar(2, { requiresGrad: true });
    const y = x.mul(x).add(x);

    y.backward();

    close(x.grad?.item() ?? NaN, 5);
  });

  it("backpropagates through matmul", () => {
    const a = Tensor.from([[1, 2]], { requiresGrad: true });
    const b = Tensor.from([[3], [4]], { requiresGrad: true });

    a.matmul(b).sum().backward();

    expect(Array.from(a.grad?.data ?? [])).toEqual([3, 4]);
    expect(Array.from(b.grad?.data ?? [])).toEqual([1, 2]);
  });

  it("matches finite differences for a mixed expression", () => {
    const x = Tensor.from([[0.2, -0.4, 0.7]], { requiresGrad: true });
    const w = Tensor.from([[0.5], [-0.25], [0.1]], { requiresGrad: true });
    const loss = x.matmul(w).tanh().pow(2).mean();
    loss.backward();

    const epsilon = 1e-3;
    for (let i = 0; i < w.data.length; i += 1) {
      const plus = w.clone({ requiresGrad: true });
      const minus = w.clone({ requiresGrad: true });
      plus.data[i] += epsilon;
      minus.data[i] -= epsilon;
      const numerical = (x.matmul(plus).tanh().pow(2).mean().item() - x.matmul(minus).tanh().pow(2).mean().item()) / (2 * epsilon);
      close(w.grad?.data[i] ?? NaN, numerical, 2e-3);
    }
  });

  it.each([
    ["add", (x: Tensor) => x.add(2).mean()],
    ["mul", (x: Tensor) => x.mul(x).mean()],
    ["pow", (x: Tensor) => x.pow(3).mean()],
    ["relu", (x: Tensor) => x.relu().mean()],
    ["sigmoid", (x: Tensor) => x.sigmoid().mean()],
    ["mean", (x: Tensor) => x.mean()]
  ])("passes scalar finite-difference gradient check for %s", (_, fn) => {
    const x = Tensor.from([0.35, -0.2, 0.9], { requiresGrad: true });
    fn(x).backward();
    const epsilon = 1e-3;

    for (let i = 0; i < x.data.length; i += 1) {
      const plus = x.clone({ requiresGrad: true });
      const minus = x.clone({ requiresGrad: true });
      plus.data[i] += epsilon;
      minus.data[i] -= epsilon;
      const numerical = (fn(plus).item() - fn(minus).item()) / (2 * epsilon);

      close(x.grad?.data[i] ?? NaN, numerical, 3e-3);
    }
  });
});
