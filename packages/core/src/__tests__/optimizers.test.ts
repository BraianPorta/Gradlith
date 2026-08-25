import { describe, expect, it } from "vitest";
import { Adam, Momentum, RMSProp, SGD, Tensor } from "../index";

describe("Optimizers", () => {
  it("moves parameters opposite their gradients", () => {
    const x = Tensor.scalar(2, { requiresGrad: true });
    x.pow(2).backward();

    new SGD({ learningRate: 0.1 }).step([x]);

    expect(x.item()).toBeCloseTo(1.6, 5);
  });

  it("applies a Momentum reference step", () => {
    const x = Tensor.from([1, -2]);
    x.grad = Tensor.from([0.5, -0.25]);

    new Momentum({ learningRate: 0.1, momentum: 0.9 }).step([x]);

    expect(x.data[0]).toBeCloseTo(0.95, 5);
    expect(x.data[1]).toBeCloseTo(-1.975, 5);
  });

  it("applies an RMSProp reference step", () => {
    const x = Tensor.from([1, -2]);
    x.grad = Tensor.from([0.5, -0.25]);

    new RMSProp({ learningRate: 0.1, decay: 0.9, epsilon: 1e-8 }).step([x]);

    expect(x.data[0]).toBeCloseTo(0.6837722, 5);
    expect(x.data[1]).toBeCloseTo(-1.6837722, 5);
  });

  it("applies an Adam reference step", () => {
    const x = Tensor.from([1, -2]);
    x.grad = Tensor.from([0.5, -0.25]);

    new Adam({ learningRate: 0.1, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).step([x]);

    expect(x.data[0]).toBeCloseTo(0.9, 5);
    expect(x.data[1]).toBeCloseTo(-1.9, 5);
  });
});
