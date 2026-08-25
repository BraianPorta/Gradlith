import { describe, expect, it } from "vitest";
import { SGD, Tensor } from "../index";

describe("Optimizers", () => {
  it("moves parameters opposite their gradients", () => {
    const x = Tensor.scalar(2, { requiresGrad: true });
    x.pow(2).backward();

    new SGD({ learningRate: 0.1 }).step([x]);

    expect(x.item()).toBeCloseTo(1.6, 5);
  });
});

