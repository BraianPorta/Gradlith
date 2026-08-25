import { describe, expect, it } from "vitest";
import { Adam, binaryCrossEntropy, Dense, Sequential, Sigmoid, Softmax, Tanh, Tensor, xorDataset } from "../index";

describe("Neural networks", () => {
  it("trains XOR with first-principles autograd", async () => {
    const originalRandom = Math.random;
    let seed = 1337;
    Math.random = () => {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    };

    try {
      const dataset = xorDataset();
      const model = new Sequential([
        new Dense(2, 8, "xavier"),
        new Tanh(),
        new Dense(8, 1, "xavier"),
        new Sigmoid()
      ]);

      await model.fit(dataset.x, dataset.y, {
        epochs: 1200,
        optimizer: new Adam({ learningRate: 0.05 }),
        loss: binaryCrossEntropy,
        batchSize: 4
      });

      const predictions = model.forward(dataset.x);
      const classes = Array.from(predictions.data).map((value) => (value >= 0.5 ? 1 : 0));

      expect(classes).toEqual([0, 1, 1, 0]);
    } finally {
      Math.random = originalRandom;
    }
  });

  it("serializes and reloads models", () => {
    const model = new Sequential([new Dense(2, 3), new Tanh(), new Dense(3, 1)]);
    const loaded = Sequential.load(model.save());
    const output = loaded.forward(Tensor.from([[1, 2]]));

    expect(output.shape).toEqual([1, 1]);
  });

  it("exposes Softmax as a module", () => {
    const output = new Softmax().forward(Tensor.from([[1, 2, 3]]));
    const total = Array.from(output.data).reduce((sum, value) => sum + value, 0);

    expect(total).toBeCloseTo(1, 5);
    expect(output.argmax()).toBe(2);
  });

  it("gradient-checks a Dense layer parameter", () => {
    const model = new Sequential([new Dense(2, 1, "xavier"), new Sigmoid()]);
    const input = Tensor.from([[0.3, -0.7]]);
    const target = Tensor.from([[1]]);
    const weight = model.parameters()[0];
    const loss = binaryCrossEntropy(model.forward(input), target);
    loss.backward();
    const epsilon = 1e-3;
    const original = weight.data[0];

    weight.data[0] = original + epsilon;
    const plus = binaryCrossEntropy(model.forward(input), target).item();
    weight.data[0] = original - epsilon;
    const minus = binaryCrossEntropy(model.forward(input), target).item();
    weight.data[0] = original;

    expect(weight.grad?.data[0]).toBeCloseTo((plus - minus) / (2 * epsilon), 2);
  });
});
