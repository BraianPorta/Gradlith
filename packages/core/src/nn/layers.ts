import { Tensor } from "../tensor/Tensor";
import { Initializer, initializeWeights } from "./initializers";
import { Module } from "./Module";

export class Dense extends Module {
  readonly weights: Tensor;
  readonly bias: Tensor;

  constructor(readonly inputSize: number, readonly outputSize: number, initializer: Initializer = "he") {
    super();
    this.weights = initializeWeights(inputSize, outputSize, initializer);
    this.bias = Tensor.zeros([outputSize], { requiresGrad: true, label: "bias" });
  }

  forward(input: Tensor): Tensor {
    return input.matmul(this.weights).add(this.bias);
  }

  parameters(): Tensor[] {
    return [this.weights, this.bias];
  }

  toJSON(): SerializedLayer {
    return {
      type: "dense",
      input: this.inputSize,
      output: this.outputSize,
      weights: Array.from(this.weights.data),
      bias: Array.from(this.bias.data)
    };
  }
}

export class ReLU extends Module {
  forward(input: Tensor): Tensor {
    return input.relu();
  }

  toJSON(): SerializedLayer {
    return { type: "relu" };
  }
}

export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    return input.sigmoid();
  }

  toJSON(): SerializedLayer {
    return { type: "sigmoid" };
  }
}

export class Tanh extends Module {
  forward(input: Tensor): Tensor {
    return input.tanh();
  }

  toJSON(): SerializedLayer {
    return { type: "tanh" };
  }
}

export class Softmax extends Module {
  forward(input: Tensor): Tensor {
    return input.softmax();
  }

  toJSON(): SerializedLayer {
    return { type: "softmax" };
  }
}

export type SerializedLayer =
  | { type: "dense"; input: number; output: number; weights: number[]; bias: number[] }
  | { type: "relu" }
  | { type: "sigmoid" }
  | { type: "tanh" }
  | { type: "softmax" };

export function layerFromJSON(layer: SerializedLayer): Module {
  if (layer.type === "dense") {
    const dense = new Dense(layer.input, layer.output);
    dense.weights.data.set(layer.weights);
    dense.bias.data.set(layer.bias);
    return dense;
  }
  if (layer.type === "relu") {
    return new ReLU();
  }
  if (layer.type === "sigmoid") {
    return new Sigmoid();
  }
  if (layer.type === "softmax") {
    return new Softmax();
  }
  return new Tanh();
}
