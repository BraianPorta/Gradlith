import { Tensor } from "../tensor/Tensor";

export type Initializer = "normal" | "xavier" | "he";

export function initializeWeights(input: number, output: number, initializer: Initializer = "he"): Tensor {
  const scale = initializer === "xavier" ? Math.sqrt(2 / (input + output)) : initializer === "he" ? Math.sqrt(2 / input) : 0.1;
  const tensor = Tensor.randn([input, output], { requiresGrad: true, label: "weights" });
  for (let i = 0; i < tensor.data.length; i += 1) {
    tensor.data[i] *= scale;
  }
  return tensor;
}

