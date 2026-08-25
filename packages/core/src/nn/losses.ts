import { Tensor } from "../tensor/Tensor";

export type LossFunction = (prediction: Tensor, target: Tensor) => Tensor;

const eps = 1e-7;

export function mse(prediction: Tensor, target: Tensor): Tensor {
  return prediction.sub(target).pow(2).mean();
}

export function binaryCrossEntropy(prediction: Tensor, target: Tensor): Tensor {
  const clipped = clip(prediction, eps, 1 - eps);
  return target.mul(clipped.log()).add(Tensor.scalar(1).sub(target).mul(Tensor.scalar(1).sub(clipped).log())).mean().mul(-1);
}

export function crossEntropy(logits: Tensor, target: Tensor): Tensor {
  if (logits.shape.length !== 2 || target.shape.length !== 2 || logits.shape[0] !== target.shape[0] || logits.shape[1] !== target.shape[1]) {
    throw new Error("crossEntropy expects logits and one-hot targets with shape [batch, classes]");
  }
  const probabilities = logits.softmax();
  return target.mul(probabilities.add(eps).log()).sum().mul(-1 / logits.shape[0]);
}

export function sparseCrossEntropy(logits: Tensor, labels: number[]): Tensor {
  if (logits.shape.length !== 2 || logits.shape[0] !== labels.length) {
    throw new Error("sparseCrossEntropy expects logits [batch, classes] and one label per row");
  }
  const [rows, cols] = logits.shape;
  const oneHot = new Float32Array(rows * cols);
  labels.forEach((label, row) => {
    if (!Number.isInteger(label) || label < 0 || label >= cols) {
      throw new Error(`Invalid class label ${label}`);
    }
    oneHot[row * cols + label] = 1;
  });
  return crossEntropy(logits, new Tensor(oneHot, logits.shape));
}

function clip(input: Tensor, min: number, max: number): Tensor {
  const data = new Float32Array(input.data.length);
  for (let i = 0; i < input.data.length; i += 1) {
    data[i] = Math.min(max, Math.max(min, input.data[i]));
  }
  const out = new Tensor(data, input.shape, { requiresGrad: input.requiresGrad }, [input], () => {
    if (!input.requiresGrad || !out.grad) {
      return;
    }
    const grad = new Float32Array(input.size);
    for (let i = 0; i < input.size; i += 1) {
      grad[i] = input.data[i] >= min && input.data[i] <= max ? out.grad.data[i] : 0;
    }
    input.accumulateGrad(new Tensor(grad, input.shape));
  }, "clip");
  return out;
}
