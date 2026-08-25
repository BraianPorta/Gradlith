import { isGradEnabled, Tensor } from "../tensor/Tensor";

export type LossFunction = (prediction: Tensor, target: Tensor) => Tensor;

const eps = 1e-7;

export function mse(prediction: Tensor, target: Tensor): Tensor {
  return prediction.sub(target).pow(2).mean();
}

export function binaryCrossEntropy(prediction: Tensor, target: Tensor): Tensor {
  const clipped = clip(prediction, eps, 1 - eps);
  return target.mul(clipped.log()).add(Tensor.scalar(1).sub(target).mul(Tensor.scalar(1).sub(clipped).log())).mean().mul(-1);
}

export function binaryCrossEntropyWithLogits(logits: Tensor, target: Tensor): Tensor {
  if (logits.shape.length !== target.shape.length || logits.shape.some((dim, index) => dim !== target.shape[index])) {
    throw new Error("binaryCrossEntropyWithLogits expects logits and targets with the same shape");
  }
  let total = 0;
  const grad = new Float32Array(logits.size);
  for (let i = 0; i < logits.size; i += 1) {
    const logit = logits.data[i];
    const label = target.data[i];
    total += Math.max(logit, 0) - logit * label + Math.log1p(Math.exp(-Math.abs(logit)));
    grad[i] = (sigmoidNumber(logit) - label) / logits.size;
  }
  const requiresGrad = isGradEnabled() && logits.requiresGrad;
  let out: Tensor;
  out = new Tensor(new Float32Array([total / logits.size]), [], { requiresGrad }, requiresGrad ? [logits] : [], requiresGrad ? () => {
    if (!out.grad) {
      return;
    }
    const scaled = new Float32Array(grad.length);
    for (let i = 0; i < grad.length; i += 1) {
      scaled[i] = grad[i] * out.grad.data[0];
    }
    logits.accumulateGrad(new Tensor(scaled, logits.shape));
  } : undefined, requiresGrad ? "binaryCrossEntropyWithLogits" : undefined);
  return out;
}

export function crossEntropy(logits: Tensor, target: Tensor): Tensor {
  if (logits.shape.length !== 2 || target.shape.length !== 2 || logits.shape[0] !== target.shape[0] || logits.shape[1] !== target.shape[1]) {
    throw new Error("crossEntropy expects logits and one-hot targets with shape [batch, classes]");
  }
  const probabilities = logits.softmax();
  return target.mul(probabilities.add(eps).log()).sum().mul(-1 / logits.shape[0]);
}

export function crossEntropyWithLogits(logits: Tensor, target: Tensor): Tensor {
  if (logits.shape.length !== 2 || target.shape.length !== 2 || logits.shape[0] !== target.shape[0] || logits.shape[1] !== target.shape[1]) {
    throw new Error("crossEntropyWithLogits expects logits and one-hot targets with shape [batch, classes]");
  }
  const [rows, cols] = logits.shape;
  const grad = new Float32Array(logits.size);
  let total = 0;
  for (let row = 0; row < rows; row += 1) {
    let max = -Infinity;
    for (let col = 0; col < cols; col += 1) {
      max = Math.max(max, logits.data[row * cols + col]);
    }
    let expTotal = 0;
    for (let col = 0; col < cols; col += 1) {
      expTotal += Math.exp(logits.data[row * cols + col] - max);
    }
    const logSumExp = max + Math.log(expTotal);
    for (let col = 0; col < cols; col += 1) {
      const index = row * cols + col;
      total -= target.data[index] * logits.data[index];
      grad[index] = (Math.exp(logits.data[index] - logSumExp) - target.data[index]) / rows;
    }
    total += logSumExp;
  }
  const requiresGrad = isGradEnabled() && logits.requiresGrad;
  let out: Tensor;
  out = new Tensor(new Float32Array([total / rows]), [], { requiresGrad }, requiresGrad ? [logits] : [], requiresGrad ? () => {
    if (!out.grad) {
      return;
    }
    const scaled = new Float32Array(grad.length);
    for (let i = 0; i < grad.length; i += 1) {
      scaled[i] = grad[i] * out.grad.data[0];
    }
    logits.accumulateGrad(new Tensor(scaled, logits.shape));
  } : undefined, requiresGrad ? "crossEntropyWithLogits" : undefined);
  return out;
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

function sigmoidNumber(value: number): number {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function clip(input: Tensor, min: number, max: number): Tensor {
  const data = new Float32Array(input.data.length);
  for (let i = 0; i < input.data.length; i += 1) {
    data[i] = Math.min(max, Math.max(min, input.data[i]));
  }
  const requiresGrad = isGradEnabled() && input.requiresGrad;
  const out = new Tensor(data, input.shape, { requiresGrad }, requiresGrad ? [input] : [], requiresGrad ? () => {
    if (!input.requiresGrad || !out.grad) {
      return;
    }
    const grad = new Float32Array(input.size);
    for (let i = 0; i < input.size; i += 1) {
      grad[i] = input.data[i] >= min && input.data[i] <= max ? out.grad.data[i] : 0;
    }
    input.accumulateGrad(new Tensor(grad, input.shape));
  } : undefined, requiresGrad ? "clip" : undefined);
  return out;
}
