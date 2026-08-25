import { Tensor } from "../tensor/Tensor";

export interface Optimizer {
  zeroGrad(parameters: Tensor[]): void;
  step(parameters: Tensor[]): void;
}

export interface OptimizerOptions {
  learningRate?: number;
}

export class SGD implements Optimizer {
  constructor(private readonly options: OptimizerOptions = {}) {}

  zeroGrad(parameters: Tensor[]): void {
    parameters.forEach((parameter) => parameter.zeroGrad());
  }

  step(parameters: Tensor[]): void {
    const learningRate = this.options.learningRate ?? 0.01;
    for (const parameter of parameters) {
      if (!parameter.grad) {
        continue;
      }
      for (let i = 0; i < parameter.data.length; i += 1) {
        parameter.data[i] -= learningRate * parameter.grad.data[i];
      }
    }
  }
}

export class Momentum implements Optimizer {
  private readonly velocity = new WeakMap<Tensor, Float32Array>();

  constructor(private readonly options: OptimizerOptions & { momentum?: number } = {}) {}

  zeroGrad(parameters: Tensor[]): void {
    parameters.forEach((parameter) => parameter.zeroGrad());
  }

  step(parameters: Tensor[]): void {
    const learningRate = this.options.learningRate ?? 0.01;
    const momentum = this.options.momentum ?? 0.9;
    for (const parameter of parameters) {
      if (!parameter.grad) {
        continue;
      }
      const velocity = this.velocity.get(parameter) ?? new Float32Array(parameter.data.length);
      for (let i = 0; i < parameter.data.length; i += 1) {
        velocity[i] = momentum * velocity[i] - learningRate * parameter.grad.data[i];
        parameter.data[i] += velocity[i];
      }
      this.velocity.set(parameter, velocity);
    }
  }
}

export class RMSProp implements Optimizer {
  private readonly average = new WeakMap<Tensor, Float32Array>();

  constructor(private readonly options: OptimizerOptions & { decay?: number; epsilon?: number } = {}) {}

  zeroGrad(parameters: Tensor[]): void {
    parameters.forEach((parameter) => parameter.zeroGrad());
  }

  step(parameters: Tensor[]): void {
    const learningRate = this.options.learningRate ?? 0.001;
    const decay = this.options.decay ?? 0.99;
    const epsilon = this.options.epsilon ?? 1e-8;
    for (const parameter of parameters) {
      if (!parameter.grad) {
        continue;
      }
      const average = this.average.get(parameter) ?? new Float32Array(parameter.data.length);
      for (let i = 0; i < parameter.data.length; i += 1) {
        average[i] = decay * average[i] + (1 - decay) * parameter.grad.data[i] ** 2;
        parameter.data[i] -= learningRate * parameter.grad.data[i] / (Math.sqrt(average[i]) + epsilon);
      }
      this.average.set(parameter, average);
    }
  }
}

export class Adam implements Optimizer {
  private readonly firstMoment = new WeakMap<Tensor, Float32Array>();
  private readonly secondMoment = new WeakMap<Tensor, Float32Array>();
  private stepCount = 0;

  constructor(private readonly options: OptimizerOptions & { beta1?: number; beta2?: number; epsilon?: number } = {}) {}

  zeroGrad(parameters: Tensor[]): void {
    parameters.forEach((parameter) => parameter.zeroGrad());
  }

  step(parameters: Tensor[]): void {
    this.stepCount += 1;
    const learningRate = this.options.learningRate ?? 0.001;
    const beta1 = this.options.beta1 ?? 0.9;
    const beta2 = this.options.beta2 ?? 0.999;
    const epsilon = this.options.epsilon ?? 1e-8;
    for (const parameter of parameters) {
      if (!parameter.grad) {
        continue;
      }
      const m = this.firstMoment.get(parameter) ?? new Float32Array(parameter.data.length);
      const v = this.secondMoment.get(parameter) ?? new Float32Array(parameter.data.length);
      for (let i = 0; i < parameter.data.length; i += 1) {
        const gradient = parameter.grad.data[i];
        m[i] = beta1 * m[i] + (1 - beta1) * gradient;
        v[i] = beta2 * v[i] + (1 - beta2) * gradient * gradient;
        const mHat = m[i] / (1 - beta1 ** this.stepCount);
        const vHat = v[i] / (1 - beta2 ** this.stepCount);
        parameter.data[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
      }
      this.firstMoment.set(parameter, m);
      this.secondMoment.set(parameter, v);
    }
  }
}

