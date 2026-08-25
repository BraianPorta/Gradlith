import {
  assertShape,
  broadcastOffset,
  broadcastShapes,
  flatten,
  inferShape,
  ravelIndex,
  sameShape,
  Shape,
  sizeOf,
  stridesOf,
  unbroadcast,
  unravelIndex
} from "./shape";

type BackwardFn = () => void;

export interface TensorOptions {
  requiresGrad?: boolean;
  label?: string;
}

export class Tensor {
  readonly data: Float32Array;
  readonly shape: Shape;
  readonly strides: number[];
  readonly requiresGrad: boolean;
  readonly label?: string;
  grad?: Tensor;
  private parents: Tensor[];
  private backwardFn?: BackwardFn;
  private operation?: string;

  constructor(data: Float32Array | number[], shape: Shape = [data.length], options: TensorOptions = {}, parents: Tensor[] = [], backwardFn?: BackwardFn, operation?: string) {
    assertShape(shape);
    if (sizeOf(shape) !== data.length) {
      throw new Error(`Data length ${data.length} does not match shape [${shape.join(", ")}]`);
    }
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = [...shape];
    this.strides = stridesOf(shape);
    this.requiresGrad = options.requiresGrad ?? false;
    this.label = options.label;
    this.parents = parents;
    this.backwardFn = backwardFn;
    this.operation = operation;
  }

  get size(): number {
    return this.data.length;
  }

  static scalar(value: number, options: TensorOptions = {}): Tensor {
    return new Tensor(new Float32Array([value]), [], options);
  }

  static from(value: number | unknown[], options: TensorOptions = {}): Tensor {
    if (typeof value === "number") {
      return Tensor.scalar(value, options);
    }
    const shape = inferShape(value);
    return new Tensor(new Float32Array(flatten(value)), shape, options);
  }

  static zeros(shape: Shape, options: TensorOptions = {}): Tensor {
    return new Tensor(new Float32Array(sizeOf(shape)), shape, options);
  }

  static ones(shape: Shape, options: TensorOptions = {}): Tensor {
    return new Tensor(new Float32Array(sizeOf(shape)).fill(1), shape, options);
  }

  static onesLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
    return Tensor.ones(tensor.shape, options);
  }

  static rand(shape: Shape, options: TensorOptions = {}): Tensor {
    const data = new Float32Array(sizeOf(shape));
    for (let i = 0; i < data.length; i += 1) {
      data[i] = Math.random();
    }
    return new Tensor(data, shape, options);
  }

  static randn(shape: Shape, options: TensorOptions = {}): Tensor {
    const data = new Float32Array(sizeOf(shape));
    for (let i = 0; i < data.length; i += 2) {
      const u = 1 - Math.random();
      const v = Math.random();
      const radius = Math.sqrt(-2 * Math.log(u));
      data[i] = radius * Math.cos(2 * Math.PI * v);
      if (i + 1 < data.length) {
        data[i + 1] = radius * Math.sin(2 * Math.PI * v);
      }
    }
    return new Tensor(data, shape, options);
  }

  clone(options: TensorOptions = {}): Tensor {
    return new Tensor(new Float32Array(this.data), this.shape, {
      requiresGrad: options.requiresGrad ?? this.requiresGrad,
      label: options.label ?? this.label
    });
  }

  item(): number {
    if (this.data.length !== 1) {
      throw new Error("item() is only valid for scalar tensors");
    }
    return this.data[0];
  }

  toArray(): unknown {
    const build = (shape: Shape, offset: number): unknown => {
      if (shape.length === 0) {
        return this.data[offset];
      }
      const [head, ...tail] = shape;
      return Array.from({ length: head }, (_, index) => build(tail, offset + index * this.strides[this.shape.length - shape.length]));
    };
    return build(this.shape, 0);
  }

  zeroGrad(): void {
    this.grad = undefined;
  }

  add(other: Tensor | number): Tensor {
    return binaryOp(this, ensureTensor(other), "add", (a, b) => a + b, (grad) => [grad, grad]);
  }

  sub(other: Tensor | number): Tensor {
    return binaryOp(this, ensureTensor(other), "sub", (a, b) => a - b, (grad) => [grad, mapData(grad, (value) => -value)]);
  }

  mul(other: Tensor | number): Tensor {
    const rhs = ensureTensor(other);
    return binaryOp(this, rhs, "mul", (a, b) => a * b, (grad, a, b) => [multiplyData(grad, b), multiplyData(grad, a)]);
  }

  div(other: Tensor | number): Tensor {
    const rhs = ensureTensor(other);
    return binaryOp(this, rhs, "div", (a, b) => a / b, (grad, a, b) => [
      divideData(grad, b),
      multiplyData(mapData(grad, (value) => -value), divideData(a, multiplyData(b, b)))
    ]);
  }

  pow(exponent: number): Tensor {
    const outData = mapData(this.data, (value) => value ** exponent);
    const out = new Tensor(outData, this.shape, { requiresGrad: this.requiresGrad }, [this], () => {
      if (!this.requiresGrad || !out.grad) {
        return;
      }
      const local = mapData(this.data, (value) => exponent * value ** (exponent - 1));
      this.accumulateGrad(new Tensor(multiplyData(out.grad.data, local), this.shape));
    }, "pow");
    return out;
  }

  exp(): Tensor {
    return unaryOp(this, "exp", Math.exp, (out) => out);
  }

  log(): Tensor {
    return unaryOp(this, "log", Math.log, (_, input) => mapData(input, (value) => 1 / value));
  }

  sqrt(): Tensor {
    return unaryOp(this, "sqrt", Math.sqrt, (out) => mapData(out, (value) => 0.5 / value));
  }

  relu(): Tensor {
    return unaryOp(this, "relu", (value) => Math.max(0, value), (_, input) => mapData(input, (value) => (value > 0 ? 1 : 0)));
  }

  sigmoid(): Tensor {
    return unaryOp(this, "sigmoid", (value) => 1 / (1 + Math.exp(-value)), (out) => mapData(out, (value) => value * (1 - value)));
  }

  tanh(): Tensor {
    return unaryOp(this, "tanh", Math.tanh, (out) => mapData(out, (value) => 1 - value * value));
  }

  sum(): Tensor {
    const value = this.data.reduce((total, next) => total + next, 0);
    const out = Tensor.scalar(value, { requiresGrad: this.requiresGrad });
    out.setGraph([this], () => {
      if (this.requiresGrad && out.grad) {
        this.accumulateGrad(new Tensor(new Float32Array(this.size).fill(out.grad.data[0]), this.shape));
      }
    }, "sum");
    return out;
  }

  mean(): Tensor {
    return this.sum().div(this.size);
  }

  max(): Tensor {
    const value = this.data.reduce((current, next) => Math.max(current, next), -Infinity);
    const out = Tensor.scalar(value, { requiresGrad: this.requiresGrad });
    out.setGraph([this], () => {
      if (!this.requiresGrad || !out.grad) {
        return;
      }
      let matches = 0;
      for (const item of this.data) {
        if (item === value) {
          matches += 1;
        }
      }
      const grad = new Float32Array(this.size);
      for (let i = 0; i < this.size; i += 1) {
        grad[i] = this.data[i] === value ? out.grad.data[0] / matches : 0;
      }
      this.accumulateGrad(new Tensor(grad, this.shape));
    }, "max");
    return out;
  }

  min(): Tensor {
    const value = this.data.reduce((current, next) => Math.min(current, next), Infinity);
    const out = Tensor.scalar(value, { requiresGrad: this.requiresGrad });
    out.setGraph([this], () => {
      if (!this.requiresGrad || !out.grad) {
        return;
      }
      let matches = 0;
      for (const item of this.data) {
        if (item === value) {
          matches += 1;
        }
      }
      const grad = new Float32Array(this.size);
      for (let i = 0; i < this.size; i += 1) {
        grad[i] = this.data[i] === value ? out.grad.data[0] / matches : 0;
      }
      this.accumulateGrad(new Tensor(grad, this.shape));
    }, "min");
    return out;
  }

  argmax(): number {
    let bestIndex = 0;
    for (let i = 1; i < this.data.length; i += 1) {
      if (this.data[i] > this.data[bestIndex]) {
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  broadcastTo(shape: Shape): Tensor {
    const outShape = broadcastShapes(this.shape, shape);
    if (!sameShape(outShape, shape)) {
      throw new Error(`Cannot broadcast [${this.shape.join(", ")}] to [${shape.join(", ")}]`);
    }
    const data = new Float32Array(sizeOf(shape));
    for (let i = 0; i < data.length; i += 1) {
      const indices = unravelIndex(i, shape);
      data[i] = this.data[broadcastOffset(indices, shape, this.shape, this.strides)];
    }
    const out = new Tensor(data, shape, { requiresGrad: this.requiresGrad }, [this], () => {
      if (this.requiresGrad && out.grad) {
        this.accumulateGrad(new Tensor(unbroadcast(out.grad.data, shape, this.shape), this.shape));
      }
    }, "broadcastTo");
    return out;
  }

  reshape(shape: Shape): Tensor {
    if (sizeOf(shape) !== this.size) {
      throw new Error(`Cannot reshape [${this.shape.join(", ")}] to [${shape.join(", ")}]`);
    }
    const out = new Tensor(this.data, shape, { requiresGrad: this.requiresGrad }, [this], () => {
      if (this.requiresGrad && out.grad) {
        this.accumulateGrad(new Tensor(out.grad.data, this.shape));
      }
    }, "reshape");
    return out;
  }

  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("transpose() currently supports rank-2 tensors");
    }
    const [rows, cols] = this.shape;
    const data = new Float32Array(this.size);
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        data[col * rows + row] = this.data[row * cols + col];
      }
    }
    const out = new Tensor(data, [cols, rows], { requiresGrad: this.requiresGrad }, [this], () => {
      if (this.requiresGrad && out.grad) {
        this.accumulateGrad(out.grad.transpose());
      }
    }, "transpose");
    return out;
  }

  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error("matmul() currently supports rank-2 tensors");
    }
    const [m, k] = this.shape;
    const [k2, n] = other.shape;
    if (k !== k2) {
      throw new Error(`matmul shape mismatch [${this.shape.join(", ")}] x [${other.shape.join(", ")}]`);
    }
    const data = new Float32Array(m * n);
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < n; col += 1) {
        let total = 0;
        for (let inner = 0; inner < k; inner += 1) {
          total += this.data[row * k + inner] * other.data[inner * n + col];
        }
        data[row * n + col] = total;
      }
    }
    const out = new Tensor(data, [m, n], { requiresGrad: this.requiresGrad || other.requiresGrad }, [this, other], () => {
      if (!out.grad) {
        return;
      }
      if (this.requiresGrad) {
        this.accumulateGrad(out.grad.matmul(other.transpose()));
      }
      if (other.requiresGrad) {
        other.accumulateGrad(this.transpose().matmul(out.grad));
      }
    }, "matmul");
    return out;
  }

  softmax(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("softmax() currently expects [batch, classes]");
    }
    const [rows, cols] = this.shape;
    const data = new Float32Array(this.size);
    for (let row = 0; row < rows; row += 1) {
      let max = -Infinity;
      for (let col = 0; col < cols; col += 1) {
        max = Math.max(max, this.data[row * cols + col]);
      }
      let total = 0;
      for (let col = 0; col < cols; col += 1) {
        const value = Math.exp(this.data[row * cols + col] - max);
        data[row * cols + col] = value;
        total += value;
      }
      for (let col = 0; col < cols; col += 1) {
        data[row * cols + col] /= total;
      }
    }
    const out = new Tensor(data, this.shape, { requiresGrad: this.requiresGrad }, [this], () => {
      if (!this.requiresGrad || !out.grad) {
        return;
      }
      const grad = new Float32Array(this.size);
      for (let row = 0; row < rows; row += 1) {
        let dot = 0;
        for (let col = 0; col < cols; col += 1) {
          dot += out.grad.data[row * cols + col] * out.data[row * cols + col];
        }
        for (let col = 0; col < cols; col += 1) {
          const index = row * cols + col;
          grad[index] = out.data[index] * (out.grad.data[index] - dot);
        }
      }
      this.accumulateGrad(new Tensor(grad, this.shape));
    }, "softmax");
    return out;
  }

  backward(gradient?: Tensor): void {
    if (this.size !== 1 && !gradient) {
      throw new Error("backward() on non-scalar tensors requires an explicit gradient");
    }
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const visit = (tensor: Tensor) => {
      if (visited.has(tensor)) {
        return;
      }
      visited.add(tensor);
      for (const parent of tensor.parents) {
        visit(parent);
      }
      topo.push(tensor);
    };
    visit(this);
    this.grad = gradient ?? Tensor.onesLike(this);
    for (let i = topo.length - 1; i >= 0; i -= 1) {
      topo[i].backwardFn?.();
    }
  }

  graph(): Array<{ id: number; label: string; operation: string; parents: number[]; shape: Shape }> {
    const nodes: Tensor[] = [];
    const visited = new Set<Tensor>();
    const visit = (tensor: Tensor) => {
      if (visited.has(tensor)) {
        return;
      }
      visited.add(tensor);
      for (const parent of tensor.parents) {
        visit(parent);
      }
      nodes.push(tensor);
    };
    visit(this);
    return nodes.map((node, id) => ({
      id,
      label: node.label ?? `t${id}`,
      operation: node.operation ?? "leaf",
      parents: node.parents.map((parent) => nodes.indexOf(parent)),
      shape: node.shape
    }));
  }

  accumulateGrad(incoming: Tensor): void {
    if (!sameShape(incoming.shape, this.shape)) {
      throw new Error(`Gradient shape mismatch [${incoming.shape.join(", ")}] for [${this.shape.join(", ")}]`);
    }
    if (!this.grad) {
      this.grad = incoming.clone();
      return;
    }
    for (let i = 0; i < this.grad.data.length; i += 1) {
      this.grad.data[i] += incoming.data[i];
    }
  }

  private setGraph(parents: Tensor[], backwardFn: BackwardFn, operation: string): void {
    this.parents = parents;
    this.backwardFn = backwardFn;
    this.operation = operation;
  }
}

function ensureTensor(value: Tensor | number): Tensor {
  return value instanceof Tensor ? value : Tensor.scalar(value);
}

function unaryOp(input: Tensor, operation: string, forward: (value: number) => number, derivative: (out: Float32Array, input: Float32Array) => Float32Array): Tensor {
  const outData = mapData(input.data, forward);
  const out = new Tensor(outData, input.shape, { requiresGrad: input.requiresGrad }, [input], () => {
    if (!input.requiresGrad || !out.grad) {
      return;
    }
    input.accumulateGrad(new Tensor(multiplyData(out.grad.data, derivative(out.data, input.data)), input.shape));
  }, operation);
  return out;
}

function binaryOp(
  left: Tensor,
  right: Tensor,
  operation: string,
  forward: (a: number, b: number) => number,
  backward: (grad: Float32Array, a: Float32Array, b: Float32Array) => [Float32Array, Float32Array]
): Tensor {
  const outShape = broadcastShapes(left.shape, right.shape);
  const outSize = sizeOf(outShape);
  const data = new Float32Array(outSize);
  const expandedLeft = new Float32Array(outSize);
  const expandedRight = new Float32Array(outSize);
  for (let i = 0; i < outSize; i += 1) {
    const indices = unravelIndex(i, outShape);
    const leftValue = left.data[broadcastOffset(indices, outShape, left.shape, left.strides)];
    const rightValue = right.data[broadcastOffset(indices, outShape, right.shape, right.strides)];
    expandedLeft[i] = leftValue;
    expandedRight[i] = rightValue;
    data[i] = forward(leftValue, rightValue);
  }
  const out = new Tensor(data, outShape, { requiresGrad: left.requiresGrad || right.requiresGrad }, [left, right], () => {
    if (!out.grad) {
      return;
    }
    const [leftGrad, rightGrad] = backward(out.grad.data, expandedLeft, expandedRight);
    if (left.requiresGrad) {
      left.accumulateGrad(new Tensor(unbroadcast(leftGrad, outShape, left.shape), left.shape));
    }
    if (right.requiresGrad) {
      right.accumulateGrad(new Tensor(unbroadcast(rightGrad, outShape, right.shape), right.shape));
    }
  }, operation);
  return out;
}

function mapData(data: Float32Array, fn: (value: number) => number): Float32Array {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) {
    out[i] = fn(data[i]);
  }
  return out;
}

function multiplyData(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    out[i] = a[i] * b[i];
  }
  return out;
}

function divideData(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    out[i] = a[i] / b[i];
  }
  return out;
}

export function tensorAt(tensor: Tensor, indices: number[]): number {
  if (indices.length !== tensor.shape.length) {
    throw new Error("Incorrect index rank");
  }
  return tensor.data[ravelIndex(indices, tensor.strides)];
}
