import { broadcastOffset, sizeOf, unravelIndex, type Shape } from "../tensor/shape";
import { assertCPUStorage, CPUStorage, type TensorStorage } from "../tensor/storage";
import type { Backend, BinaryOpSpec, BroadcastSpec, MatMulSpec, UnaryOpSpec } from "./Backend";

export class CPUBackend implements Backend {
  readonly name = "cpu" as const;

  binary(left: TensorStorage, right: TensorStorage, spec: BinaryOpSpec): TensorStorage {
    const leftData = assertCPUStorage(left).data;
    const rightData = assertCPUStorage(right).data;
    const outSize = sizeOf(spec.outShape);
    const data = new Float32Array(outSize);
    for (let i = 0; i < outSize; i += 1) {
      const indices = unravelIndex(i, spec.outShape);
      const a = leftData[broadcastOffset(indices, spec.outShape, spec.leftShape, spec.leftStrides)];
      const b = rightData[broadcastOffset(indices, spec.outShape, spec.rightShape, spec.rightStrides)];
      data[i] = applyBinary(spec.operation, a, b);
    }
    return new CPUStorage(data);
  }

  unary(input: TensorStorage, _shape: Shape, spec: UnaryOpSpec): TensorStorage {
    const inputData = assertCPUStorage(input).data;
    const data = new Float32Array(inputData.length);
    for (let i = 0; i < inputData.length; i += 1) {
      data[i] = applyUnary(spec.operation, inputData[i]);
    }
    return new CPUStorage(data);
  }

  pow(input: TensorStorage, _shape: Shape, exponent: number): TensorStorage {
    const inputData = assertCPUStorage(input).data;
    const data = new Float32Array(inputData.length);
    for (let i = 0; i < inputData.length; i += 1) {
      data[i] = inputData[i] ** exponent;
    }
    return new CPUStorage(data);
  }

  sum(input: TensorStorage): TensorStorage {
    const inputData = assertCPUStorage(input).data;
    let total = 0;
    for (const value of inputData) {
      total += value;
    }
    return new CPUStorage(new Float32Array([total]));
  }

  max(input: TensorStorage): TensorStorage {
    const inputData = assertCPUStorage(input).data;
    let value = -Infinity;
    for (const item of inputData) {
      value = Math.max(value, item);
    }
    return new CPUStorage(new Float32Array([value]));
  }

  min(input: TensorStorage): TensorStorage {
    const inputData = assertCPUStorage(input).data;
    let value = Infinity;
    for (const item of inputData) {
      value = Math.min(value, item);
    }
    return new CPUStorage(new Float32Array([value]));
  }

  broadcastTo(input: TensorStorage, spec: BroadcastSpec): TensorStorage {
    const inputData = assertCPUStorage(input).data;
    const data = new Float32Array(sizeOf(spec.outShape));
    for (let i = 0; i < data.length; i += 1) {
      const indices = unravelIndex(i, spec.outShape);
      data[i] = inputData[broadcastOffset(indices, spec.outShape, spec.sourceShape, spec.sourceStrides)];
    }
    return new CPUStorage(data);
  }

  reshape(input: TensorStorage, _shape: Shape): TensorStorage {
    return input;
  }

  transpose(input: TensorStorage, rows: number, cols: number): TensorStorage {
    const inputData = assertCPUStorage(input).data;
    const data = new Float32Array(inputData.length);
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        data[col * rows + row] = inputData[row * cols + col];
      }
    }
    return new CPUStorage(data);
  }

  matmul(left: TensorStorage, right: TensorStorage, spec: MatMulSpec): TensorStorage {
    const leftData = assertCPUStorage(left).data;
    const rightData = assertCPUStorage(right).data;
    const data = new Float32Array(spec.m * spec.n);
    for (let row = 0; row < spec.m; row += 1) {
      for (let col = 0; col < spec.n; col += 1) {
        let total = 0;
        for (let inner = 0; inner < spec.k; inner += 1) {
          total += leftData[row * spec.k + inner] * rightData[inner * spec.n + col];
        }
        data[row * spec.n + col] = total;
      }
    }
    return new CPUStorage(data);
  }

  softmax(input: TensorStorage, rows: number, cols: number): TensorStorage {
    const inputData = assertCPUStorage(input).data;
    const data = new Float32Array(inputData.length);
    for (let row = 0; row < rows; row += 1) {
      let max = -Infinity;
      for (let col = 0; col < cols; col += 1) {
        max = Math.max(max, inputData[row * cols + col]);
      }
      let total = 0;
      for (let col = 0; col < cols; col += 1) {
        const value = Math.exp(inputData[row * cols + col] - max);
        data[row * cols + col] = value;
        total += value;
      }
      for (let col = 0; col < cols; col += 1) {
        data[row * cols + col] /= total;
      }
    }
    return new CPUStorage(data);
  }

  dispose(_storage: TensorStorage): void {
    // CPU memory is managed by the JavaScript runtime.
  }
}

function applyBinary(operation: BinaryOpSpec["operation"], a: number, b: number): number {
  switch (operation) {
    case "add":
      return a + b;
    case "sub":
      return a - b;
    case "mul":
      return a * b;
    case "div":
      return a / b;
  }
}

function applyUnary(operation: UnaryOpSpec["operation"], value: number): number {
  switch (operation) {
    case "exp":
      return Math.exp(value);
    case "log":
      return Math.log(value);
    case "sqrt":
      return Math.sqrt(value);
    case "relu":
      return Math.max(0, value);
    case "sigmoid":
      return 1 / (1 + Math.exp(-value));
    case "tanh":
      return Math.tanh(value);
  }
}
