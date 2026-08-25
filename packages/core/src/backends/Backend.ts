import type { Shape } from "../tensor/shape";
import type { TensorStorage } from "../tensor/storage";

export type BinaryOperation = "add" | "sub" | "mul" | "div";
export type UnaryOperation = "exp" | "log" | "sqrt" | "relu" | "sigmoid" | "tanh";

export interface BinaryOpSpec {
  operation: BinaryOperation;
  outShape: Shape;
  leftShape: Shape;
  rightShape: Shape;
  leftStrides: number[];
  rightStrides: number[];
}

export interface UnaryOpSpec {
  operation: UnaryOperation;
}

export interface BroadcastSpec {
  sourceShape: Shape;
  sourceStrides: number[];
  outShape: Shape;
}

export interface MatMulSpec {
  m: number;
  k: number;
  n: number;
}

export interface Backend {
  readonly name: "cpu" | "webgpu";
  binary(left: TensorStorage, right: TensorStorage, spec: BinaryOpSpec): TensorStorage;
  unary(input: TensorStorage, shape: Shape, spec: UnaryOpSpec): TensorStorage;
  pow(input: TensorStorage, shape: Shape, exponent: number): TensorStorage;
  sum(input: TensorStorage): TensorStorage;
  max(input: TensorStorage): TensorStorage;
  min(input: TensorStorage): TensorStorage;
  broadcastTo(input: TensorStorage, spec: BroadcastSpec): TensorStorage;
  reshape(input: TensorStorage, shape: Shape): TensorStorage;
  transpose(input: TensorStorage, rows: number, cols: number): TensorStorage;
  matmul(left: TensorStorage, right: TensorStorage, spec: MatMulSpec): TensorStorage;
  softmax(input: TensorStorage, rows: number, cols: number): TensorStorage;
  dispose(storage: TensorStorage): void;
}
