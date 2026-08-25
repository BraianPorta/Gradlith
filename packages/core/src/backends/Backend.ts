import { Tensor } from "../tensor/Tensor";

export interface Backend {
  readonly name: "cpu" | "webgpu";
  add(a: Tensor, b: Tensor): Tensor;
  mul(a: Tensor, b: Tensor): Tensor;
  matmul(a: Tensor, b: Tensor): Tensor;
  relu(x: Tensor): Tensor;
}

