import { Tensor } from "../tensor/Tensor";
import { Backend } from "./Backend";

export class CPUBackend implements Backend {
  readonly name = "cpu" as const;

  add(a: Tensor, b: Tensor): Tensor {
    return a.add(b);
  }

  mul(a: Tensor, b: Tensor): Tensor {
    return a.mul(b);
  }

  matmul(a: Tensor, b: Tensor): Tensor {
    return a.matmul(b);
  }

  relu(x: Tensor): Tensor {
    return x.relu();
  }
}

