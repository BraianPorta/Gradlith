import { Tensor } from "../tensor/Tensor";

export abstract class Module {
  abstract forward(input: Tensor): Tensor;

  parameters(): Tensor[] {
    return [];
  }

  zeroGrad(): void {
    for (const parameter of this.parameters()) {
      parameter.zeroGrad();
    }
  }
}

