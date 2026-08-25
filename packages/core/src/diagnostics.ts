import { Sequential } from "./nn/Sequential";

export interface Diagnostic {
  level: "info" | "warning";
  code: "vanishing-gradient" | "exploding-gradient" | "dead-relu";
  message: string;
}

export function diagnoseGradients(model: Sequential): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  model.parameters().forEach((parameter, index) => {
    if (!parameter.grad) {
      return;
    }
    const norm = Math.sqrt(parameter.grad.data.reduce((total, value) => total + value * value, 0));
    if (norm > 100) {
      diagnostics.push({ level: "warning", code: "exploding-gradient", message: `Parameter ${index} gradient norm is ${norm.toExponential(2)}` });
    }
    if (norm > 0 && norm < 1e-7) {
      diagnostics.push({ level: "warning", code: "vanishing-gradient", message: `Parameter ${index} gradient norm is ${norm.toExponential(2)}` });
    }
  });
  return diagnostics;
}

