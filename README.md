# ∇ GRADLITH

Deep learning from first principles.

Gradlith is a browser-native deep learning framework built from scratch in TypeScript, featuring tensor operations, reverse-mode automatic differentiation, neural networks, optimizers, visualization and WebGPU acceleration.

Gradlith does not use TensorFlow.js, PyTorch, ONNX Runtime, Brain.js or any external machine-learning runtime. Tensor operations, automatic differentiation, neural-network layers and optimizers are implemented from first principles in TypeScript.

## 1.0 Surface

- `@gradlith/core`: multidimensional `Float32Array` tensors, broadcasting, reverse-mode autograd, neural-network modules, losses, optimizers, datasets, diagnostics and CPU/WebGPU backend interfaces.
- `@gradlith/playground`: React/Vite browser lab with decision boundaries, loss chart, gradient inspector, real optimizer race, computational graph view, benchmark runner, IndexedDB experiments and model builder code generation.
- Tests: tensor shape behavior, broadcasting, gradient accumulation, finite-difference gradient checking, matmul backward, optimizers, serialization, WebGPU fallback parity and XOR training.
- Docs: architecture, benchmark methodology, contribution notes and GitHub Actions.

## Quick Start

```bash
pnpm install
pnpm test
pnpm dev
```

The playground runs locally through Vite. The framework code lives in `packages/core/src`.

## API Example

```ts
import { Adam, binaryCrossEntropy, Dense, Sequential, Sigmoid, Tanh, Tensor } from "@gradlith/core";

const X = Tensor.from([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]);

const y = Tensor.from([[0], [1], [1], [0]]);

const model = new Sequential([
  new Dense(2, 8, "xavier"),
  new Tanh(),
  new Dense(8, 1, "xavier"),
  new Sigmoid()
]);

await model.fit(X, y, {
  optimizer: new Adam({ learningRate: 0.05 }),
  loss: binaryCrossEntropy,
  epochs: 1200
});
```

## Roadmap

- Add browser-hosted CPU/WebGPU parity tests for machines with WebGPU enabled.
- Expand the graph debugger into an animated backward-pass timeline.
- Publish benchmark tables from real hardware runs.
- Add drag-and-drop editing to the model builder.

## Playground Routes

- `/playground/`: training dashboard and decision boundary.
- `/optimizers/`: live optimizer comparison.
- `/graph/`: computational graph explorer.
- `/benchmarks/`: CPU/WebGPU benchmark runner.
- `/docs/`: interactive architecture notes.
