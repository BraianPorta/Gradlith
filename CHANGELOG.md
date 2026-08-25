# Changelog

## 0.1.0

- Initial monorepo.
- Added TypeScript tensor engine with broadcasting and row-major storage.
- Added reverse-mode autograd with gradient accumulation.
- Added Dense, Tanh, Sigmoid and ReLU modules.
- Added MSE, binary cross entropy and cross entropy losses.
- Added SGD, Momentum, RMSProp and Adam.
- Added synthetic XOR, circles, moons and spiral datasets.
- Added React playground with worker-based training visualization.
- Added CI and GitHub Pages workflow.

## 1.0.0

- Added `max`, `min`, `argmax` and `broadcastTo`.
- Added `Softmax` module and sparse cross entropy helper.
- Expanded finite-difference gradient checks.
- Added WebGPU async kernels for add, ReLU and matmul behind CPU fallback.
- Added real optimizer race training for SGD, Momentum, RMSProp and Adam.
- Added IndexedDB experiment storage.
- Added benchmark, graph, docs and model-builder playground views.
- Added Playwright smoke test configuration.
