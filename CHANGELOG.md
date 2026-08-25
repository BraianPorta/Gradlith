# Changelog

## 2.0.0

- Added a `Runtime` singleton with backend selection, gradient mode, async `noGrad`, profiler toggles and deterministic seeding.
- Added `TensorStorage`, `CPUStorage` and device metadata so tensors are no longer modeled only as raw CPU arrays.
- Moved CPU numerical execution behind the backend contract for unary, binary, reduction, broadcast, reshape, transpose, matmul and softmax operations.
- Updated WebGPU to the new backend contract while keeping current async WGSL benchmark kernels.
- Added public `Gradlith`, `setBackend`, `getBackend`, `manualSeed` and storage exports.
- Added runtime, storage and deterministic random tests.

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
