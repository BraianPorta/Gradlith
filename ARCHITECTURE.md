# Gradlith Architecture

Gradlith separates the framework from the playground. The web app demonstrates the engine; it is not the engine.

```text
Public TypeScript API
  |
  +-- Runtime
  |     |
  |     +-- backend, grad mode, training mode, seed, profiler state
  |
  +-- Tensor API
  |     |
  |     +-- Shape, strides, broadcasting, autograd metadata
  |     |
  |     +-- TensorStorage
  |           |
  |           +-- CPUStorage / Float32Array
  |           +-- GPUStorage / GPUBuffer contract
  |
  +-- Neural Networks
        |
        +-- Layers, activations, losses, optimizers
              |
              +-- Reverse-mode autograd
                    |
                    +-- Computational graph
                          |
                          +-- CPU backend
                          +-- WebGPU backend interface and WGSL kernels
```

## Runtime

The runtime owns global execution state: selected backend, gradient mode, training mode, deterministic RNG seed and profiler toggles. The public API exposes `Gradlith.setBackend()`, `Gradlith.getBackend()`, `Gradlith.noGrad()`, `Gradlith.noGradAsync()` and `Gradlith.manualSeed()`.

Keeping this state in one place prepares Gradlith for async GPU execution without scattering global flags through tensor, neural-network and optimizer code.

## Tensor Storage

Tensors are now shape + strides + autograd metadata + `TensorStorage`. CPU tensors use row-major `CPUStorage` backed by `Float32Array`. A tensor with shape `[2, 3]` has strides `[3, 1]`, so `tensor[1, 2]` maps to flat offset `1 * 3 + 2 * 1 = 5`.

The storage contract also defines the future `GPUStorage` shape: tensors can eventually point at `GPUBuffer` instead of eagerly returning to CPU memory.

## Broadcasting

Element-wise operations compute a result shape using NumPy-style trailing-dimension broadcasting. During backward propagation, gradients are reduced back to the source tensor shape with `unbroadcast`, so a bias tensor shaped `[features]` can safely accumulate gradients from a batch shaped `[batch, features]`.

## Autograd

Each tensor produced by an operation records:

- parent tensors
- operation name
- a local backward function

Calling `backward()` performs a topological traversal of the graph and then applies local derivatives in reverse order. Gradients are accumulated, not overwritten, so reused tensors receive contributions from every path.

## Neural Networks

`Module` is the base abstraction. `Dense`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax` and `Sequential` compose into trainable models. Parameters are ordinary tensors with `requiresGrad: true`, which means optimizers do not need special layer knowledge.

## Numerical Stability

Softmax subtracts the row maximum before exponentiation. Binary cross entropy clips predictions away from exact `0` and `1`.

## Backends

The backend contract works on storage, not high-level tensors. Tensor methods keep validation, shape logic and autograd wiring, while the selected backend performs numeric execution for unary, binary, reduction, broadcast, transpose, matrix multiplication and softmax operations.

The CPU backend is the trusted implementation. The WebGPU backend exposes support detection, CPU fallback through the same storage contract and native async WGSL execution for `add`, `relu` and rank-2 `matmul` benchmarks. GPU execution is intentionally isolated so parity tests can compare CPU and GPU paths operation by operation.

## Worker Training

The playground runs training in `training.worker.ts`. React receives epoch metrics, boundary samples, graph metadata and gradient summaries through `postMessage`, keeping the interface responsive while the model trains.

## Experiments

Completed runs are stored in IndexedDB with dataset, optimizer, hyperparameters, final metrics, loss history and serialized model JSON. This keeps Gradlith deployable as a static site without requiring a server.
