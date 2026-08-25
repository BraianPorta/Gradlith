# Benchmarks

Gradlith does not publish invented performance claims.

Use this methodology when collecting browser results:

- Hardware: CPU, GPU and memory.
- Browser: name, version and WebGPU status.
- Commit: exact Gradlith commit hash.
- Warm-up: 10 iterations.
- Measurement: 100 iterations unless the tensor size makes that impractical.
- Dtype: `float32`.
- Operations: element-wise add, ReLU, matrix multiplication, forward pass, backward pass and training step.
- Sizes: `128`, `256`, `512`, `1024` where practical.

## Browser Runner

The playground includes a `/benchmarks/` view that measures CPU execution and WebGPU async kernels when available in the current browser. Results are measured at runtime on the user's device.

## Results

No official benchmark table has been committed yet. Do not add numbers here unless they were collected with the methodology above.
