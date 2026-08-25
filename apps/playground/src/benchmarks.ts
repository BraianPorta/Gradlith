import { CPUBackend, Tensor, WebGPUBackend } from "@gradlith/core";

export interface BenchmarkResult {
  operation: string;
  size: string;
  cpuMs: number;
  gpuMs?: number;
}

export async function runBenchmarks(): Promise<BenchmarkResult[]> {
  const cpu = new CPUBackend();
  const gpu = new WebGPUBackend();
  const results: BenchmarkResult[] = [];
  for (const size of [64, 128, 192]) {
    const a = Tensor.rand([size, size]);
    const b = Tensor.rand([size, size]);
    results.push({
      operation: "matmul",
      size: `${size}x${size}`,
      cpuMs: measure(() => cpu.matmul(a, b)),
      gpuMs: WebGPUBackend.isSupported() ? await measureAsync(() => gpu.matmulAsync(a, b)) : undefined
    });
    results.push({
      operation: "relu",
      size: `${size * size}`,
      cpuMs: measure(() => cpu.relu(a)),
      gpuMs: WebGPUBackend.isSupported() ? await measureAsync(() => gpu.reluAsync(a)) : undefined
    });
    results.push({
      operation: "add",
      size: `${size * size}`,
      cpuMs: measure(() => cpu.add(a, b)),
      gpuMs: WebGPUBackend.isSupported() ? await measureAsync(() => gpu.addAsync(a, b)) : undefined
    });
  }
  return results;
}

function measure(fn: () => unknown): number {
  fn();
  const start = performance.now();
  for (let i = 0; i < 10; i += 1) {
    fn();
  }
  return (performance.now() - start) / 10;
}

async function measureAsync(fn: () => Promise<unknown>): Promise<number> {
  await fn();
  const start = performance.now();
  for (let i = 0; i < 5; i += 1) {
    await fn();
  }
  return (performance.now() - start) / 5;
}

