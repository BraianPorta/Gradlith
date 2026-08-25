import type { Backend } from "./backends/Backend";
import { CPUBackend } from "./backends/CPUBackend";

export type BackendName = "cpu" | "webgpu";

export class Runtime {
  backend: Backend = new CPUBackend();
  gradEnabled = true;
  training = true;
  profilerEnabled = false;
  seed?: number;
  private seedState?: number;

  async setBackend(backend: BackendName | Backend): Promise<Backend> {
    if (typeof backend !== "string") {
      this.backend = backend;
      return this.backend;
    }
    if (backend === "cpu") {
      this.backend = new CPUBackend();
      return this.backend;
    }
    const { WebGPUBackend } = await import("./backends/WebGPUBackend");
    this.backend = new WebGPUBackend();
    return this.backend;
  }

  getBackend(): BackendName {
    return this.backend.name;
  }

  noGrad<T>(fn: () => T): T {
    const previous = this.gradEnabled;
    this.gradEnabled = false;
    try {
      return fn();
    } finally {
      this.gradEnabled = previous;
    }
  }

  async noGradAsync<T>(fn: () => Promise<T>): Promise<T> {
    const previous = this.gradEnabled;
    this.gradEnabled = false;
    try {
      return await fn();
    } finally {
      this.gradEnabled = previous;
    }
  }

  manualSeed(seed: number): void {
    if (!Number.isInteger(seed)) {
      throw new Error("manualSeed expects an integer seed");
    }
    this.seed = seed >>> 0;
    this.seedState = this.seed;
  }

  random(): number {
    if (this.seedState === undefined) {
      return Math.random();
    }
    this.seedState = (1664525 * this.seedState + 1013904223) >>> 0;
    return this.seedState / 0x100000000;
  }

  enableProfiler(): void {
    this.profilerEnabled = true;
  }

  disableProfiler(): void {
    this.profilerEnabled = false;
  }
}

const runtime = new Runtime();

export function getRuntime(): Runtime {
  return runtime;
}

export const Gradlith = {
  setBackend: (backend: BackendName | Backend) => runtime.setBackend(backend),
  getBackend: () => runtime.getBackend(),
  noGrad: <T>(fn: () => T) => runtime.noGrad(fn),
  noGradAsync: <T>(fn: () => Promise<T>) => runtime.noGradAsync(fn),
  manualSeed: (seed: number) => runtime.manualSeed(seed),
  enableProfiler: () => runtime.enableProfiler(),
  disableProfiler: () => runtime.disableProfiler()
};

export const setBackend = Gradlith.setBackend;
export const getBackend = Gradlith.getBackend;
export const noGradAsync = Gradlith.noGradAsync;
export const manualSeed = Gradlith.manualSeed;
