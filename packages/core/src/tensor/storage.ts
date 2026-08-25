export type TensorDevice = "cpu" | "webgpu";

export interface TensorStorage {
  readonly device: TensorDevice;
  readonly size: number;
}

export class CPUStorage implements TensorStorage {
  readonly device = "cpu" as const;

  constructor(readonly data: Float32Array) {}

  get size(): number {
    return this.data.length;
  }

  clone(): CPUStorage {
    return new CPUStorage(new Float32Array(this.data));
  }
}

export interface GPUStorage extends TensorStorage {
  readonly device: "webgpu";
  readonly buffer: unknown;
}

export function cpuStorageFrom(data: Float32Array | number[]): CPUStorage {
  return new CPUStorage(data instanceof Float32Array ? data : new Float32Array(data));
}

export function assertCPUStorage(storage: TensorStorage): CPUStorage {
  if (storage instanceof CPUStorage) {
    return storage;
  }
  throw new Error(`Tensor data is resident on ${storage.device}; move it to CPU before reading`);
}

export function isTensorStorage(value: unknown): value is TensorStorage {
  return typeof value === "object" && value !== null && "device" in value && "size" in value;
}
