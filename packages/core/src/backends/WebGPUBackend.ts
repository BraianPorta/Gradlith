import { Tensor } from "../tensor/Tensor";
import type { Shape } from "../tensor/shape";
import type { TensorStorage } from "../tensor/storage";
import { CPUBackend } from "./CPUBackend";
import type { Backend, BinaryOpSpec, BroadcastSpec, MatMulSpec, UnaryOpSpec } from "./Backend";

export class WebGPUBackend implements Backend {
  readonly name = "webgpu" as const;
  private readonly fallback = new CPUBackend();
  private devicePromise?: Promise<unknown>;

  static isSupported(): boolean {
    return typeof navigator !== "undefined" && "gpu" in navigator;
  }

  binary(left: TensorStorage, right: TensorStorage, spec: BinaryOpSpec): TensorStorage {
    return this.fallback.binary(left, right, spec);
  }

  unary(input: TensorStorage, shape: Shape, spec: UnaryOpSpec): TensorStorage {
    return this.fallback.unary(input, shape, spec);
  }

  pow(input: TensorStorage, shape: Shape, exponent: number): TensorStorage {
    return this.fallback.pow(input, shape, exponent);
  }

  sum(input: TensorStorage): TensorStorage {
    return this.fallback.sum(input);
  }

  max(input: TensorStorage): TensorStorage {
    return this.fallback.max(input);
  }

  min(input: TensorStorage): TensorStorage {
    return this.fallback.min(input);
  }

  broadcastTo(input: TensorStorage, spec: BroadcastSpec): TensorStorage {
    return this.fallback.broadcastTo(input, spec);
  }

  reshape(input: TensorStorage, shape: Shape): TensorStorage {
    return this.fallback.reshape(input, shape);
  }

  transpose(input: TensorStorage, rows: number, cols: number): TensorStorage {
    return this.fallback.transpose(input, rows, cols);
  }

  matmul(left: TensorStorage, right: TensorStorage, spec: MatMulSpec): TensorStorage {
    return this.fallback.matmul(left, right, spec);
  }

  softmax(input: TensorStorage, rows: number, cols: number): TensorStorage {
    return this.fallback.softmax(input, rows, cols);
  }

  dispose(storage: TensorStorage): void {
    this.fallback.dispose(storage);
  }

  async addAsync(a: Tensor, b: Tensor): Promise<Tensor> {
    if (!sameShape(a.shape, b.shape)) {
      return a.add(b);
    }
    const data = await this.runElementwise(addShaderWGSL, [a.data, b.data], a.size);
    return new Tensor(data, a.shape);
  }

  async reluAsync(x: Tensor): Promise<Tensor> {
    const data = await this.runElementwise(reluShaderWGSL, [x.data], x.size);
    return new Tensor(data, x.shape);
  }

  async matmulAsync(a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      return a.matmul(b);
    }
    const [m, k] = a.shape;
    const [k2, n] = b.shape;
    if (k !== k2) {
      return a.matmul(b);
    }
    const device = await this.device();
    const outputSize = m * n;
    const buffers = [
      storageBuffer(device, a.data),
      storageBuffer(device, b.data),
      emptyStorageBuffer(device, outputSize),
      uniformBuffer(device, new Uint32Array([m, k, n, 0]))
    ];
    await runCompute(device, matmulShaderWGSL, buffers, Math.ceil(outputSize / 64));
    return new Tensor(await readBuffer(device, buffers[2], outputSize), [m, n]);
  }

  private async runElementwise(shader: string, inputs: Float32Array[], size: number): Promise<Float32Array> {
    const device = await this.device();
    const buffers = [
      ...inputs.map((input) => storageBuffer(device, input)),
      emptyStorageBuffer(device, size),
      uniformBuffer(device, new Uint32Array([size, 0, 0, 0]))
    ];
    await runCompute(device, shader, buffers, Math.ceil(size / 64));
    return readBuffer(device, buffers[inputs.length], size);
  }

  private async device(): Promise<any> {
    if (!WebGPUBackend.isSupported()) {
      throw new Error("WebGPU is not available in this browser");
    }
    this.devicePromise ??= (async () => {
      const gpu = (navigator as unknown as { gpu: { requestAdapter: () => Promise<any> } }).gpu;
      const adapter = await gpu.requestAdapter();
      if (!adapter) {
        throw new Error("No WebGPU adapter found");
      }
      return adapter.requestDevice();
    })();
    return this.devicePromise;
  }
}

export const addShaderWGSL = `
struct Meta {
  size: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> meta: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= meta.size) {
    return;
  }
  out[i] = a[i] + b[i];
}
`;

export const reluShaderWGSL = `
struct Meta {
  size: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> meta: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= meta.size) {
    return;
  }
  out[i] = max(x[i], 0.0);
}
`;

export const matmulShaderWGSL = `
struct Meta {
  m: u32,
  k: u32,
  n: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> meta: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let total = meta.m * meta.n;
  if (index >= total) {
    return;
  }
  let row = index / meta.n;
  let col = index % meta.n;
  var sum = 0.0;
  for (var inner = 0u; inner < meta.k; inner = inner + 1u) {
    sum = sum + a[row * meta.k + inner] * b[inner * meta.n + col];
  }
  out[index] = sum;
}
`;

function sameShape(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((dim, index) => dim === b[index]);
}

function storageBuffer(device: any, data: Float32Array): any {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: gpuBufferUsage().STORAGE | gpuBufferUsage().COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function emptyStorageBuffer(device: any, size: number): any {
  return device.createBuffer({
    size: size * Float32Array.BYTES_PER_ELEMENT,
    usage: gpuBufferUsage().STORAGE | gpuBufferUsage().COPY_SRC
  });
}

function uniformBuffer(device: any, data: Uint32Array): any {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: gpuBufferUsage().UNIFORM | gpuBufferUsage().COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

async function runCompute(device: any, shader: string, buffers: any[], workgroups: number): Promise<void> {
  const module = device.createShaderModule({ code: shader });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" }
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function readBuffer(device: any, source: any, size: number): Promise<Float32Array> {
  const bytes = size * Float32Array.BYTES_PER_ELEMENT;
  const readback = device.createBuffer({
    size: bytes,
    usage: gpuBufferUsage().COPY_DST | gpuBufferUsage().MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(gpuMapMode().READ);
  const copy = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return copy;
}

function gpuBufferUsage(): any {
  return (globalThis as unknown as { GPUBufferUsage: unknown }).GPUBufferUsage;
}

function gpuMapMode(): any {
  return (globalThis as unknown as { GPUMapMode: unknown }).GPUMapMode;
}
