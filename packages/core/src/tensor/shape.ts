export type Shape = number[];

export function sizeOf(shape: Shape): number {
  return shape.reduce((product, dim) => product * dim, 1);
}

export function stridesOf(shape: Shape): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i -= 1) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

export function assertShape(shape: Shape): void {
  if (shape.length === 0) {
    return;
  }
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new Error(`Invalid tensor shape [${shape.join(", ")}]`);
    }
  }
}

export function inferShape(value: unknown): Shape {
  if (!Array.isArray(value)) {
    return [];
  }
  const length = value.length;
  if (length === 0) {
    throw new Error("Cannot infer shape from an empty array");
  }
  const childShape = inferShape(value[0]);
  for (const child of value) {
    const nextShape = inferShape(child);
    if (!sameShape(childShape, nextShape)) {
      throw new Error("Nested arrays must be rectangular");
    }
  }
  return [length, ...childShape];
}

export function flatten(value: unknown, output: number[] = []): number[] {
  if (Array.isArray(value)) {
    for (const child of value) {
      flatten(child, output);
    }
    return output;
  }
  if (typeof value !== "number") {
    throw new Error("Tensor values must be numbers");
  }
  output.push(value);
  return output;
}

export function sameShape(a: Shape, b: Shape): boolean {
  return a.length === b.length && a.every((dim, index) => dim === b[index]);
}

export function broadcastShapes(a: Shape, b: Shape): Shape {
  const length = Math.max(a.length, b.length);
  const out = new Array<number>(length);
  for (let i = 0; i < length; i += 1) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`Cannot broadcast [${a.join(", ")}] with [${b.join(", ")}]`);
    }
    out[length - 1 - i] = Math.max(aDim, bDim);
  }
  return out;
}

export function unravelIndex(flatIndex: number, shape: Shape): number[] {
  const strides = stridesOf(shape);
  return shape.map((_, index) => Math.floor(flatIndex / strides[index]) % shape[index]);
}

export function ravelIndex(indices: number[], strides: number[]): number {
  return indices.reduce((offset, index, axis) => offset + index * strides[axis], 0);
}

export function broadcastOffset(outIndices: number[], outShape: Shape, inShape: Shape, inStrides: number[]): number {
  if (inShape.length === 0) {
    return 0;
  }
  const offset = outShape.length - inShape.length;
  let flat = 0;
  for (let i = 0; i < inShape.length; i += 1) {
    const sourceIndex = inShape[i] === 1 ? 0 : outIndices[i + offset];
    flat += sourceIndex * inStrides[i];
  }
  return flat;
}

export function unbroadcast(data: Float32Array, outShape: Shape, targetShape: Shape): Float32Array {
  const target = new Float32Array(sizeOf(targetShape));
  const targetStrides = stridesOf(targetShape);
  for (let i = 0; i < data.length; i += 1) {
    const outIndices = unravelIndex(i, outShape);
    if (targetShape.length === 0) {
      target[0] += data[i];
      continue;
    }
    const offset = outShape.length - targetShape.length;
    let targetOffset = 0;
    for (let axis = 0; axis < targetShape.length; axis += 1) {
      const targetIndex = targetShape[axis] === 1 ? 0 : outIndices[axis + offset];
      targetOffset += targetIndex * targetStrides[axis];
    }
    target[targetOffset] += data[i];
  }
  return target;
}

