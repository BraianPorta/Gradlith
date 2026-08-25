import { Tensor } from "../tensor/Tensor";
import { LossFunction } from "./losses";
import { Module } from "./Module";
import { layerFromJSON, SerializedLayer } from "./layers";
import { Optimizer } from "../optim/optimizers";

export interface FitOptions {
  epochs: number;
  optimizer: Optimizer;
  loss: LossFunction;
  batchSize?: number;
  onEpoch?: (metrics: FitMetrics) => void;
}

export interface FitMetrics {
  epoch: number;
  loss: number;
  accuracy?: number;
}

export class Sequential extends Module {
  constructor(readonly layers: Module[]) {
    super();
  }

  forward(input: Tensor): Tensor {
    return this.layers.reduce((activation, layer) => layer.forward(activation), input);
  }

  parameters(): Tensor[] {
    return this.layers.flatMap((layer) => layer.parameters());
  }

  async fit(x: Tensor, y: Tensor, options: FitOptions): Promise<FitMetrics[]> {
    const metrics: FitMetrics[] = [];
    const batchSize = options.batchSize ?? x.shape[0] ?? x.size;
    for (let epoch = 1; epoch <= options.epochs; epoch += 1) {
      let epochLoss = 0;
      let batches = 0;
      for (const [xb, yb] of batchesOf(x, y, batchSize)) {
        options.optimizer.zeroGrad(this.parameters());
        const prediction = this.forward(xb);
        const loss = options.loss(prediction, yb);
        loss.backward();
        options.optimizer.step(this.parameters());
        epochLoss += loss.item();
        batches += 1;
      }
      const prediction = this.forward(x);
      const metric = {
        epoch,
        loss: epochLoss / batches,
        accuracy: binaryAccuracy(prediction, y)
      };
      metrics.push(metric);
      options.onEpoch?.(metric);
      if (epoch % 25 === 0) {
        await Promise.resolve();
      }
    }
    return metrics;
  }

  save(): SerializedModel {
    return {
      format: "gradlith",
      version: 1,
      layers: this.layers.map((layer) => {
        const serializable = layer as Module & { toJSON?: () => SerializedLayer };
        if (!serializable.toJSON) {
          throw new Error("Layer is not serializable");
        }
        return serializable.toJSON();
      })
    };
  }

  static load(serialized: SerializedModel): Sequential {
    if (serialized.format !== "gradlith" || serialized.version !== 1) {
      throw new Error("Unsupported Gradlith model format");
    }
    return new Sequential(serialized.layers.map(layerFromJSON));
  }
}

export interface SerializedModel {
  format: "gradlith";
  version: 1;
  layers: SerializedLayer[];
}

function* batchesOf(x: Tensor, y: Tensor, batchSize: number): Generator<[Tensor, Tensor]> {
  if (x.shape.length !== 2 || y.shape.length !== 2 || x.shape[0] !== y.shape[0]) {
    throw new Error("fit() expects rank-2 feature and target tensors with the same row count");
  }
  const rows = x.shape[0];
  const xCols = x.shape[1];
  const yCols = y.shape[1];
  const indices = Array.from({ length: rows }, (_, index) => index);
  for (let i = rows - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  for (let start = 0; start < rows; start += batchSize) {
    const slice = indices.slice(start, start + batchSize);
    const xb = new Float32Array(slice.length * xCols);
    const yb = new Float32Array(slice.length * yCols);
    slice.forEach((row, outRow) => {
      xb.set(x.data.slice(row * xCols, row * xCols + xCols), outRow * xCols);
      yb.set(y.data.slice(row * yCols, row * yCols + yCols), outRow * yCols);
    });
    yield [new Tensor(xb, [slice.length, xCols]), new Tensor(yb, [slice.length, yCols])];
  }
}

function binaryAccuracy(prediction: Tensor, target: Tensor): number | undefined {
  if (prediction.shape.length !== 2 || target.shape.length !== 2 || prediction.shape[1] !== 1 || target.shape[1] !== 1) {
    return undefined;
  }
  let correct = 0;
  for (let i = 0; i < prediction.data.length; i += 1) {
    if ((prediction.data[i] >= 0.5 ? 1 : 0) === (target.data[i] >= 0.5 ? 1 : 0)) {
      correct += 1;
    }
  }
  return correct / prediction.data.length;
}

