import {
  Adam,
  binaryCrossEntropy,
  circlesDataset,
  Dense,
  Momentum,
  moonsDataset,
  noGrad,
  RMSProp,
  Sequential,
  SGD,
  Sigmoid,
  Tanh,
  spiralDataset,
  Tensor,
  xorDataset
} from "@gradlith/core";
import type { Dataset, Optimizer, SerializedModel } from "@gradlith/core";

type DatasetName = "spiral" | "moons" | "circles" | "xor";
type OptimizerName = "adam" | "sgd" | "momentum" | "rmsprop";

interface StartMessage {
  type: "start";
  dataset: DatasetName;
  optimizer: OptimizerName;
  epochs: number;
  hidden: number;
  learningRate: number;
}

let cancelled = false;

self.onmessage = async (event: MessageEvent<StartMessage | { type: "stop" }>) => {
  if (event.data.type === "stop") {
    cancelled = true;
    return;
  }
  cancelled = false;
  const { dataset, optimizer, epochs, hidden, learningRate } = event.data;
  const data = createDataset(dataset);
  const base = createModel(hidden);
  const runners = optimizerNames.map((name) => {
    const model = Sequential.load(base.save());
    return {
      name,
      model,
      optimizer: createOptimizer(name, tunedLearningRate(name, learningRate)),
      history: [] as Array<{ epoch: number; loss: number; accuracy: number }>
    };
  });

  for (let epoch = 1; epoch <= epochs && !cancelled; epoch += 1) {
    for (const runner of runners) {
      const parameters = runner.model.parameters();
      runner.optimizer.zeroGrad(parameters);
      const prediction = runner.model.forward(data.x);
      const loss = binaryCrossEntropy(prediction, data.y);
      loss.backward();
      runner.optimizer.step(parameters);
      const acc = noGrad(() => accuracy(runner.model.forward(data.x), data.y));
      runner.history.push({ epoch, loss: loss.item(), accuracy: acc });
    }

    if (epoch === 1 || epoch % 5 === 0 || epoch === epochs) {
      const selected = runners.find((runner) => runner.name === optimizer) ?? runners[0];
      const selectedLoss = selected.history[selected.history.length - 1];
      postMessage({
        type: "epoch",
        epoch,
        loss: selectedLoss.loss,
        accuracy: noGrad(() => accuracy(selected.model.forward(data.x), data.y)),
        boundary: boundary(selected.model),
        points: data.points,
        weights: weights(selected.model),
        graph: binaryCrossEntropy(selected.model.forward(data.x), data.y).graph(),
        race: runners.map((runner) => ({
          optimizer: runner.name,
          loss: runner.history[runner.history.length - 1].loss,
          accuracy: runner.history[runner.history.length - 1].accuracy,
          history: runner.history.filter((_, index) => index % 8 === 0).slice(-80)
        })),
        model: selected.model.save()
      });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  const selected = runners.find((runner) => runner.name === optimizer) ?? runners[0];
  postMessage({
    type: "done",
    experiment: {
      name: `${dataset}/${optimizer}`,
      dataset,
      optimizer,
      hidden,
      learningRate,
      model: selected.model.save(),
      metrics: selected.history[selected.history.length - 1],
      history: selected.history
    }
  });
};

const optimizerNames: OptimizerName[] = ["adam", "sgd", "momentum", "rmsprop"];

function createModel(hidden: number): Sequential {
  return new Sequential([
    new Dense(2, hidden, "xavier"),
    new Tanh(),
    new Dense(hidden, hidden, "xavier"),
    new Tanh(),
    new Dense(hidden, 1, "xavier"),
    new Sigmoid()
  ]);
}

function createDataset(name: DatasetName): Dataset {
  if (name === "xor") {
    return xorDataset();
  }
  if (name === "moons") {
    return moonsDataset(180);
  }
  if (name === "circles") {
    return circlesDataset(180);
  }
  return spiralDataset(220);
}

function createOptimizer(name: OptimizerName, learningRate: number): Optimizer {
  if (name === "sgd") {
    return new SGD({ learningRate });
  }
  if (name === "momentum") {
    return new Momentum({ learningRate, momentum: 0.88 });
  }
  if (name === "rmsprop") {
    return new RMSProp({ learningRate });
  }
  return new Adam({ learningRate });
}

function tunedLearningRate(name: OptimizerName, learningRate: number): number {
  if (name === "sgd") {
    return Math.min(0.2, learningRate * 1.7);
  }
  if (name === "rmsprop") {
    return Math.min(0.04, learningRate);
  }
  return learningRate;
}

function accuracy(prediction: Tensor, target: Tensor): number {
  let correct = 0;
  for (let i = 0; i < prediction.data.length; i += 1) {
    if ((prediction.data[i] >= 0.5 ? 1 : 0) === target.data[i]) {
      correct += 1;
    }
  }
  return correct / prediction.data.length;
}

function boundary(model: Sequential): Array<{ x: number; y: number; p: number }> {
  const grid: number[][] = [];
  const values: Array<{ x: number; y: number; p: number }> = [];
  for (let row = 0; row < 44; row += 1) {
    for (let col = 0; col < 44; col += 1) {
      const x = -1.25 + (col / 43) * 2.5;
      const y = -1.25 + (row / 43) * 2.5;
      grid.push([x, y]);
      values.push({ x, y, p: 0 });
    }
  }
  const prediction = noGrad(() => model.forward(Tensor.from(grid)));
  for (let i = 0; i < values.length; i += 1) {
    values[i].p = prediction.data[i];
  }
  return values;
}

function weights(model: Sequential): Array<{ layer: string; values: number[]; gradientNorm: number }> {
  return model.parameters().map((parameter, index) => ({
    layer: index % 2 === 0 ? `Dense ${Math.floor(index / 2) + 1} weights` : `Dense ${Math.floor(index / 2) + 1} bias`,
    values: Array.from(parameter.data.slice(0, 28)),
    gradientNorm: parameter.grad ? Math.sqrt(parameter.grad.data.reduce((total, value) => total + value * value, 0)) : 0
  }));
}

export interface WorkerExperiment {
  name: string;
  dataset: DatasetName;
  optimizer: OptimizerName;
  hidden: number;
  learningRate: number;
  model: SerializedModel;
  metrics: { epoch: number; loss: number; accuracy: number };
  history: Array<{ epoch: number; loss: number; accuracy: number }>;
}
