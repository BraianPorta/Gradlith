import { useEffect, useMemo, useRef, useState } from "react";
import type { SerializedModel } from "@gradlith/core";
import { WebGPUBackend } from "@gradlith/core";
import { runBenchmarks, type BenchmarkResult } from "./benchmarks";
import { deleteExperiment, listExperiments, saveExperiment, type StoredExperiment } from "./experiments";

type DatasetName = "spiral" | "moons" | "circles" | "xor";
type OptimizerName = "adam" | "sgd" | "momentum" | "rmsprop";
type View = "playground" | "optimizers" | "graph" | "benchmarks" | "experiments" | "builder" | "docs";

interface Point {
  x: number;
  y: number;
  label: number;
}

interface BoundaryCell {
  x: number;
  y: number;
  p: number;
}

interface RaceRow {
  optimizer: OptimizerName;
  loss: number;
  accuracy: number;
  history: Array<{ epoch: number; loss: number; accuracy: number }>;
}

interface EpochMessage {
  type: "epoch";
  epoch: number;
  loss: number;
  accuracy: number;
  points: Point[];
  boundary: BoundaryCell[];
  weights: Array<{ layer: string; values: number[]; gradientNorm: number }>;
  graph: Array<{ id: number; label: string; operation: string; parents: number[]; shape: number[] }>;
  race: RaceRow[];
  model: SerializedModel;
}

interface DoneMessage {
  type: "done";
  experiment: Omit<StoredExperiment, "id" | "createdAt">;
}

const datasets: DatasetName[] = ["spiral", "moons", "circles", "xor"];
const optimizers: OptimizerName[] = ["adam", "sgd", "momentum", "rmsprop"];
const views: View[] = ["playground", "optimizers", "graph", "benchmarks", "experiments", "builder", "docs"];

export function App() {
  const [dataset, setDataset] = useState<DatasetName>("spiral");
  const [optimizer, setOptimizer] = useState<OptimizerName>("adam");
  const [hidden, setHidden] = useState(24);
  const [learningRate, setLearningRate] = useState(0.03);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<EpochMessage | undefined>();
  const [history, setHistory] = useState<Array<{ epoch: number; loss: number; accuracy: number }>>([]);
  const [view, setView] = useState<View>(() => routeFromHash());
  const [experiments, setExperiments] = useState<StoredExperiment[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([]);
  const [benchmarking, setBenchmarking] = useState(false);
  const [builderLayers, setBuilderLayers] = useState(["Dense(2, 24)", "Tanh", "Dense(24, 24)", "Tanh", "Dense(24, 1)", "Sigmoid"]);
  const [importedModel, setImportedModel] = useState<SerializedModel | undefined>();
  const workerRef = useRef<Worker | undefined>(undefined);

  const backend = useMemo(() => (WebGPUBackend.isSupported() ? "WebGPU ready" : "CPU backend"), []);

  useEffect(() => {
    refreshExperiments();
    const onHash = () => setView(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => {
      window.removeEventListener("hashchange", onHash);
      workerRef.current?.terminate();
    };
  }, []);

  function start() {
    workerRef.current?.terminate();
    const worker = new Worker(new URL("./training.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    setRunning(true);
    setHistory([]);
    worker.onmessage = async (event: MessageEvent<EpochMessage | DoneMessage>) => {
      const data = event.data;
      if (data.type === "done") {
        setRunning(false);
        await saveExperiment(data.experiment);
        await refreshExperiments();
        return;
      }
      setMessage(data);
      setHistory((current) => [...current.slice(-110), { epoch: data.epoch, loss: data.loss, accuracy: data.accuracy }]);
    };
    worker.postMessage({ type: "start", dataset, optimizer, epochs: 900, hidden, learningRate });
  }

  function stop() {
    workerRef.current?.postMessage({ type: "stop" });
    setRunning(false);
  }

  async function refreshExperiments() {
    if ("indexedDB" in window) {
      setExperiments(await listExperiments());
    }
  }

  async function measure() {
    setBenchmarking(true);
    try {
      setBenchmarks(await runBenchmarks());
    } finally {
      setBenchmarking(false);
    }
  }

  function exportModel() {
    if (!message?.model) {
      return;
    }
    const blob = new Blob([JSON.stringify(message.model, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "gradlith-model.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="mark">∇</span>
          <div>
            <h1>GRADLITH</h1>
            <p>Browser-native deep learning from first principles.</p>
          </div>
        </div>
        <nav className="tabs" aria-label="Gradlith views">
          {views.map((item) => (
            <button className={view === item ? "selected" : ""} key={item} onClick={() => navigate(item)}>
              {item}
            </button>
          ))}
        </nav>
        <div className="status">
          <span>{backend}</span>
          <strong>{message ? `${Math.round(message.accuracy * 100)}%` : "idle"}</strong>
        </div>
      </header>

      {view === "playground" && (
        <>
          <section className="workspace">
            <RunPanel dataset={dataset} optimizer={optimizer} hidden={hidden} learningRate={learningRate} running={running} onDataset={setDataset} onOptimizer={setOptimizer} onHidden={setHidden} onLearningRate={setLearningRate} onRun={running ? stop : start} onExport={exportModel} canExport={Boolean(message?.model)} />
            <section className="stage">
              <DecisionCanvas points={message?.points ?? []} boundary={message?.boundary ?? []} />
              <div className="metrics">
                <Metric label="epoch" value={message?.epoch ?? 0} />
                <Metric label="loss" value={message ? message.loss.toFixed(4) : "0.0000"} />
                <Metric label="accuracy" value={message ? `${Math.round(message.accuracy * 100)}%` : "0%"} />
              </div>
            </section>
            <GradientInspector weights={message?.weights ?? []} />
          </section>
          <section className="bottom-grid">
            <LossChart history={history} />
            <GraphView graph={message?.graph ?? []} />
            <OptimizerRace selected={optimizer} race={message?.race ?? []} />
          </section>
        </>
      )}

      {view === "optimizers" && <OptimizerRace selected={optimizer} race={message?.race ?? []} expanded />}
      {view === "graph" && <GraphExplorer graph={message?.graph ?? []} />}
      {view === "benchmarks" && <BenchmarkPanel results={benchmarks} running={benchmarking} onRun={measure} />}
      {view === "experiments" && <ExperimentsPanel experiments={experiments} onDelete={async (id) => { await deleteExperiment(id); await refreshExperiments(); }} />}
      {view === "builder" && <BuilderPanel layers={builderLayers} importedModel={importedModel} onImport={setImportedModel} onLayers={setBuilderLayers} />}
      {view === "docs" && <DocsPanel />}
    </main>
  );
}

function navigate(view: View) {
  window.location.hash = view;
}

function routeFromHash(): View {
  const value = window.location.hash.replace("#", "");
  if (views.includes(value as View)) {
    return value as View;
  }
  const path = window.location.pathname.split("/").filter(Boolean).at(-1) ?? "";
  return views.includes(path as View) ? (path as View) : "playground";
}

function RunPanel(props: {
  dataset: DatasetName;
  optimizer: OptimizerName;
  hidden: number;
  learningRate: number;
  running: boolean;
  canExport: boolean;
  onDataset: (value: DatasetName) => void;
  onOptimizer: (value: OptimizerName) => void;
  onHidden: (value: number) => void;
  onLearningRate: (value: number) => void;
  onRun: () => void;
  onExport: () => void;
}) {
  return (
    <aside className="panel controls">
      <h2>Run</h2>
      <label>
        Dataset
        <select value={props.dataset} onChange={(event) => props.onDataset(event.target.value as DatasetName)} disabled={props.running}>
          {datasets.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </label>
      <label>
        Optimizer
        <select value={props.optimizer} onChange={(event) => props.onOptimizer(event.target.value as OptimizerName)} disabled={props.running}>
          {optimizers.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </label>
      <label>
        Hidden units
        <input type="range" min="8" max="48" value={props.hidden} disabled={props.running} onChange={(event) => props.onHidden(Number(event.target.value))} />
        <b>{props.hidden}</b>
      </label>
      <label>
        Learning rate
        <input type="range" min="0.005" max="0.08" step="0.005" value={props.learningRate} disabled={props.running} onChange={(event) => props.onLearningRate(Number(event.target.value))} />
        <b>{props.learningRate.toFixed(3)}</b>
      </label>
      <button onClick={props.onRun}>{props.running ? "Stop" : "Train"}</button>
      <button className="secondary" onClick={props.onExport} disabled={!props.canExport}>
        Export model
      </button>
    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DecisionCanvas({ points, boundary }: { points: Point[]; boundary: BoundaryCell[] }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0a0f18";
    ctx.fillRect(0, 0, width, height);
    const scaleX = (x: number) => ((x + 1.3) / 2.6) * width;
    const scaleY = (y: number) => height - ((y + 1.3) / 2.6) * height;
    const cell = width / 44;
    for (const item of boundary) {
      const hot = Math.floor(item.p * 180);
      const cold = Math.floor((1 - item.p) * 160);
      ctx.fillStyle = `rgba(${hot + 35}, ${cold + 45}, 150, 0.54)`;
      ctx.fillRect(scaleX(item.x), scaleY(item.y), cell + 1, cell + 1);
    }
    ctx.strokeStyle = "rgba(226,232,240,.18)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 7; i += 1) {
      const x = (i / 6) * width;
      const y = (i / 6) * height;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    for (const point of points) {
      ctx.beginPath();
      ctx.fillStyle = point.label ? "#f97316" : "#2dd4bf";
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 1.4;
      ctx.arc(scaleX(point.x), scaleY(point.y), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [points, boundary]);

  return <canvas ref={ref} width={820} height={540} className="decision" aria-label="Decision boundary" />;
}

function GradientInspector({ weights }: { weights: EpochMessage["weights"] }) {
  return (
    <aside className="panel inspector">
      <h2>Gradient Inspector</h2>
      {weights.length === 0 && <p className="empty">Start training to inspect parameter norms.</p>}
      {weights.map((group) => (
        <div className="weight-row" key={group.layer}>
          <div>
            <strong>{group.layer}</strong>
            <span>grad norm {group.gradientNorm.toExponential(2)}</span>
          </div>
          <div className="bars">
            {group.values.map((value, index) => (
              <i key={index} style={{ height: `${Math.min(42, Math.abs(value) * 22 + 4)}px`, background: value >= 0 ? "#2dd4bf" : "#f97316" }} />
            ))}
          </div>
        </div>
      ))}
    </aside>
  );
}

function LossChart({ history }: { history: Array<{ epoch: number; loss: number; accuracy: number }> }) {
  const points = history.map((item, index) => {
    const x = 20 + (index / Math.max(1, history.length - 1)) * 300;
    const maxLoss = Math.max(0.01, ...history.map((row) => row.loss));
    const y = 150 - (item.loss / maxLoss) * 120;
    return `${x},${y}`;
  });
  return (
    <section className="panel">
      <h2>Loss</h2>
      <svg viewBox="0 0 340 170" role="img" aria-label="Training loss curve">
        <path d="M20 12 V150 H324" className="axis" />
        <polyline points={points.join(" ")} className="loss-line" />
      </svg>
    </section>
  );
}

function GraphView({ graph }: { graph: EpochMessage["graph"] }) {
  return (
    <section className="panel graph">
      <h2>Computational Graph</h2>
      <div>
        {graph.slice(-11).map((node) => (
          <span key={node.id}>{node.operation}</span>
        ))}
      </div>
    </section>
  );
}

function GraphExplorer({ graph }: { graph: EpochMessage["graph"] }) {
  return (
    <section className="wide panel graph-explorer">
      <h2>Computational Graph</h2>
      <div className="graph-grid">
        {graph.map((node) => (
          <article key={node.id}>
            <strong>{node.operation}</strong>
            <span>{node.shape.length ? `[${node.shape.join(", ")}]` : "scalar"}</span>
            <small>{node.parents.length ? `parents ${node.parents.join(", ")}` : "leaf"}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

function OptimizerRace({ selected, race, expanded = false }: { selected: OptimizerName; race: RaceRow[]; expanded?: boolean }) {
  const rows = race.length ? race : optimizers.map((item) => ({ optimizer: item, loss: 0, accuracy: 0, history: [] }));
  return (
    <section className={expanded ? "wide panel race" : "panel race"}>
      <h2>Optimizer Race</h2>
      {rows.map((item) => (
        <div className={item.optimizer === selected ? "active" : ""} key={item.optimizer}>
          <span>{item.optimizer}</span>
          <meter min={0} max={1} value={item.accuracy} />
          {expanded && <strong>{`${Math.round(item.accuracy * 100)}% / ${item.loss.toFixed(4)}`}</strong>}
        </div>
      ))}
      {expanded && <RaceChart rows={rows} />}
    </section>
  );
}

function RaceChart({ rows }: { rows: RaceRow[] }) {
  const colors: Record<OptimizerName, string> = { adam: "#bef264", sgd: "#38bdf8", momentum: "#f97316", rmsprop: "#c084fc" };
  const maxLoss = Math.max(0.01, ...rows.flatMap((row) => row.history.map((item) => item.loss)));
  return (
    <svg viewBox="0 0 700 220" role="img" aria-label="Optimizer loss comparison">
      <path d="M30 12 V195 H680" className="axis" />
      {rows.map((row) => (
        <polyline
          key={row.optimizer}
          points={row.history
            .map((item, index) => {
              const x = 30 + (index / Math.max(1, row.history.length - 1)) * 640;
              const y = 195 - (item.loss / maxLoss) * 170;
              return `${x},${y}`;
            })
            .join(" ")}
          fill="none"
          stroke={colors[row.optimizer]}
          strokeWidth="3"
        />
      ))}
    </svg>
  );
}

function BenchmarkPanel({ results, running, onRun }: { results: BenchmarkResult[]; running: boolean; onRun: () => void }) {
  return (
    <section className="wide panel benchmarks">
      <div className="section-head">
        <h2>Benchmarks</h2>
        <button onClick={onRun} disabled={running}>{running ? "Measuring" : "Run"}</button>
      </div>
      <table>
        <thead>
          <tr>
            <th>Operation</th>
            <th>Size</th>
            <th>CPU ms</th>
            <th>GPU ms</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result) => (
            <tr key={`${result.operation}-${result.size}`}>
              <td>{result.operation}</td>
              <td>{result.size}</td>
              <td>{result.cpuMs.toFixed(3)}</td>
              <td>{result.gpuMs === undefined ? "not available" : result.gpuMs.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function ExperimentsPanel({ experiments, onDelete }: { experiments: StoredExperiment[]; onDelete: (id: string) => void }) {
  return (
    <section className="wide panel experiments">
      <h2>Experiments</h2>
      <div className="experiment-grid">
        {experiments.map((experiment) => (
          <article key={experiment.id}>
            <strong>{experiment.name}</strong>
            <span>{new Date(experiment.createdAt).toLocaleString()}</span>
            <b>{Math.round(experiment.metrics.accuracy * 100)}%</b>
            <small>{`loss ${experiment.metrics.loss.toFixed(4)} / epoch ${experiment.metrics.epoch}`}</small>
            <button className="secondary" onClick={() => onDelete(experiment.id)}>Delete</button>
          </article>
        ))}
      </div>
    </section>
  );
}

function BuilderPanel({ layers, importedModel, onImport, onLayers }: { layers: string[]; importedModel?: SerializedModel; onImport: (model: SerializedModel) => void; onLayers: (layers: string[]) => void }) {
  const code = `const model = new Sequential([\n${layers.map((layer) => `  new ${layer}${layer.includes("(") ? "" : "()"},`).join("\n")}\n]);`;
  async function importModel(file?: File) {
    if (!file) {
      return;
    }
    const parsed = JSON.parse(await file.text()) as SerializedModel;
    if (parsed.format === "gradlith" && parsed.version === 1) {
      onImport(parsed);
    }
  }
  return (
    <section className="wide builder-layout">
      <aside className="panel palette">
        <h2>Model Builder</h2>
        {["Dense(2, 24)", "Dense(24, 24)", "Tanh", "ReLU", "Sigmoid", "Softmax"].map((layer) => (
          <button className="secondary" key={layer} onClick={() => onLayers([...layers, layer])}>{layer}</button>
        ))}
        <label>
          Import JSON
          <input type="file" accept="application/json" onChange={(event) => importModel(event.target.files?.[0])} />
        </label>
        {importedModel && <small>{`${importedModel.layers.length} imported layers`}</small>}
      </aside>
      <section className="panel stack">
        {layers.map((layer, index) => (
          <article key={`${layer}-${index}`}>
            <span>{index + 1}</span>
            <strong>{layer}</strong>
            <button className="secondary" onClick={() => onLayers(layers.filter((_, item) => item !== index))}>Remove</button>
          </article>
        ))}
      </section>
      <pre className="panel code">{code}</pre>
    </section>
  );
}

function DocsPanel() {
  return (
    <section className="wide docs-grid">
      {[
        ["Tensor Engine", "Float32Array storage, row-major strides, broadcasting and differentiable math operations."],
        ["Autograd", "Reverse-mode graph traversal with accumulated gradients across reused tensors."],
        ["Neural Networks", "Sequential modules, Dense layers, activations, losses and JSON serialization."],
        ["Optimizers", "SGD, Momentum, RMSProp and Adam update ordinary tensor parameters."],
        ["WebGPU", "Native async kernels for add, ReLU and matmul with CPU fallback when browser support is absent."],
        ["Playground", "Training runs in a Web Worker and streams metrics, boundaries, graphs and experiments."]
      ].map(([title, body]) => (
        <article className="panel" key={title}>
          <h2>{title}</h2>
          <p>{body}</p>
        </article>
      ))}
    </section>
  );
}
