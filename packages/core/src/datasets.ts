import { Tensor } from "./tensor/Tensor";

export interface Dataset {
  x: Tensor;
  y: Tensor;
  points: Array<{ x: number; y: number; label: number }>;
}

export function xorDataset(): Dataset {
  const points = [
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
    { x: 1, y: 1, label: 0 }
  ];
  return fromPoints(points);
}

export function circlesDataset(samples = 160, noise = 0.06): Dataset {
  const points = Array.from({ length: samples }, (_, index) => {
    const label = index % 2;
    const radius = label === 0 ? 0.45 : 0.9;
    const angle = Math.random() * Math.PI * 2;
    return {
      x: Math.cos(angle) * radius + gaussian() * noise,
      y: Math.sin(angle) * radius + gaussian() * noise,
      label
    };
  });
  return fromPoints(points);
}

export function moonsDataset(samples = 160, noise = 0.08): Dataset {
  const points = Array.from({ length: samples }, (_, index) => {
    const label = index % 2;
    const angle = Math.random() * Math.PI;
    const x = label === 0 ? Math.cos(angle) : 1 - Math.cos(angle);
    const y = label === 0 ? Math.sin(angle) : 0.45 - Math.sin(angle);
    return {
      x: x - 0.5 + gaussian() * noise,
      y: y - 0.25 + gaussian() * noise,
      label
    };
  });
  return fromPoints(points);
}

export function spiralDataset(samples = 200, noise = 0.08): Dataset {
  const points = Array.from({ length: samples }, (_, index) => {
    const label = index % 2;
    const t = (index / samples) * 4 * Math.PI + label * Math.PI;
    const radius = index / samples;
    return {
      x: Math.cos(t) * radius + gaussian() * noise,
      y: Math.sin(t) * radius + gaussian() * noise,
      label
    };
  });
  return fromPoints(points);
}

function fromPoints(points: Array<{ x: number; y: number; label: number }>): Dataset {
  return {
    x: Tensor.from(points.map((point) => [point.x, point.y])),
    y: Tensor.from(points.map((point) => [point.label])),
    points
  };
}

function gaussian(): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

