import { describe, it, expect } from 'vitest';
import { DATASETS, makeDataset, XOR_DATA } from '../datasets.js';
import { initNetwork, forwardPass, backprop } from '../network.js';
import { trainOneEpoch, evaluateDataset, runGradientCheck } from '../training.js';

const GEOMETRIC = ['circles', 'moons', 'spiral', 'linear', 'blobs'];
const ALL_IDS = Object.keys(DATASETS);

describe('dataset registry', () => {
  it('every dataset has an id, label, kind and description', () => {
    for (const id of ALL_IDS) {
      const d = DATASETS[id];
      expect(d.id).toBe(id);
      expect(typeof d.label).toBe('string');
      expect(['logical', 'geometric']).toContain(d.kind);
      expect(d.description.length).toBeGreaterThan(0);
    }
  });

  it('XOR_DATA is still the canonical 4-point truth table', () => {
    expect(makeDataset('xor')).toEqual(XOR_DATA);
  });

  it('logical gates produce exactly four points with binary labels', () => {
    for (const id of ['xor', 'and', 'or']) {
      const pts = makeDataset(id);
      expect(pts).toHaveLength(4);
      for (const p of pts) expect([0, 1]).toContain(p.label);
    }
  });
});

describe('geometric generators', () => {
  it('keep every point inside the unit square [0,1]²', () => {
    for (const id of GEOMETRIC) {
      const pts = makeDataset(id, { points: 120, noise: 0.05 });
      expect(pts.length).toBeGreaterThan(0);
      for (const { input } of pts) {
        expect(input[0]).toBeGreaterThanOrEqual(0);
        expect(input[0]).toBeLessThanOrEqual(1);
        expect(input[1]).toBeGreaterThanOrEqual(0);
        expect(input[1]).toBeLessThanOrEqual(1);
      }
    }
  });

  it('produce both classes', () => {
    for (const id of GEOMETRIC) {
      const pts = makeDataset(id, { points: 120 });
      const labels = new Set(pts.map(p => p.label));
      expect(labels.has(0)).toBe(true);
      expect(labels.has(1)).toBe(true);
    }
  });

  it('are deterministic for a fixed seed and vary with the seed', () => {
    const a = makeDataset('moons', { seed: 7 });
    const b = makeDataset('moons', { seed: 7 });
    const c = makeDataset('moons', { seed: 8 });
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });
});

describe('math core is dataset-parameterized', () => {
  it('trainOneEpoch reduces loss on a generated dataset', () => {
    const data = makeDataset('blobs', { points: 80, noise: 0.02, seed: 1 });
    let { weights, biases } = initNetwork([2, 6, 1]);
    const acts = ['tanh'];
    const first = trainOneEpoch(weights, biases, acts, 0.5, data).loss;
    let loss = first;
    for (let e = 0; e < 1500; e++) {
      const r = trainOneEpoch(weights, biases, acts, 0.5, data);
      weights = r.weights; biases = r.biases; loss = r.loss;
    }
    expect(loss).toBeLessThan(first);
  });

  it('evaluateDataset returns one result per point of the given dataset', () => {
    const data = makeDataset('circles', { points: 60 });
    const { weights, biases } = initNetwork([2, 4, 1]);
    const results = evaluateDataset(weights, biases, ['relu'], data);
    expect(results).toHaveLength(data.length);
  });

  it('gradient check still matches finite differences on a generated dataset', () => {
    const data = makeDataset('moons', { points: 40, seed: 3 });
    const { weights, biases } = initNetwork([2, 5, 1]);
    const w = weights.map(W => W.map(row => row.map(v => v + 0.3)));
    const acts = ['tanh'];
    for (let l = 0; l < w.length; l++) {
      const { backpropGrad, fdGrad, relError } =
        runGradientCheck(w, biases, acts, l, 0, 0, 1e-4, data);
      const bothTiny = Math.abs(backpropGrad) < 1e-6 && Math.abs(fdGrad) < 1e-6;
      expect(bothTiny || relError < 1e-3).toBe(true);
    }
  });
});
