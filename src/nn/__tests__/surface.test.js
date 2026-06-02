import { describe, it, expect } from 'vitest';
import { computeLossSurface, computeDescentPath } from '../surface.js';
import { initNetwork, forwardPass, computeLoss } from '../network.js';
import { makeDataset } from '../datasets.js';

const COORD_A = [0, 0, 0];
const COORD_B = [0, 1, 0];

describe('computeLossSurface', () => {
  it('returns a gridSize×gridSize grid centered on the current weights', () => {
    const net = initNetwork([2, 3, 1]);
    const data = makeDataset('xor');
    const s = computeLossSurface(net, ['relu'], data, COORD_A, COORD_B, { span: 2, gridSize: 11 });
    expect(s.grid).toHaveLength(11);
    expect(s.grid[0]).toHaveLength(11);
    expect(s.center.a).toBeCloseTo(net.weights[0][0][0], 12);
    expect(s.center.b).toBeCloseTo(net.weights[0][1][0], 12);
    // axes span ±span around the center
    expect(s.aVals[0]).toBeCloseTo(s.center.a - 2, 12);
    expect(s.aVals[10]).toBeCloseTo(s.center.a + 2, 12);
  });

  it('does not mutate the source network', () => {
    const net = initNetwork([2, 3, 1]);
    const before = JSON.stringify(net.weights);
    computeLossSurface(net, ['relu'], makeDataset('xor'), COORD_A, COORD_B, { gridSize: 9 });
    expect(JSON.stringify(net.weights)).toBe(before);
  });

  it('reports min/max consistent with the grid contents', () => {
    const net = initNetwork([2, 4, 1]);
    const s = computeLossSurface(net, ['tanh'], makeDataset('xor'), COORD_A, COORD_B, { gridSize: 13 });
    const flat = s.grid.flat();
    expect(s.min).toBeCloseTo(Math.min(...flat), 12);
    expect(s.max).toBeCloseTo(Math.max(...flat), 12);
    expect(s.min).toBeGreaterThanOrEqual(0); // BCE loss is non-negative
  });

  it('the grid cell at the center matches a direct loss evaluation', () => {
    const net = initNetwork([2, 3, 1]);
    const data = makeDataset('xor');
    const acts = ['relu'];
    const gridSize = 11; // odd → exact center cell
    const s = computeLossSurface(net, acts, data, COORD_A, COORD_B, { span: 2, gridSize });
    const mid = (gridSize - 1) / 2;
    // direct: evaluate loss at the untouched network
    const preds = data.map(d => forwardPass(d.input, net.weights, net.biases, acts).activations.at(-1));
    const direct = computeLoss(preds, data.map(d => d.label));
    expect(s.grid[mid][mid]).toBeCloseTo(direct, 10);
  });
});

describe('computeDescentPath', () => {
  it('records a position per epoch plus the starting point', () => {
    const net = initNetwork([2, 4, 1]);
    const path = computeDescentPath(net, ['tanh'], makeDataset('xor'), COORD_A, COORD_B, 0.1, 20, 'sgd');
    expect(path).toHaveLength(21);            // start + 20 steps
    expect(path[0].loss).toBeNull();          // no loss recorded before training
    expect(path[0].a).toBeCloseTo(net.weights[0][0][0], 12);
    for (let i = 1; i < path.length; i++) expect(Number.isFinite(path[i].loss)).toBe(true);
  });

  it('does not mutate the source network', () => {
    const net = initNetwork([2, 4, 1]);
    const before = JSON.stringify(net.weights);
    computeDescentPath(net, ['tanh'], makeDataset('xor'), COORD_A, COORD_B, 0.1, 30, 'adam');
    expect(JSON.stringify(net.weights)).toBe(before);
  });

  it('drives the recorded loss downward overall', () => {
    const net = initNetwork([2, 6, 1]);
    const path = computeDescentPath(net, ['tanh'], makeDataset('xor'), COORD_A, COORD_B, 0.1, 800, 'adam');
    expect(path.at(-1).loss).toBeLessThan(path[1].loss);
  });
});
