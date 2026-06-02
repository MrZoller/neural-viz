import { describe, it, expect } from 'vitest';
import {
  initNetwork,
  forwardPass,
  computeLoss,
  backprop,
  updateWeights,
  computeDecisionBoundary,
} from '../network.js';

describe('initNetwork', () => {
  it('produces one weight matrix and bias vector per layer gap', () => {
    const { weights, biases } = initNetwork([2, 3, 1]);
    expect(weights).toHaveLength(2);
    expect(biases).toHaveLength(2);
  });

  it('shapes each weight matrix as [out × in] (PyTorch nn.Linear orientation)', () => {
    const { weights, biases } = initNetwork([2, 4, 1]);
    expect(weights[0]).toHaveLength(4);      // out = 4
    expect(weights[0][0]).toHaveLength(2);   // in = 2
    expect(weights[1]).toHaveLength(1);      // out = 1
    expect(weights[1][0]).toHaveLength(4);   // in = 4
    expect(biases[0]).toHaveLength(4);
    expect(biases[1]).toHaveLength(1);
  });

  it('initializes biases to zero', () => {
    const { biases } = initNetwork([2, 5, 3, 1]);
    for (const b of biases) for (const v of b) expect(v).toBe(0);
  });

  it('keeps Xavier-scaled weights within the expected bound', () => {
    const { weights } = initNetwork([2, 3, 1]);
    // scale = sqrt(2/(fan_in+fan_out)); weights are uniform in (-scale, scale).
    const scale = Math.sqrt(2 / (2 + 3));
    for (const row of weights[0]) for (const w of row) expect(Math.abs(w)).toBeLessThanOrEqual(scale);
  });
});

describe('forwardPass', () => {
  it('records activations and pre-activations for every layer', () => {
    // Identity-ish hand-built net: 2 -> 1, weights [[1, 1]], bias [0].
    const weights = [[[1, 1]]];
    const biases = [[0]];
    const { activations, preActivations } = forwardPass([0.3, 0.7], weights, biases, []);
    expect(activations).toHaveLength(2);          // input + output
    expect(preActivations).toHaveLength(2);
    expect(preActivations[0]).toBeNull();         // no pre-activation for input
    expect(preActivations[1][0]).toBeCloseTo(1.0, 12);
    // output layer is always sigmoid: σ(1) ≈ 0.7310585786
    expect(activations[1][0]).toBeCloseTo(1 / (1 + Math.exp(-1)), 12);
  });

  it('applies the chosen hidden activation, output always sigmoid', () => {
    // 1 hidden neuron with ReLU, fed a negative pre-activation -> 0 after ReLU.
    const weights = [[[-1]], [[1]]];
    const biases = [[0], [0]];
    const { activations } = forwardPass([1], weights, biases, ['relu']);
    expect(activations[1][0]).toBe(0);            // ReLU(-1) = 0
    expect(activations[2][0]).toBeCloseTo(0.5, 12); // σ(0) = 0.5
  });
});

describe('computeLoss', () => {
  it('computes binary cross-entropy averaged over samples', () => {
    // Perfect predictions -> near-zero loss.
    const loss = computeLoss([[0.999999], [0.000001]], [1, 0]);
    expect(loss).toBeGreaterThan(0);
    expect(loss).toBeLessThan(1e-4);
  });

  it('penalizes confident wrong predictions heavily', () => {
    const good = computeLoss([[0.9]], [1]);
    const bad = computeLoss([[0.1]], [1]);
    expect(bad).toBeGreaterThan(good);
  });

  it('clamps to avoid log(0) producing Infinity', () => {
    expect(Number.isFinite(computeLoss([[0]], [1]))).toBe(true);
    expect(Number.isFinite(computeLoss([[1]], [0]))).toBe(true);
  });
});

describe('updateWeights', () => {
  it('steps weights and biases down the gradient by the learning rate', () => {
    const weights = [[[1, 2]]];
    const biases = [[0.5]];
    const dW = [[[0.1, 0.2]]];
    const dB = [[1]];
    const out = updateWeights(weights, biases, dW, dB, 0.5);
    expect(out.weights[0][0][0]).toBeCloseTo(1 - 0.5 * 0.1, 12);
    expect(out.weights[0][0][1]).toBeCloseTo(2 - 0.5 * 0.2, 12);
    expect(out.biases[0][0]).toBeCloseTo(0.5 - 0.5 * 1, 12);
  });

  it('does not mutate the inputs (returns fresh arrays)', () => {
    const weights = [[[1]]];
    const biases = [[0]];
    updateWeights(weights, biases, [[[1]]], [[1]], 0.1);
    expect(weights[0][0][0]).toBe(1);
    expect(biases[0][0]).toBe(0);
  });
});

describe('backprop', () => {
  it('uses the BCE+sigmoid shortcut δ[L] = ŷ − y at the output', () => {
    const weights = [[[0.5, 0.5]]];
    const biases = [[0]];
    const { activations, preActivations } = forwardPass([1, 1], weights, biases, []);
    const yhat = activations[1][0];
    const { deltas } = backprop([1], activations, preActivations, weights, []);
    expect(deltas[1][0]).toBeCloseTo(yhat - 1, 12);
  });

  it('produces gradients with the same shape as the weights', () => {
    const { weights, biases } = initNetwork([2, 3, 1]);
    const { activations, preActivations } = forwardPass([1, 0], weights, biases, ['relu']);
    const { dWeights, dBiases } = backprop([1], activations, preActivations, weights, ['relu']);
    expect(dWeights).toHaveLength(weights.length);
    dWeights.forEach((W, l) => {
      expect(W).toHaveLength(weights[l].length);
      W.forEach((row, j) => expect(row).toHaveLength(weights[l][j].length));
    });
    expect(dBiases[0]).toHaveLength(biases[0].length);
  });
});

describe('computeDecisionBoundary', () => {
  it('returns a gridSize×gridSize grid of probabilities in (0,1)', () => {
    const { weights, biases } = initNetwork([2, 3, 1]);
    const grid = computeDecisionBoundary(weights, biases, ['relu'], 8);
    expect(grid).toHaveLength(8);
    for (const row of grid) {
      expect(row).toHaveLength(8);
      for (const p of row) {
        expect(p).toBeGreaterThan(0);
        expect(p).toBeLessThan(1);
      }
    }
  });

  it('flips the y-axis so row 0 corresponds to x₂ = 1', () => {
    // Net keyed only on x₂: σ(10·x₂ − 5). Top row (x₂=1) should be high prob.
    const weights = [[[0, 10]]];
    const biases = [[-5]];
    const grid = computeDecisionBoundary(weights, biases, [], 5);
    expect(grid[0][0]).toBeGreaterThan(0.9);   // row 0 -> x₂ = 1
    expect(grid[4][0]).toBeLessThan(0.1);      // row 4 -> x₂ = 0
  });
});
