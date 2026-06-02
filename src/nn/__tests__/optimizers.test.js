import { describe, it, expect } from 'vitest';
import { OPTIMIZERS, createOptimizer, cloneOptimizer, optimizerStep } from '../optimizers.js';
import { initNetwork } from '../network.js';
import { makeDataset } from '../datasets.js';
import { trainOneEpoch, runOptimizerComparison } from '../training.js';

const net = () => ({ weights: [[[0.5, -0.5]]], biases: [[0.1]] });

describe('OPTIMIZERS registry', () => {
  it('defines sgd, momentum, rmsprop, adam with labels and descriptions', () => {
    for (const id of ['sgd', 'momentum', 'rmsprop', 'adam']) {
      expect(OPTIMIZERS[id].id).toBe(id);
      expect(OPTIMIZERS[id].label.length).toBeGreaterThan(0);
      expect(OPTIMIZERS[id].description.length).toBeGreaterThan(0);
    }
  });
});

describe('createOptimizer', () => {
  it('allocates moment buffers only when the rule needs them', () => {
    const n = net();
    const sgd = createOptimizer('sgd', 0.1, n);
    expect(sgd.mW).toBeNull();
    expect(sgd.vW).toBeNull();

    const mom = createOptimizer('momentum', 0.1, n);
    expect(mom.mW).not.toBeNull();
    expect(mom.vW).toBeNull();

    const rms = createOptimizer('rmsprop', 0.1, n);
    expect(rms.mW).toBeNull();
    expect(rms.vW).not.toBeNull();

    const adam = createOptimizer('adam', 0.1, n);
    expect(adam.mW).not.toBeNull();
    expect(adam.vW).not.toBeNull();
    // buffers mirror the weight/bias shape and start at zero
    expect(adam.mW).toEqual([[[0, 0]]]);
    expect(adam.mB).toEqual([[0]]);
  });
});

describe('optimizerStep', () => {
  it('SGD reproduces W ← W − lr·g exactly', () => {
    const { weights, biases } = net();
    const opt = createOptimizer('sgd', 0.1, { weights, biases });
    const dW = [[[1, 2]]];
    const dB = [[3]];
    const out = optimizerStep(opt, weights, biases, dW, dB);
    expect(out.weights[0][0][0]).toBeCloseTo(0.5 - 0.1 * 1, 12);
    expect(out.weights[0][0][1]).toBeCloseTo(-0.5 - 0.1 * 2, 12);
    expect(out.biases[0][0]).toBeCloseTo(0.1 - 0.1 * 3, 12);
  });

  it('does not mutate the input weights', () => {
    const { weights, biases } = net();
    const opt = createOptimizer('adam', 0.1, { weights, biases });
    optimizerStep(opt, weights, biases, [[[1, 1]]], [[1]]);
    expect(weights[0][0][0]).toBe(0.5);
    expect(biases[0][0]).toBe(0.1);
  });

  it('momentum accumulates velocity across steps', () => {
    let { weights, biases } = net();
    const opt = createOptimizer('momentum', 0.1, { weights, biases }, { momentum: 0.9 });
    const g = [[[1, 0]]]; const gb = [[0]];
    // step 1: v = 1            → ΔW = lr·1
    let out = optimizerStep(opt, weights, biases, g, gb);
    const d1 = weights[0][0][0] - out.weights[0][0][0];
    weights = out.weights; biases = out.biases;
    // step 2: v = 0.9·1 + 1 = 1.9 → ΔW = lr·1.9  (larger than step 1)
    out = optimizerStep(opt, weights, biases, g, gb);
    const d2 = weights[0][0][0] - out.weights[0][0][0];
    expect(d1).toBeCloseTo(0.1, 12);
    expect(d2).toBeCloseTo(0.19, 12);
    expect(d2).toBeGreaterThan(d1);
  });

  it("adam's first step is ≈ lr·sign(g) regardless of gradient magnitude", () => {
    // With bias correction, step 1 ≈ lr * g/|g| = lr for positive g of any scale.
    const { weights, biases } = net();
    const optA = createOptimizer('adam', 0.05, { weights, biases });
    const optB = createOptimizer('adam', 0.05, { weights, biases });
    const small = optimizerStep(optA, weights, biases, [[[0.01, 0]]], [[0]]);
    const large = optimizerStep(optB, weights, biases, [[[100, 0]]], [[0]]);
    const dSmall = weights[0][0][0] - small.weights[0][0][0];
    const dLarge = weights[0][0][0] - large.weights[0][0][0];
    expect(dSmall).toBeCloseTo(0.05, 4);
    expect(dLarge).toBeCloseTo(0.05, 4);
  });

  it('advances the timestep each call', () => {
    const { weights, biases } = net();
    const opt = createOptimizer('adam', 0.1, { weights, biases });
    expect(opt.t).toBe(0);
    optimizerStep(opt, weights, biases, [[[1, 1]]], [[1]]);
    optimizerStep(opt, weights, biases, [[[1, 1]]], [[1]]);
    expect(opt.t).toBe(2);
  });
});

describe('cloneOptimizer', () => {
  it('produces an independent copy whose buffers do not alias the original', () => {
    const { weights, biases } = net();
    const opt = createOptimizer('adam', 0.1, { weights, biases });
    optimizerStep(opt, weights, biases, [[[1, 1]]], [[1]]);
    const clone = cloneOptimizer(opt);
    expect(clone.t).toBe(opt.t);
    expect(clone.mW).toEqual(opt.mW);
    // advancing the clone must not touch the original
    optimizerStep(clone, weights, biases, [[[1, 1]]], [[1]]);
    expect(clone.t).toBe(opt.t + 1);
    expect(clone.mW).not.toEqual(opt.mW);
  });
});

describe('trainOneEpoch with an optimizer', () => {
  it('matches the SGD code path when given an sgd optimizer', () => {
    const data = makeDataset('xor');
    const n1 = initNetwork([2, 4, 1]);
    // identical starting point for both runs
    const n2 = { weights: structuredClone(n1.weights), biases: structuredClone(n1.biases) };
    const acts = ['tanh'];
    const plain = trainOneEpoch(n1.weights, n1.biases, acts, 0.1, data);
    const opt = createOptimizer('sgd', 0.1, n2);
    const viaOpt = trainOneEpoch(n2.weights, n2.biases, acts, 0.1, data, opt);
    expect(viaOpt.weights).toEqual(plain.weights);
    expect(viaOpt.loss).toBeCloseTo(plain.loss, 12);
  });

  it('Adam reaches low loss on XOR in far fewer epochs than SGD', () => {
    const data = makeDataset('xor');
    const start = initNetwork([2, 6, 1]);
    const acts = ['tanh'];

    const run = (optType, lr, epochs) => {
      let w = structuredClone(start.weights);
      let b = structuredClone(start.biases);
      const opt = createOptimizer(optType, lr, { weights: w, biases: b });
      let loss = Infinity;
      for (let e = 0; e < epochs; e++) {
        const r = trainOneEpoch(w, b, acts, lr, data, opt);
        w = r.weights; b = r.biases; loss = r.loss;
      }
      return loss;
    };

    const adamLoss = run('adam', 0.05, 400);
    const sgdLoss  = run('sgd',  0.05, 400);
    expect(adamLoss).toBeLessThan(0.05);     // Adam converges quickly
    expect(adamLoss).toBeLessThan(sgdLoss);  // and beats plain SGD at equal lr/epochs
  });
});

describe('runOptimizerComparison', () => {
  it('returns one fixed-length loss curve per optimizer from a shared start', () => {
    const net = initNetwork([2, 5, 1]);
    const data = makeDataset('xor');
    const types = ['sgd', 'momentum', 'rmsprop', 'adam'];
    const results = runOptimizerComparison(net, ['tanh'], data, types, 0.05, 120);
    expect(results.map(r => r.type)).toEqual(types);
    for (const r of results) {
      expect(r.losses).toHaveLength(120);
      expect(r.finalLoss).toBe(r.losses[119]);
      expect(Number.isFinite(r.finalLoss)).toBe(true);
    }
  });

  it('does not mutate the starting network (each run is independent)', () => {
    const net = initNetwork([2, 4, 1]);
    const before = JSON.stringify(net.weights);
    runOptimizerComparison(net, ['relu'], makeDataset('xor'), ['sgd', 'adam'], 0.1, 50);
    expect(JSON.stringify(net.weights)).toBe(before);
  });
});
