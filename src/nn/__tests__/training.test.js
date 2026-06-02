import { describe, it, expect } from 'vitest';
import { initNetwork, forwardPass, backprop } from '../network.js';
import { XOR_DATA } from '../datasets.js';
import {
  trainOneEpoch,
  evaluateXOR,
  checkConvergence,
  runGradientCheck,
  CONVERGENCE_LOSS_THRESHOLD,
} from '../training.js';

// Smooth activations only for the fuzz test. ReLU is intentionally excluded:
// its f′(0)=0 kink makes the centered finite-difference disagree with backprop
// whenever a pre-activation lands within ±ε of zero, which would be a flaky
// failure of a *correct* implementation rather than a real bug. ReLU is covered
// separately below with explicit kink filtering.
const SMOOTH_ACTS = ['tanh', 'sigmoid'];

// Deterministic PRNG (mulberry32) so the fuzz loop is fully reproducible — a
// given seed always produces the same architectures, weights and checked
// indices, so a pass/fail is stable across runs and CI.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Build a network whose weights come from `rng` (not Math.random) so the whole
// test is deterministic. Weights are O(1) so smooth units stay in their active,
// non-saturated range where the gradient check is most informative.
function seededNet(sizes, rng) {
  const weights = [];
  const biases = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const W = Array.from({ length: sizes[l + 1] }, () =>
      Array.from({ length: sizes[l] }, () => (rng() * 2 - 1) * 0.9)
    );
    weights.push(W);
    biases.push(Array.from({ length: sizes[l + 1] }, () => (rng() * 2 - 1) * 0.3));
  }
  return { weights, biases };
}

// True if perturbing W[l][j][k] by ±ε would push any ReLU pre-activation across
// its kink for some sample — i.e. a hidden-layer z that sits within `margin` of
// zero. Such cases make the finite-difference estimate straddle the kink.
function nearReluKink(weights, biases, acts, dataset, margin) {
  for (const { input } of dataset) {
    const { preActivations } = forwardPass(input, weights, biases, acts);
    // preActivations[l+1] is the pre-activation of hidden layer l (acts[l]).
    for (let l = 0; l < acts.length; l++) {
      if (acts[l] !== 'relu') continue;
      for (const z of preActivations[l + 1]) {
        if (Math.abs(z) < margin) return true;
      }
    }
  }
  return false;
}

describe('runGradientCheck — backprop matches finite differences', () => {
  it('agrees to <1e-4 relative error for every weight in a small smooth net', () => {
    const rng = mulberry32(12345);
    const { weights, biases } = seededNet([2, 3, 1], rng);
    const acts = ['tanh'];
    weights.forEach((W, l) => {
      W.forEach((row, j) => {
        row.forEach((_, k) => {
          const { relError } = runGradientCheck(weights, biases, acts, l, j, k);
          expect(relError).toBeLessThan(1e-4);
        });
      });
    });
  });

  // The strongest credibility test: fuzz across many random architectures and
  // (smooth) activations, checking analytical backprop against numerical
  // gradients. Fully seeded, so it is reproducible and never flaky.
  it('agrees across many random architectures and smooth activations', () => {
    const rng = mulberry32(2024);
    for (let trial = 0; trial < 40; trial++) {
      const hidden = 1 + Math.floor(rng() * 3);                 // 1–3 hidden layers
      const sizes = [2];
      const acts = [];
      for (let i = 0; i < hidden; i++) {
        sizes.push(2 + Math.floor(rng() * 5));                  // 2–6 neurons
        acts.push(SMOOTH_ACTS[Math.floor(rng() * SMOOTH_ACTS.length)]);
      }
      sizes.push(1);

      const { weights: w, biases } = seededNet(sizes, rng);
      for (let s = 0; s < 5; s++) {
        const l = Math.floor(rng() * w.length);
        const j = Math.floor(rng() * w[l].length);
        const k = Math.floor(rng() * w[l][j].length);
        const { backpropGrad, fdGrad, relError } = runGradientCheck(w, biases, acts, l, j, k);
        // Either the gradients agree relatively, or both are essentially zero.
        const bothTiny = Math.abs(backpropGrad) < 1e-6 && Math.abs(fdGrad) < 1e-6;
        expect(bothTiny || relError < 1e-3).toBe(true);
      }
    }
  });

  // ReLU is checked explicitly, skipping only the kink cases (z within ε of 0)
  // where the centered finite-difference is known not to model f′(0)=0.
  it('agrees for ReLU networks away from the f′(0)=0 kink', () => {
    const rng = mulberry32(909);
    const margin = 1e-3; // comfortably larger than the 1e-4 FD epsilon
    let checked = 0;
    for (let trial = 0; trial < 30; trial++) {
      const sizes = [2, 2 + Math.floor(rng() * 4), 1];
      const acts = ['relu'];
      const { weights, biases } = seededNet(sizes, rng);
      weights.forEach((W, l) => W.forEach((row, j) => row.forEach((_, k) => {
        if (nearReluKink(weights, biases, acts, XOR_DATA, margin)) return; // skip kink cases
        const { backpropGrad, fdGrad, relError } = runGradientCheck(weights, biases, acts, l, j, k);
        const bothTiny = Math.abs(backpropGrad) < 1e-6 && Math.abs(fdGrad) < 1e-6;
        expect(bothTiny || relError < 1e-3).toBe(true);
        checked++;
      })));
    }
    expect(checked).toBeGreaterThan(0); // the filter must not skip everything
  });

  it('reports the analytical gradient consistent with a direct backprop average', () => {
    const { weights, biases } = initNetwork([2, 4, 1]);
    const acts = ['tanh'];
    const [l, j, k] = [0, 1, 0];
    let total = 0;
    for (const { input, label } of XOR_DATA) {
      const { activations, preActivations } = forwardPass(input, weights, biases, acts);
      const { dWeights } = backprop([label], activations, preActivations, weights, acts);
      total += dWeights[l][j][k];
    }
    const direct = total / XOR_DATA.length;
    const { backpropGrad } = runGradientCheck(weights, biases, acts, l, j, k);
    expect(backpropGrad).toBeCloseTo(direct, 12);
  });
});

describe('trainOneEpoch', () => {
  it('returns updated parameters, a finite loss, and per-sample forward data', () => {
    const { weights, biases } = initNetwork([2, 4, 1]);
    const result = trainOneEpoch(weights, biases, ['relu'], 0.5);
    expect(Number.isFinite(result.loss)).toBe(true);
    expect(result.weights).toHaveLength(weights.length);
    expect(result.allForwardData).toHaveLength(XOR_DATA.length);
  });

  it('does not mutate the input weights', () => {
    const { weights, biases } = initNetwork([2, 4, 1]);
    const before = JSON.stringify(weights);
    trainOneEpoch(weights, biases, ['relu'], 0.5);
    expect(JSON.stringify(weights)).toBe(before);
  });

  it('drives the loss down over many epochs (learns XOR)', () => {
    // Tanh is a reliable XOR learner; run enough epochs that the trend is clear.
    let { weights, biases } = initNetwork([2, 6, 1]);
    const acts = ['tanh'];
    const firstLoss = trainOneEpoch(weights, biases, acts, 0.5).loss;
    let loss = firstLoss;
    for (let e = 0; e < 4000; e++) {
      const r = trainOneEpoch(weights, biases, acts, 0.5);
      weights = r.weights;
      biases = r.biases;
      loss = r.loss;
    }
    expect(loss).toBeLessThan(firstLoss);
    expect(loss).toBeLessThan(0.05);
  });
});

describe('evaluateXOR', () => {
  it('returns one result per XOR point with class/confidence fields', () => {
    const { weights, biases } = initNetwork([2, 4, 1]);
    const results = evaluateXOR(weights, biases, ['relu']);
    expect(results).toHaveLength(4);
    for (const r of results) {
      expect([0, 1]).toContain(r.predictedClass);
      expect(r.confidence).toBeGreaterThanOrEqual(0);
      expect(r.confidence).toBeLessThanOrEqual(1);
      expect(typeof r.correct).toBe('boolean');
    }
  });

  it('classifies all four points correctly once trained', () => {
    let { weights, biases } = initNetwork([2, 6, 1]);
    const acts = ['tanh'];
    for (let e = 0; e < 4000; e++) {
      const r = trainOneEpoch(weights, biases, acts, 0.5);
      weights = r.weights;
      biases = r.biases;
    }
    const results = evaluateXOR(weights, biases, acts);
    expect(results.every(r => r.correct)).toBe(true);
  });
});

describe('checkConvergence', () => {
  const highConf = [0, 1, 1, 0].map(label => ({ correct: true, confidence: 0.99, label }));

  it('converges when loss drops below the threshold', () => {
    const out = checkConvergence(CONVERGENCE_LOSS_THRESHOLD / 2, highConf, 0);
    expect(out.converged).toBe(true);
    expect(out.reason).toMatch(/Loss dropped below/);
  });

  it('converges on sustained high-confidence correctness above the loss threshold', () => {
    const out = checkConvergence(0.02, highConf, 50);
    expect(out.converged).toBe(true);
    expect(out.reason).toMatch(/consecutive epochs/);
  });

  it('does not converge before the consecutive-epoch count is reached', () => {
    expect(checkConvergence(0.02, highConf, 49).converged).toBe(false);
  });

  it('does not converge when a point is misclassified', () => {
    const oneWrong = [...highConf];
    oneWrong[0] = { correct: false, confidence: 0.99 };
    expect(checkConvergence(0.02, oneWrong, 100).converged).toBe(false);
  });
});
