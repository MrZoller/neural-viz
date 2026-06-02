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

const ACT_TYPES = ['relu', 'tanh', 'sigmoid'];

// Build a randomized but reproducible-enough architecture for fuzzing.
function randomArch() {
  const hidden = 1 + Math.floor(Math.random() * 3);            // 1–3 hidden layers
  const sizes = [2];
  const acts = [];
  for (let i = 0; i < hidden; i++) {
    sizes.push(2 + Math.floor(Math.random() * 5));             // 2–6 neurons
    acts.push(ACT_TYPES[Math.floor(Math.random() * ACT_TYPES.length)]);
  }
  sizes.push(1);
  return { sizes, acts };
}

describe('runGradientCheck — backprop matches finite differences', () => {
  it('agrees to <1e-4 relative error for every weight in a small net', () => {
    const { weights, biases } = initNetwork([2, 3, 1]);
    const acts = ['relu'];
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
  // activations, checking analytical backprop against numerical gradients.
  it('agrees across many random architectures and activations', () => {
    for (let trial = 0; trial < 40; trial++) {
      const { sizes, acts } = randomArch();
      const { weights, biases } = initNetwork(sizes);
      // Nudge weights away from zero so ReLU units are actually active and the
      // numerical/analytical comparison is meaningful.
      const w = weights.map(W => W.map(row => row.map(v => v + (Math.random() * 2 - 1) * 0.5)));
      // Check a handful of randomly chosen weights per architecture.
      for (let s = 0; s < 5; s++) {
        const l = Math.floor(Math.random() * w.length);
        const j = Math.floor(Math.random() * w[l].length);
        const k = Math.floor(Math.random() * w[l][j].length);
        const { backpropGrad, fdGrad, relError } = runGradientCheck(w, biases, acts, l, j, k);
        // Either the gradients agree relatively, or both are essentially zero.
        const bothTiny = Math.abs(backpropGrad) < 1e-6 && Math.abs(fdGrad) < 1e-6;
        expect(bothTiny || relError < 1e-3).toBe(true);
      }
    }
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
