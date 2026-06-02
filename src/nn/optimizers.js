// =============================================================================
// OPTIMIZERS
// =============================================================================
// Gradient-descent update rules. The simulator computes ∂L/∂W via backprop
// (see network.js); the optimizer decides how to turn those gradients into a
// weight step. All four match their PyTorch counterparts.
//
// An optimizer is a small stateful object: it owns the running buffers
// (velocity / moment estimates) that some rules accumulate across epochs, plus
// a timestep `t`. `optimizerStep` mutates those buffers in place and returns a
// fresh { weights, biases } pair (weights are never mutated, so React state
// updates stay clean).
// =============================================================================

export const OPTIMIZERS = {
  sgd: {
    id: 'sgd', label: 'SGD',
    description: 'Vanilla gradient descent: W ← W − lr·g. The update the rest of the simulator narrates.',
  },
  momentum: {
    id: 'momentum', label: 'Momentum',
    description: 'SGD with velocity: v ← μ·v + g, W ← W − lr·v. Builds speed along consistent directions and rolls through small bumps.',
  },
  rmsprop: {
    id: 'rmsprop', label: 'RMSProp',
    description: 'Per-weight adaptive step: divides by a running root-mean-square of recent gradients, so steep and flat directions move at similar rates.',
  },
  adam: {
    id: 'adam', label: 'Adam',
    description: 'Momentum + RMSProp with bias correction. The robust default for most problems; usually escapes plateaus SGD gets stuck on.',
  },
};

const DEFAULTS = { momentum: 0.9, alpha: 0.99, beta1: 0.9, beta2: 0.999, eps: 1e-8 };

// Build a zero buffer with the same nested shape as a weights/biases tree.
function zerosLike(x) {
  return Array.isArray(x) ? x.map(zerosLike) : 0;
}

const needsFirstMoment  = type => type === 'momentum' || type === 'adam';
const needsSecondMoment = type => type === 'rmsprop'  || type === 'adam';

// Create an optimizer for a given network. `overrides` can tune the momentum/
// beta/eps hyper-parameters; everything falls back to the PyTorch defaults.
export function createOptimizer(type, lr, net, overrides = {}) {
  const cfg = { lr, ...DEFAULTS, ...overrides };
  return {
    type,
    cfg,
    t: 0,
    mW: needsFirstMoment(type)  ? zerosLike(net.weights) : null,
    mB: needsFirstMoment(type)  ? zerosLike(net.biases)  : null,
    vW: needsSecondMoment(type) ? zerosLike(net.weights) : null,
    vB: needsSecondMoment(type) ? zerosLike(net.biases)  : null,
  };
}

// Deep copy so an explained-step preview can advance a throwaway optimizer
// without committing buffer changes until the user actually applies the epoch.
export function cloneOptimizer(opt) {
  return {
    type: opt.type,
    cfg: { ...opt.cfg },
    t: opt.t,
    mW: opt.mW && structuredClone(opt.mW),
    mB: opt.mB && structuredClone(opt.mB),
    vW: opt.vW && structuredClone(opt.vW),
    vB: opt.vB && structuredClone(opt.vB),
  };
}

// Recursively walk a parameter tree alongside its gradient and (optional)
// moment buffers, returning a new parameter tree. Buffers are mutated in place.
function updateTree(param, grad, m, v, t, type, cfg) {
  return param.map((p, i) => {
    if (Array.isArray(p)) {
      return updateTree(p, grad[i], m && m[i], v && v[i], t, type, cfg);
    }
    const g = grad[i];
    let step;
    if (type === 'sgd') {
      step = cfg.lr * g;
    } else if (type === 'momentum') {
      m[i] = cfg.momentum * m[i] + g;
      step = cfg.lr * m[i];
    } else if (type === 'rmsprop') {
      v[i] = cfg.alpha * v[i] + (1 - cfg.alpha) * g * g;
      step = (cfg.lr * g) / (Math.sqrt(v[i]) + cfg.eps);
    } else if (type === 'adam') {
      m[i] = cfg.beta1 * m[i] + (1 - cfg.beta1) * g;
      v[i] = cfg.beta2 * v[i] + (1 - cfg.beta2) * g * g;
      const mHat = m[i] / (1 - Math.pow(cfg.beta1, t));
      const vHat = v[i] / (1 - Math.pow(cfg.beta2, t));
      step = (cfg.lr * mHat) / (Math.sqrt(vHat) + cfg.eps);
    } else {
      step = cfg.lr * g; // unknown type → fall back to SGD
    }
    return p - step;
  });
}

// Apply one optimizer step. Advances the timestep, mutates the moment buffers,
// and returns freshly-allocated { weights, biases }.
export function optimizerStep(opt, weights, biases, dWeights, dBiases) {
  opt.t += 1;
  return {
    weights: updateTree(weights, dWeights, opt.mW, opt.vW, opt.t, opt.type, opt.cfg),
    biases:  updateTree(biases,  dBiases,  opt.mB, opt.vB, opt.t, opt.type, opt.cfg),
  };
}
