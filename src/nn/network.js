// =============================================================================
// NETWORK CORE — initialization, forward pass, loss, backprop, weight update
// =============================================================================
// All neural-network math is implemented from scratch in JavaScript.
//
// Math conventions used throughout:
//   L         = number of weight matrices (= number of layers minus 1)
//   a[l]      = activation vector at layer l  (post-activation)
//   z[l]      = pre-activation vector at layer l  (W[l-1]·a[l-1] + b[l-1])
//   W[l]      = weight matrix between layer l and l+1, shape [out × in]
//   b[l]      = bias vector for layer l+1, shape [out]
//   δ[l]      = error signal (gradient w.r.t. z[l]) at layer l
//   dW[l]     = gradient of loss w.r.t. W[l]
// =============================================================================
import { ACTIVATIONS } from './activations.js';

// -----------------------------------------------------------------------------
// NETWORK INITIALIZATION
// Xavier (Glorot): scale = sqrt(2 / (fan_in + fan_out))
// -----------------------------------------------------------------------------
export function initNetwork(layerSizes) {
  const weights = [];
  const biases = [];
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const fanIn  = layerSizes[l];
    const fanOut = layerSizes[l + 1];
    const scale  = Math.sqrt(2.0 / (fanIn + fanOut));
    const W = Array.from({ length: fanOut }, () =>
      Array.from({ length: fanIn }, () => (Math.random() * 2 - 1) * scale)
    );
    weights.push(W);
    biases.push(new Array(fanOut).fill(0));
  }
  return { weights, biases };
}

// -----------------------------------------------------------------------------
// FORWARD PASS
// Propagates one input vector through all layers, storing z and a at every step.
// Returns all intermediates so backprop can reuse them without re-computing.
// -----------------------------------------------------------------------------
export function forwardPass(input, weights, biases, hiddenActivationTypes) {
  const activations    = [input];
  const preActivations = [null];
  const L = weights.length;
  for (let l = 0; l < L; l++) {
    const W     = weights[l];
    const b     = biases[l];
    const prevA = activations[l];
    const z = W.map((row, j) => {
      let sum = b[j];
      for (let k = 0; k < prevA.length; k++) sum += row[k] * prevA[k];
      return sum;
    });
    const isOutput     = l === L - 1;
    const activType    = isOutput ? 'sigmoid' : (hiddenActivationTypes[l] || 'relu');
    const activationFn = ACTIVATIONS[activType].fn;
    preActivations.push(z);
    activations.push(z.map(activationFn));
  }
  return { activations, preActivations };
}

// -----------------------------------------------------------------------------
// LOSS FUNCTION
// Binary Cross-Entropy: L = −(1/N)·Σ[y·log(ŷ)+(1−y)·log(1−ŷ)]
// -----------------------------------------------------------------------------
export function computeLoss(predictions, targets) {
  const eps = 1e-12;
  let total = 0;
  for (let i = 0; i < predictions.length; i++) {
    const p = Math.max(eps, Math.min(1 - eps, predictions[i][0]));
    const y = targets[i];
    total -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
  }
  return total / predictions.length;
}

// -----------------------------------------------------------------------------
// BACKPROPAGATION
// δ[L] = a[L]−y  (BCE+sigmoid shortcut, cancels σ' in output layer)
// δ[l] = (W[l]ᵀ·δ[l+1]) ⊙ activation'(z[l])
// dW[l] = δ[l+1]·a[l]ᵀ
// -----------------------------------------------------------------------------
export function backprop(targets, activations, preActivations, weights, hiddenActivationTypes) {
  const L = weights.length;
  const dWeights = weights.map(W => W.map(row => row.map(() => 0)));
  const dBiases  = weights.map(W => new Array(W.length).fill(0));
  const deltas   = new Array(L + 1).fill(null);

  deltas[L] = activations[L].map((val, j) => val - targets[j]);

  for (let l = L - 1; l >= 1; l--) {
    const activType = hiddenActivationTypes[l - 1] || 'relu';
    const actDeriv  = ACTIVATIONS[activType].derivative;
    const z         = preActivations[l];
    const nextDelta = deltas[l + 1];
    const nextW     = weights[l];
    deltas[l] = z.map((zVal, j) => {
      let sum = 0;
      for (let k = 0; k < nextDelta.length; k++) sum += nextW[k][j] * nextDelta[k];
      return sum * actDeriv(zVal);
    });
  }

  for (let l = 0; l < L; l++) {
    const delta = deltas[l + 1];
    const prevA = activations[l];
    for (let j = 0; j < delta.length; j++) {
      for (let k = 0; k < prevA.length; k++) dWeights[l][j][k] += delta[j] * prevA[k];
      dBiases[l][j] += delta[j];
    }
  }

  return { dWeights, dBiases, deltas };
}

// -----------------------------------------------------------------------------
// GRADIENT DESCENT UPDATE
// W[l] ← W[l] − lr · ∂L/∂W[l]
// -----------------------------------------------------------------------------
export function updateWeights(weights, biases, dWeights, dBiases, lr) {
  return {
    weights: weights.map((W, l) =>
      W.map((row, j) => row.map((w, k) => w - lr * dWeights[l][j][k]))
    ),
    biases: biases.map((b, l) =>
      b.map((val, j) => val - lr * dBiases[l][j])
    ),
  };
}

// -----------------------------------------------------------------------------
// DECISION BOUNDARY COMPUTATION
// Samples a gridSize×gridSize grid over [0,1]² using real forward passes.
// y-axis flipped: row=0 → x₂=1 (canvas top), matching cy = (1−x₂)·H convention.
// -----------------------------------------------------------------------------
export function computeDecisionBoundary(weights, biases, hiddenActivationTypes, gridSize = 40) {
  const L    = weights.length;
  const grid = [];
  for (let row = 0; row < gridSize; row++) {
    const gridRow = [];
    for (let col = 0; col < gridSize; col++) {
      const x1 = col / (gridSize - 1);
      const x2 = 1 - row / (gridSize - 1); // flip so row=0 is x₂=1 (top)
      const { activations } = forwardPass([x1, x2], weights, biases, hiddenActivationTypes);
      gridRow.push(activations[L][0]);
    }
    grid.push(gridRow);
  }
  return grid;
}
