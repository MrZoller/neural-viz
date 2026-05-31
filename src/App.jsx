// =============================================================================
// Neural Network Learning Tool — Phase 1 MVP
// =============================================================================
// All neural-network math is implemented from scratch in JavaScript.
// No ML libraries. PyTorch snippets are explanatory text only.
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

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

// =============================================================================
// SECTION 1 — ACTIVATION FUNCTIONS
// Each entry has: fn (the activation), derivative (its derivative w.r.t. input)
// These are the exact functions PyTorch uses under the hood.
// =============================================================================
const ACTIVATIONS = {
  relu: {
    label: 'ReLU',
    // ReLU: max(0, x). Kills negative values, keeps positives unchanged.
    fn: x => Math.max(0, x),
    // Derivative is 1 when x > 0, 0 otherwise (technically undefined at 0, we use 0).
    derivative: x => (x > 0 ? 1 : 0),
    color: '#60a5fa',
  },
  tanh: {
    label: 'Tanh',
    // tanh maps any real to (-1, 1). Zero-centered, so gradients are better behaved than sigmoid.
    fn: x => Math.tanh(x),
    // Derivative: 1 - tanh²(x). Peaks at 1 when x=0, approaches 0 for large |x|.
    derivative: x => 1 - Math.tanh(x) ** 2,
    color: '#a78bfa',
  },
  sigmoid: {
    label: 'Sigmoid',
    // σ(x) = 1 / (1 + e^−x). Maps any real to (0, 1).
    // Used for binary classification output to get a probability.
    fn: x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
    // Derivative: σ(x) · (1 − σ(x)). Max value is 0.25 — this causes vanishing gradients
    // in deep networks because each layer multiplies by at most 0.25.
    derivative: x => {
      const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
      return s * (1 - s);
    },
    color: '#f472b6',
  },
};

// =============================================================================
// SECTION 2 — XOR DATASET
// XOR is the classic test for neural networks because it's not linearly separable.
// A single-layer perceptron cannot solve XOR; you need at least one hidden layer.
// =============================================================================
const XOR_DATA = [
  { input: [0, 0], label: 0 },
  { input: [0, 1], label: 1 },
  { input: [1, 0], label: 1 },
  { input: [1, 1], label: 0 },
];

// =============================================================================
// SECTION 3 — NETWORK INITIALIZATION
// Xavier (Glorot) initialization scales weights by sqrt(2 / (fan_in + fan_out)).
// This keeps variance stable across layers at the start of training.
// In PyTorch: torch.nn.init.xavier_uniform_(layer.weight)
// =============================================================================
function initNetwork(layerSizes) {
  const weights = [];
  const biases = [];

  for (let l = 0; l < layerSizes.length - 1; l++) {
    const fanIn = layerSizes[l];
    const fanOut = layerSizes[l + 1];
    // Xavier scale: maintains signal variance through forward pass
    const scale = Math.sqrt(2.0 / (fanIn + fanOut));

    // W[l] is a [fanOut × fanIn] matrix, stored as an array of rows.
    // W[l][j][k] is the weight from neuron k in layer l to neuron j in layer l+1.
    const W = Array.from({ length: fanOut }, () =>
      Array.from({ length: fanIn }, () => (Math.random() * 2 - 1) * scale)
    );
    weights.push(W);

    // Biases initialized to zero. PyTorch default for Linear layers.
    biases.push(new Array(fanOut).fill(0));
  }

  return { weights, biases };
}

// =============================================================================
// SECTION 4 — FORWARD PASS
// Propagates an input vector through all layers, computing z and a at each step.
// Returns all intermediate values so backprop can reuse them.
//
// PyTorch equivalent:
//   output = model(input_tensor)
//   # Internally: for each Linear + activation layer in model.forward()
// =============================================================================
function forwardPass(input, weights, biases, hiddenActivationTypes) {
  // activations[0] = the raw input vector
  const activations = [input];
  // preActivations[l] = z[l] = W[l-1]·a[l-1] + b[l-1]  (null for input layer)
  const preActivations = [null];

  const L = weights.length; // number of weight matrices

  for (let l = 0; l < L; l++) {
    const W = weights[l];
    const b = biases[l];
    const prevA = activations[l];

    // Compute z = W · prevA + b  (matrix-vector product)
    const z = W.map((row, j) => {
      let sum = b[j];
      for (let k = 0; k < prevA.length; k++) {
        sum += row[k] * prevA[k];
      }
      return sum;
    });

    // Apply activation:
    //   - Last layer (output): always sigmoid to get a probability in (0,1)
    //   - Hidden layers: use the configured activation for that layer
    const isOutputLayer = l === L - 1;
    const activationType = isOutputLayer ? 'sigmoid' : (hiddenActivationTypes[l] || 'relu');
    const activationFn = ACTIVATIONS[activationType].fn;
    const a = z.map(activationFn);

    preActivations.push(z);
    activations.push(a);
  }

  return { activations, preActivations };
}

// =============================================================================
// SECTION 5 — LOSS FUNCTION
// Binary Cross-Entropy (BCE): the standard loss for binary classification.
//   L = -1/N · Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
// where y is the true label and ŷ is the predicted probability.
// Intuitively: punishes confident wrong predictions very harshly (log of small number).
//
// PyTorch: torch.nn.BCELoss() or F.binary_cross_entropy(output, target)
// =============================================================================
function computeLoss(predictions, targets) {
  const eps = 1e-12; // clamp to avoid log(0) = -Infinity
  let total = 0;
  for (let i = 0; i < predictions.length; i++) {
    const p = Math.max(eps, Math.min(1 - eps, predictions[i][0]));
    const y = targets[i];
    total -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
  }
  return total / predictions.length;
}

// =============================================================================
// SECTION 6 — BACKPROPAGATION
// Computes gradients of the loss w.r.t. every weight and bias using the chain rule.
// This is what PyTorch's loss.backward() does automatically via autograd.
//
// The key recurrence:
//   δ[L]   = a[L] − y                          (output layer, BCE + sigmoid shortcut)
//   δ[l]   = (W[l]ᵀ · δ[l+1]) ⊙ σ'(z[l])    (hidden layers, ⊙ = elementwise)
//   dW[l]  = δ[l+1] · a[l]ᵀ                   (outer product)
//   db[l]  = δ[l+1]
//
// The "BCE + sigmoid" shortcut: when BCE loss is paired with sigmoid output,
// the combined gradient simplifies to (output − target). This cancels the
// sigmoid derivative in the output layer and avoids numerical instability.
// =============================================================================
function backprop(targets, activations, preActivations, weights, hiddenActivationTypes) {
  const L = weights.length;

  // Initialize gradient arrays with zeros, same shape as weights/biases
  const dWeights = weights.map(W => W.map(row => row.map(() => 0)));
  const dBiases = weights.map((W, l) => new Array(W.length).fill(0));

  // δ[l] stored for each layer l from 1..L
  const deltas = new Array(L + 1).fill(null);

  // Output layer delta: δ[L] = a[L] − y  (per sample, averaged across batch)
  const outputA = activations[L];
  deltas[L] = outputA.map((val, j) => val - targets[j]);

  // Backpropagate the error signal from layer L-1 down to layer 1
  for (let l = L - 1; l >= 1; l--) {
    // The activation at layer l is hiddenActivationTypes[l-1]
    // (layer 0 is input, layer 1 is first hidden layer, etc.)
    const activationType = hiddenActivationTypes[l - 1] || 'relu';
    const actDeriv = ACTIVATIONS[activationType].derivative;
    const z = preActivations[l];
    const nextDelta = deltas[l + 1];
    const nextW = weights[l]; // W[l] connects layer l to layer l+1

    // δ[l] = (W[l]ᵀ · δ[l+1]) ⊙ activation'(z[l])
    deltas[l] = z.map((zVal, j) => {
      // Sum_{k} W[l][k][j] * δ[l+1][k]  — transposed matrix multiply
      let sum = 0;
      for (let k = 0; k < nextDelta.length; k++) {
        sum += nextW[k][j] * nextDelta[k];
      }
      // Multiply by the local derivative (chain rule)
      return sum * actDeriv(zVal);
    });
  }

  // Compute weight and bias gradients from deltas
  for (let l = 0; l < L; l++) {
    const delta = deltas[l + 1]; // error signal flowing into layer l+1
    const prevA = activations[l]; // activations from layer l (the inputs to this weight matrix)

    for (let j = 0; j < delta.length; j++) {
      // dW[l][j][k] = δ[l+1][j] · a[l][k]  (outer product row j)
      for (let k = 0; k < prevA.length; k++) {
        dWeights[l][j][k] += delta[j] * prevA[k];
      }
      // db[l][j] = δ[l+1][j]
      dBiases[l][j] += delta[j];
    }
  }

  return { dWeights, dBiases, deltas };
}

// =============================================================================
// SECTION 7 — GRADIENT DESCENT UPDATE
// W[l] ← W[l] − lr · dW[l]   (subtract because we want to minimize loss)
// b[l] ← b[l] − lr · db[l]
//
// PyTorch equivalent:
//   optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
//   optimizer.step()
// =============================================================================
function updateWeights(weights, biases, dWeights, dBiases, lr) {
  const newWeights = weights.map((W, l) =>
    W.map((row, j) =>
      row.map((w, k) => w - lr * dWeights[l][j][k])
    )
  );
  const newBiases = biases.map((b, l) =>
    b.map((val, j) => val - lr * dBiases[l][j])
  );
  return { weights: newWeights, biases: newBiases };
}

// =============================================================================
// SECTION 8 — ONE TRAINING EPOCH (FULL BATCH)
// Runs one complete forward+backward+update cycle over all training examples.
// We use full-batch gradient descent here (not mini-batch/SGD).
// In full batch, we sum gradients over all examples then average.
//
// PyTorch equivalent:
//   for epoch in range(num_epochs):
//     optimizer.zero_grad()
//     output = model(X)
//     loss = criterion(output, y)
//     loss.backward()
//     optimizer.step()
// =============================================================================
function trainOneEpoch(weights, biases, hiddenActivationTypes, lr) {
  const layerSizes = deriveLayerSizes(weights);
  const L = weights.length;
  const N = XOR_DATA.length;

  // Accumulated gradients across the batch
  const totalDW = weights.map(W => W.map(row => row.map(() => 0)));
  const totalDB = weights.map((W, l) => new Array(W.length).fill(0));

  // Collect all predictions for loss computation
  const allPredictions = [];
  const allTargets = [];
  const allForwardData = [];

  for (const { input, label } of XOR_DATA) {
    const { activations, preActivations } = forwardPass(
      input, weights, biases, hiddenActivationTypes
    );
    allPredictions.push(activations[L]);
    allTargets.push(label);
    allForwardData.push({ activations, preActivations });

    // Backprop for this sample
    const { dWeights, dBiases } = backprop(
      [label], activations, preActivations, weights, hiddenActivationTypes
    );

    // Accumulate gradients (will average after the loop)
    for (let l = 0; l < L; l++) {
      for (let j = 0; j < dWeights[l].length; j++) {
        for (let k = 0; k < dWeights[l][j].length; k++) {
          totalDW[l][j][k] += dWeights[l][j][k];
        }
        totalDB[l][j] += dBiases[l][j];
      }
    }
  }

  // Average gradients over the batch, then update weights
  const avgDW = totalDW.map(W => W.map(row => row.map(v => v / N)));
  const avgDB = totalDB.map(b => b.map(v => v / N));

  const { weights: newWeights, biases: newBiases } = updateWeights(
    weights, biases, avgDW, avgDB, lr
  );

  const loss = computeLoss(allPredictions, allTargets);

  return { weights: newWeights, biases: newBiases, loss, avgDW, avgDB, allForwardData };
}

// Helper: infer layer sizes from weight matrices
function deriveLayerSizes(weights) {
  if (weights.length === 0) return [];
  const sizes = [weights[0][0].length];
  for (const W of weights) {
    sizes.push(W.length);
  }
  return sizes;
}

// =============================================================================
// SECTION 9 — DECISION BOUNDARY COMPUTATION
// Sample a grid over [0,1]×[0,1] and run forward pass at each point.
// Returns a 2D array of predicted probabilities.
// This is how you'd visualize a classifier's decision surface in PyTorch too —
// just pass a meshgrid through the model with torch.no_grad().
// =============================================================================
function computeDecisionBoundary(weights, biases, hiddenActivationTypes, gridSize = 40) {
  const grid = [];
  const L = weights.length;
  for (let row = 0; row < gridSize; row++) {
    const gridRow = [];
    for (let col = 0; col < gridSize; col++) {
      // Map grid indices to [0,1] space.
      // row=0 is the top of the canvas; the y-axis convention used everywhere
      // else in this file places x₂=1 at the top (cy = (1-x₂)·H).
      // Flipping here keeps the boundary colours aligned with the drawn points.
      const x1 = col / (gridSize - 1);
      const x2 = 1 - row / (gridSize - 1);
      const { activations } = forwardPass([x1, x2], weights, biases, hiddenActivationTypes);
      gridRow.push(activations[L][0]); // output probability
    }
    grid.push(gridRow);
  }
  return grid;
}

// =============================================================================
// SECTION 10 — PYTORCH CODE GENERATOR
// Generates a PyTorch code string reflecting the current network architecture.
// Updates live as the user changes layers/neurons/activations.
// =============================================================================
function generatePyTorchCode(layerSizes, hiddenActivationTypes) {
  const activationImports = [...new Set(hiddenActivationTypes)];
  const actMap = { relu: 'nn.ReLU()', tanh: 'nn.Tanh()', sigmoid: 'nn.Sigmoid()' };

  let layers = '';
  // Input → first hidden
  layers += `    nn.Linear(${layerSizes[0]}, ${layerSizes[1]}),  # W shape: [${layerSizes[1]}, ${layerSizes[0]}]\n`;
  layers += `    ${actMap[hiddenActivationTypes[0]] || 'nn.ReLU()'},\n`;

  // Hidden → hidden
  for (let i = 1; i < layerSizes.length - 2; i++) {
    layers += `    nn.Linear(${layerSizes[i]}, ${layerSizes[i + 1]}),  # W shape: [${layerSizes[i + 1]}, ${layerSizes[i]}]\n`;
    layers += `    ${actMap[hiddenActivationTypes[i]] || 'nn.ReLU()'},\n`;
  }

  // Last hidden → output
  const lastHidden = layerSizes[layerSizes.length - 2];
  layers += `    nn.Linear(${lastHidden}, 1),  # Output: single logit\n`;
  layers += `    nn.Sigmoid(),                  # → probability in (0, 1)\n`;

  return `import torch
import torch.nn as nn

# Network architecture: ${layerSizes.join(' → ')} → 1
# Hidden activations: ${hiddenActivationTypes.join(', ')}
model = nn.Sequential(
${layers})

# Loss: Binary Cross-Entropy
# Matches our JS: -(y*log(p) + (1-y)*log(1-p))
criterion = nn.BCELoss()

# Optimizer: vanilla SGD (same as our gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# XOR dataset
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()   # Clear old gradients
    output = model(X)       # Forward pass
    loss = criterion(output, y)  # BCE loss
    loss.backward()         # Backprop: computes all dW, db
    optimizer.step()        # W -= lr * dW  for each parameter`;
}

// =============================================================================
// SECTION 11 — COLOR UTILITIES
// Map a value in [0,1] to an RGB color for visualization.
// =============================================================================

// Activation magnitude → color (blue=low, orange=high)
function activationColor(value, alpha = 1) {
  // Clamp to [-1, 1] range for display (activations can be negative with tanh/relu)
  const t = Math.max(0, Math.min(1, (value + 1) / 2));
  const r = Math.round(t * 251 + (1 - t) * 30);
  const g = Math.round(t * 146 + (1 - t) * 144);
  const b = Math.round(t * 60 + (1 - t) * 255);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Decision boundary: class 0 = blue, class 1 = orange, interpolated by probability
function boundaryColor(prob) {
  const r = Math.round(prob * 251 + (1 - prob) * 59);
  const g = Math.round(prob * 146 + (1 - prob) * 130);
  const b = Math.round(prob * 60 + (1 - prob) * 246);
  return `rgb(${r},${g},${b})`;
}

// Gradient magnitude → heatmap color (gray=near-zero, red=large)
function gradientColor(magnitude, maxMag) {
  if (maxMag === 0) return 'rgba(100,100,100,0.3)';
  const t = Math.min(1, magnitude / maxMag);
  const r = Math.round(t * 239 + (1 - t) * 75);
  const g = Math.round(t * 68 + (1 - t) * 85);
  const b = Math.round(t * 68 + (1 - t) * 99);
  return `rgba(${r},${g},${b},${0.3 + t * 0.7})`;
}

// =============================================================================
// SECTION 12 — NETWORK GRAPH LAYOUT
// Compute SVG positions for each neuron given layer sizes.
// =============================================================================
function computeLayout(layerSizes, svgWidth, svgHeight) {
  const padding = { x: 60, y: 40 };
  const usableW = svgWidth - padding.x * 2;
  const usableH = svgHeight - padding.y * 2;
  const maxNeurons = Math.max(...layerSizes);
  const layerCount = layerSizes.length;

  const positions = layerSizes.map((count, li) => {
    const x = padding.x + (li / (layerCount - 1)) * usableW;
    return Array.from({ length: count }, (_, ni) => {
      const y = count === 1
        ? svgHeight / 2
        : padding.y + (ni / (count - 1)) * usableH;
      return { x, y, layerIdx: li, neuronIdx: ni };
    });
  });

  return positions;
}

// =============================================================================
// COMPONENT: NetworkGraph
// Renders the MLP as an SVG with circles (neurons) and lines (weighted edges).
// Colors neurons by activation magnitude and edges by gradient magnitude.
// =============================================================================
function NetworkGraph({ layerSizes, hiddenActivationTypes, forwardData, backpropData, animatingLayer }) {
  const SVG_W = 520;
  const SVG_H = 340;
  const NEURON_R = 18;

  const layout = computeLayout(layerSizes, SVG_W, SVG_H);

  // Compute max gradient magnitude for edge color normalization
  let maxGradMag = 0;
  if (backpropData) {
    for (const W of backpropData.dWeights) {
      for (const row of W) {
        for (const v of row) {
          if (Math.abs(v) > maxGradMag) maxGradMag = Math.abs(v);
        }
      }
    }
  }

  return (
    <svg width={SVG_W} height={SVG_H} className="w-full h-full">
      {/* Draw edges (connections between adjacent layers) */}
      {layout.slice(0, -1).map((fromLayer, li) =>
        fromLayer.map((from, fi) =>
          layout[li + 1].map((to, ti) => {
            // Determine edge color based on gradient (if backprop ran) or default gray
            let edgeColor = 'rgba(148,163,184,0.15)';
            let strokeW = 1;

            if (backpropData && backpropData.dWeights[li]) {
              const gradMag = Math.abs(backpropData.dWeights[li][ti][fi]);
              edgeColor = gradientColor(gradMag, maxGradMag);
              strokeW = 1 + 2 * (gradMag / (maxGradMag || 1));
            }

            // Highlight edges in the currently animating layer
            const isActive = animatingLayer >= 0 && (li === animatingLayer - 1 || li === animatingLayer);
            if (isActive) {
              edgeColor = 'rgba(96,165,250,0.5)';
              strokeW = 1.5;
            }

            return (
              <line
                key={`edge-${li}-${fi}-${ti}`}
                x1={from.x} y1={from.y}
                x2={to.x} y2={to.y}
                stroke={edgeColor}
                strokeWidth={strokeW}
                className="edge-line"
              />
            );
          })
        )
      )}

      {/* Draw neurons */}
      {layout.map((layer, li) =>
        layer.map((pos, ni) => {
          // Get activation value for this neuron (if forward pass has run)
          const actVal = forwardData?.activations?.[li]?.[ni];
          const hasAct = actVal !== undefined;

          // Determine fill color based on activation
          let fillColor = '#1e293b'; // default dark
          if (hasAct) {
            fillColor = activationColor(actVal);
          }

          // Dim neurons beyond the currently animating layer
          const isDimmed = animatingLayer >= 0 && li > animatingLayer;

          // Layer label
          let layerLabel = '';
          if (li === 0) layerLabel = 'Input';
          else if (li === layerSizes.length - 1) layerLabel = 'Output';
          else {
            const actType = hiddenActivationTypes[li - 1] || 'relu';
            layerLabel = ACTIVATIONS[actType].label;
          }

          return (
            <g key={`neuron-${li}-${ni}`} opacity={isDimmed ? 0.3 : 1}>
              {/* Outer ring shows layer type */}
              <circle
                cx={pos.x} cy={pos.y} r={NEURON_R + 2}
                fill="none"
                stroke={li === 0 ? '#64748b' : li === layerSizes.length - 1 ? '#f59e0b' :
                  ACTIVATIONS[hiddenActivationTypes[li - 1] || 'relu'].color}
                strokeWidth={1}
                opacity={0.5}
              />
              {/* Neuron body */}
              <circle
                cx={pos.x} cy={pos.y} r={NEURON_R}
                fill={fillColor}
                stroke={hasAct ? '#e2e8f0' : '#475569'}
                strokeWidth={1.5}
                className="neuron-circle"
              />
              {/* Activation value text */}
              {hasAct && (
                <text
                  x={pos.x} y={pos.y + 1}
                  textAnchor="middle" dominantBaseline="middle"
                  fill="white" fontSize={9} fontWeight="bold"
                  fontFamily="monospace"
                >
                  {actVal.toFixed(2)}
                </text>
              )}
              {/* Neuron index (small, below) */}
              {!hasAct && (
                <text
                  x={pos.x} y={pos.y + 1}
                  textAnchor="middle" dominantBaseline="middle"
                  fill="#94a3b8" fontSize={9}
                >
                  {li === 0 ? (ni === 0 ? 'x₁' : 'x₂') : `n${ni + 1}`}
                </text>
              )}
            </g>
          );
        })
      )}

      {/* Layer labels at top */}
      {layout.map((layer, li) => {
        const x = layer[0].x;
        let label = li === 0 ? 'Input (2)' : li === layerSizes.length - 1 ? 'Output (1)' :
          `Hidden ${li} (${layerSizes[li]})`;
        return (
          <text key={`label-${li}`} x={x} y={16}
            textAnchor="middle" fill="#94a3b8" fontSize={10}
            fontFamily="sans-serif"
          >
            {label}
          </text>
        );
      })}

      {/* Gradient magnitude legend (if backprop data exists) */}
      {backpropData && (
        <g transform="translate(10, 300)">
          <text fill="#94a3b8" fontSize={9}>Gradient:</text>
          {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
            <rect key={i} x={55 + i * 12} y={-8} width={12} height={10}
              fill={gradientColor(t, 1)} />
          ))}
          <text x={55} y={12} fill="#94a3b8" fontSize={8}>0</text>
          <text x={100} y={12} fill="#94a3b8" fontSize={8}>max</text>
        </g>
      )}
    </svg>
  );
}

// =============================================================================
// COMPONENT: DecisionBoundaryCanvas
// Renders a 40×40 grid colored by the network's output probability.
// Overlays the XOR training points.
// This updates every time training changes the weights.
// =============================================================================
function DecisionBoundaryCanvas({ weights, biases, hiddenActivationTypes, inferencePoint, onClick }) {
  const canvasRef = useRef(null);
  const GRID = 40;
  const CANVAS_SIZE = 260;
  const POINT_R = 7;

  useEffect(() => {
    if (!weights || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const cellSize = CANVAS_SIZE / GRID;

    // Compute and draw decision boundary grid
    const grid = computeDecisionBoundary(weights, biases, hiddenActivationTypes, GRID);

    for (let row = 0; row < GRID; row++) {
      for (let col = 0; col < GRID; col++) {
        const prob = grid[row][col];
        ctx.fillStyle = boundaryColor(prob);
        ctx.globalAlpha = 0.6;
        ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
      }
    }

    // Draw XOR training points on top
    ctx.globalAlpha = 1;
    for (const { input, label } of XOR_DATA) {
      const cx = input[0] * CANVAS_SIZE;
      const cy = (1 - input[1]) * CANVAS_SIZE; // flip y: y=0 at bottom in math, top in canvas

      // Outer ring
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_R + 2, 0, Math.PI * 2);
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Fill: orange for class 1, blue for class 0
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_R, 0, Math.PI * 2);
      ctx.fillStyle = label === 1 ? '#f97316' : '#3b82f6';
      ctx.fill();

      // Label
      ctx.fillStyle = 'white';
      ctx.font = 'bold 9px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label.toString(), cx, cy);
    }

    // Draw inference point if provided
    if (inferencePoint) {
      const cx = inferencePoint.x * CANVAS_SIZE;
      const cy = (1 - inferencePoint.y) * CANVAS_SIZE;
      const pred = inferencePoint.prediction;

      ctx.beginPath();
      ctx.arc(cx, cy, POINT_R + 3, 0, Math.PI * 2);
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = pred > 0.5 ? '#f97316' : '#3b82f6';
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_R, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('x₁: 0 → 1', CANVAS_SIZE / 2, CANVAS_SIZE - 4);
  }, [weights, biases, hiddenActivationTypes, inferencePoint]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="rounded border border-slate-700 block cursor-crosshair"
        onClick={onClick}
        style={{ cursor: onClick ? 'crosshair' : 'default' }}
      />
      <div className="absolute top-1 left-1 text-xs text-slate-500 font-mono pointer-events-none">
        x₂ ↑
      </div>
    </div>
  );
}

// =============================================================================
// COMPONENT: ConceptCallout
// Surfaces explanatory text at key training moments.
// =============================================================================
function ConceptCallout({ type, onDismiss }) {
  const callouts = {
    firstForward: {
      title: 'First Forward Pass',
      color: 'border-blue-500',
      icon: '→',
      body: `Each neuron computes z = W·x + b, then a = activation(z).
Values flow left to right. The output is a probability between 0 and 1 (sigmoid).`,
      pytorch: 'output = model(input_tensor)  # calls model.forward()',
    },
    firstBackprop: {
      title: 'Backpropagation Running',
      color: 'border-violet-500',
      icon: '←',
      body: `Gradients flow right to left. Each weight learns how much it contributed to the error.
The chain rule multiplies local derivatives layer by layer.`,
      pytorch: 'loss.backward()  # PyTorch autograd computes all ∂L/∂W',
    },
    lossPlateauing: {
      title: 'Loss Plateauing',
      color: 'border-amber-500',
      icon: '⚠',
      body: `Loss stopped decreasing. Try: raising the learning rate, adding neurons/layers,
or switching activation functions. XOR needs at least one hidden layer.`,
      pytorch: '# Try: optimizer = torch.optim.Adam(model.parameters(), lr=0.01)',
    },
    vanishingGradient: {
      title: 'Vanishing Gradient Detected',
      color: 'border-red-500',
      icon: '⚠',
      body: `Gradients near the input layer are near zero. Sigmoid activations saturate —
their derivative max is 0.25, so deep chains approach 0. Try ReLU instead.`,
      pytorch: '# Replace nn.Sigmoid() with nn.ReLU() in hidden layers',
    },
    inferencePoint: {
      title: 'Inference Mode',
      color: 'border-emerald-500',
      icon: '◎',
      body: `The network runs a forward pass on your test point using the trained weights.
No gradients are computed — we only need the output probability.`,
      pytorch: `model.eval()           # disable dropout/batchnorm training behavior
with torch.no_grad():  # skip gradient tracking (saves memory)
    pred = model(test_point)`,
    },
  };

  const c = callouts[type];
  if (!c) return null;

  return (
    <div className={`border-l-4 ${c.color} bg-slate-800/80 rounded-r p-3 mb-3 text-sm`}>
      <div className="flex justify-between items-start">
        <span className="font-bold text-white flex items-center gap-2">
          <span>{c.icon}</span> {c.title}
        </span>
        <button onClick={onDismiss} className="text-slate-500 hover:text-white ml-2">✕</button>
      </div>
      <p className="text-slate-300 mt-1 text-xs leading-relaxed">{c.body}</p>
      {c.pytorch && (
        <pre className="mt-2 text-xs bg-black/40 rounded p-2 text-emerald-400 overflow-x-auto">
          {c.pytorch}
        </pre>
      )}
    </div>
  );
}

// =============================================================================
// COMPONENT: PyTorchSidebar
// Displays a live PyTorch code snippet reflecting the current network config.
// =============================================================================
function PyTorchSidebar({ layerSizes, hiddenActivationTypes }) {
  const code = generatePyTorchCode(layerSizes, hiddenActivationTypes);
  return (
    <div className="bg-gray-950 rounded-lg border border-slate-700 p-3 h-full overflow-auto">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full bg-red-500" />
        <div className="w-3 h-3 rounded-full bg-yellow-500" />
        <div className="w-3 h-3 rounded-full bg-green-500" />
        <span className="text-slate-400 text-xs ml-2 font-mono">model.py</span>
      </div>
      <pre className="text-xs font-mono text-slate-300 leading-relaxed whitespace-pre-wrap">
        {code}
      </pre>
    </div>
  );
}

// =============================================================================
// MAIN APP COMPONENT
// =============================================================================
export default function App() {
  // ---- Architecture Config ----
  const [numHiddenLayers, setNumHiddenLayers] = useState(2);
  const [neuronsPerLayer, setNeuronsPerLayer] = useState([4, 4]);
  const [activationTypes, setActivationTypes] = useState(['relu', 'relu']);

  // ---- Network State ----
  // { weights: [...], biases: [...] }
  const [network, setNetwork] = useState(null);

  // ---- Training State ----
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [lossHistory, setLossHistory] = useState([]);
  const [learningRate, setLearningRate] = useState(0.1);
  const [lastGradients, setLastGradients] = useState(null);
  const [lastForwardData, setLastForwardData] = useState(null);

  // ---- Animation ----
  const [animatingLayer, setAnimatingLayer] = useState(-1);
  const [forwardPassDisplay, setForwardPassDisplay] = useState(null);

  // ---- Inference ----
  const [inferencePoint, setInferencePoint] = useState(null);

  // ---- Concept Callouts ----
  const [callouts, setCallouts] = useState(new Set());
  const [dismissedCallouts, setDismissedCallouts] = useState(new Set());

  // Refs for training loop
  const trainingRef = useRef(false);
  const networkRef = useRef(null);
  const epochRef = useRef(0);
  const lossHistoryRef = useRef([]);

  // Derived layer sizes: [2, ...hidden..., 1]
  const layerSizes = [2, ...neuronsPerLayer.slice(0, numHiddenLayers), 1];

  // ---- Initialize network when architecture changes ----
  const initializeNetwork = useCallback(() => {
    const net = initNetwork(layerSizes);
    setNetwork(net);
    networkRef.current = net;
    setEpoch(0);
    epochRef.current = 0;
    setLossHistory([]);
    lossHistoryRef.current = [];
    setLastGradients(null);
    setLastForwardData(null);
    setForwardPassDisplay(null);
    setInferencePoint(null);
    setIsTraining(false);
    trainingRef.current = false;
    setCallouts(new Set());
  }, [layerSizes.join(',')]);

  // Initialize on mount
  useEffect(() => {
    initializeNetwork();
  }, []);

  // Re-init when architecture changes (but not on first mount)
  const prevArchRef = useRef(null);
  useEffect(() => {
    const archKey = layerSizes.join(',') + activationTypes.join(',');
    if (prevArchRef.current !== null && prevArchRef.current !== archKey) {
      initializeNetwork();
    }
    prevArchRef.current = archKey;
  }, [layerSizes.join(','), activationTypes.join(',')]);

  // ---- Architecture change handlers ----
  const handleNumHiddenLayersChange = (n) => {
    const clamped = Math.max(1, Math.min(4, n));
    setNumHiddenLayers(clamped);
    setNeuronsPerLayer(prev => {
      const next = [...prev];
      while (next.length < clamped) next.push(4);
      return next.slice(0, clamped);
    });
    setActivationTypes(prev => {
      const next = [...prev];
      while (next.length < clamped) next.push('relu');
      return next.slice(0, clamped);
    });
  };

  const handleNeuronsChange = (layerIdx, n) => {
    setNeuronsPerLayer(prev => {
      const next = [...prev];
      next[layerIdx] = Math.max(2, Math.min(8, n));
      return next;
    });
  };

  const handleActivationChange = (layerIdx, type) => {
    setActivationTypes(prev => {
      const next = [...prev];
      next[layerIdx] = type;
      return next;
    });
  };

  // ---- Forward Pass Animation ----
  const runForwardPassAnimation = useCallback(async () => {
    if (!networkRef.current) return;
    const { weights, biases } = networkRef.current;
    const input = XOR_DATA[0].input; // animate on first XOR sample [0,0]

    // Run the full forward pass to get all values
    const { activations, preActivations } = forwardPass(
      input, weights, biases, activationTypes
    );

    // Show trigger callout
    if (!dismissedCallouts.has('firstForward')) {
      setCallouts(prev => new Set([...prev, 'firstForward']));
    }

    // Animate layer by layer
    for (let l = 0; l <= layerSizes.length - 1; l++) {
      setAnimatingLayer(l);
      // Show activations up to layer l
      const partial = { activations: activations.slice(0, l + 1), preActivations };
      setForwardPassDisplay(partial);
      await new Promise(r => setTimeout(r, 500));
    }

    setAnimatingLayer(-1);
    setForwardPassDisplay({ activations, preActivations });
    setLastForwardData({ activations, preActivations });
  }, [activationTypes, layerSizes.length, dismissedCallouts]);

  // ---- Training Loop ----
  const runTrainingStep = useCallback(() => {
    if (!networkRef.current) return;
    const { weights, biases } = networkRef.current;

    const result = trainOneEpoch(weights, biases, activationTypes, learningRate);

    networkRef.current = { weights: result.weights, biases: result.biases };
    setNetwork({ weights: result.weights, biases: result.biases });

    epochRef.current += 1;
    setEpoch(epochRef.current);

    const lossEntry = { epoch: epochRef.current, loss: result.loss };
    lossHistoryRef.current = [...lossHistoryRef.current, lossEntry];
    // Keep last 200 points for display
    if (lossHistoryRef.current.length > 200) {
      lossHistoryRef.current = lossHistoryRef.current.slice(-200);
    }
    setLossHistory([...lossHistoryRef.current]);
    setLastGradients({ dWeights: result.avgDW, dBiases: result.avgDB });
    setLastForwardData(result.allForwardData[0]); // first XOR sample for display

    // Trigger callouts
    if (epochRef.current === 1 && !dismissedCallouts.has('firstBackprop')) {
      setCallouts(prev => new Set([...prev, 'firstBackprop']));
    }

    // Detect loss plateauing (loss hasn't changed much in last 20 epochs)
    const hist = lossHistoryRef.current;
    if (hist.length > 20 && !dismissedCallouts.has('lossPlateauing')) {
      const recent = hist.slice(-20);
      const delta = Math.abs(recent[0].loss - recent[recent.length - 1].loss);
      if (delta < 0.005 && result.loss > 0.1) {
        setCallouts(prev => new Set([...prev, 'lossPlateauing']));
      }
    }

    // Detect vanishing gradient: check if input-layer gradients are tiny
    const firstLayerGrads = result.avgDW[0];
    const maxFirstGrad = Math.max(...firstLayerGrads.flat().map(Math.abs));
    const maxAnyGrad = Math.max(...result.avgDW.flat().flat().map(Math.abs));
    if (maxAnyGrad > 0 && maxFirstGrad / maxAnyGrad < 0.01 && !dismissedCallouts.has('vanishingGradient')) {
      setCallouts(prev => new Set([...prev, 'vanishingGradient']));
    }

    return result.loss;
  }, [activationTypes, learningRate, dismissedCallouts]);

  // Training loop using requestAnimationFrame
  const trainingLoopRef = useRef(null);
  useEffect(() => {
    if (!isTraining) {
      if (trainingLoopRef.current) cancelAnimationFrame(trainingLoopRef.current);
      return;
    }

    let lastTime = 0;
    const STEPS_PER_FRAME = 5; // run several epochs per animation frame for speed

    const loop = (timestamp) => {
      if (!trainingRef.current) return;
      for (let i = 0; i < STEPS_PER_FRAME; i++) {
        runTrainingStep();
      }
      trainingLoopRef.current = requestAnimationFrame(loop);
    };

    trainingRef.current = true;
    trainingLoopRef.current = requestAnimationFrame(loop);
    return () => {
      if (trainingLoopRef.current) cancelAnimationFrame(trainingLoopRef.current);
    };
  }, [isTraining, runTrainingStep]);

  const handleToggleTraining = () => {
    if (isTraining) {
      trainingRef.current = false;
      setIsTraining(false);
    } else {
      setIsTraining(true);
    }
  };

  const handleStepEpoch = () => {
    if (isTraining) return;
    runTrainingStep();
  };

  const handleReset = () => {
    trainingRef.current = false;
    setIsTraining(false);
    initializeNetwork();
  };

  // ---- Inference: click on boundary canvas ----
  const handleCanvasClick = (e) => {
    if (!networkRef.current) return;
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const CANVAS_SIZE = 260;
    const cx = (e.clientX - rect.left) / rect.width;
    const cy = (e.clientY - rect.top) / rect.height;
    // Convert canvas coordinates to [0,1] input space
    const x1 = cx;
    const x2 = 1 - cy; // flip y

    const { weights, biases } = networkRef.current;
    const { activations } = forwardPass([x1, x2], weights, biases, activationTypes);
    const prediction = activations[activations.length - 1][0];

    setInferencePoint({ x: x1, y: x2, prediction });

    if (!dismissedCallouts.has('inferencePoint')) {
      setCallouts(prev => new Set([...prev, 'inferencePoint']));
    }
  };

  // ---- Dismiss callout ----
  const dismissCallout = (type) => {
    setCallouts(prev => { const s = new Set(prev); s.delete(type); return s; });
    setDismissedCallouts(prev => new Set([...prev, type]));
  };

  // Latest loss for display
  const latestLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].loss : null;

  // Active callouts to show (max 2 at once to avoid clutter)
  const activeCallouts = [...callouts].filter(c => !dismissedCallouts.has(c)).slice(0, 2);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans">
      {/* Header */}
      <header className="border-b border-slate-700 px-6 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white">Neural Network Learning Tool</h1>
          <p className="text-xs text-slate-400">Phase 1 MVP — real math, XOR dataset</p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          {latestLoss !== null && (
            <span className="font-mono">
              Loss: <span className={`font-bold ${latestLoss < 0.1 ? 'text-emerald-400' : latestLoss < 0.3 ? 'text-amber-400' : 'text-red-400'}`}>
                {latestLoss.toFixed(4)}
              </span>
            </span>
          )}
          <span className="text-slate-500 font-mono">Epoch: {epoch}</span>
        </div>
      </header>

      <div className="flex h-[calc(100vh-56px)]">
        {/* ================================================================
            LEFT PANEL — Architecture Config + Controls
        ================================================================ */}
        <div className="w-56 border-r border-slate-700 p-4 flex flex-col gap-4 overflow-y-auto">
          <section>
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Architecture</h2>

            <label className="block mb-3">
              <span className="text-xs text-slate-400">Hidden Layers</span>
              <div className="flex items-center gap-2 mt-1">
                <button onClick={() => handleNumHiddenLayersChange(numHiddenLayers - 1)}
                  className="w-6 h-6 rounded bg-slate-700 hover:bg-slate-600 text-white text-sm font-bold">−</button>
                <span className="font-mono text-white w-4 text-center">{numHiddenLayers}</span>
                <button onClick={() => handleNumHiddenLayersChange(numHiddenLayers + 1)}
                  className="w-6 h-6 rounded bg-slate-700 hover:bg-slate-600 text-white text-sm font-bold">+</button>
              </div>
            </label>

            {Array.from({ length: numHiddenLayers }, (_, i) => (
              <div key={i} className="mb-3 pl-2 border-l-2 border-slate-700">
                <div className="text-xs text-slate-400 mb-1">Layer {i + 1}</div>
                <div className="flex items-center gap-1 mb-1">
                  <span className="text-xs text-slate-500">Neurons:</span>
                  <button onClick={() => handleNeuronsChange(i, neuronsPerLayer[i] - 1)}
                    className="w-5 h-5 rounded bg-slate-700 hover:bg-slate-600 text-xs">−</button>
                  <span className="font-mono text-white text-xs w-4 text-center">{neuronsPerLayer[i]}</span>
                  <button onClick={() => handleNeuronsChange(i, neuronsPerLayer[i] + 1)}
                    className="w-5 h-5 rounded bg-slate-700 hover:bg-slate-600 text-xs">+</button>
                </div>
                <select
                  value={activationTypes[i]}
                  onChange={e => handleActivationChange(i, e.target.value)}
                  className="w-full bg-slate-800 border border-slate-600 rounded text-xs text-white p-1"
                >
                  <option value="relu">ReLU</option>
                  <option value="tanh">Tanh</option>
                  <option value="sigmoid">Sigmoid</option>
                </select>
              </div>
            ))}

            <div className="text-xs text-slate-500 mt-1 font-mono">
              Architecture: {layerSizes.join(' → ')}
            </div>
          </section>

          <section>
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Training</h2>

            <label className="block mb-3">
              <span className="text-xs text-slate-400">Learning Rate: {learningRate.toFixed(3)}</span>
              <input
                type="range" min="0.001" max="1" step="0.001"
                value={learningRate}
                onChange={e => setLearningRate(parseFloat(e.target.value))}
                className="w-full mt-1 accent-blue-500"
              />
              <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                <span>0.001</span><span>1.0</span>
              </div>
            </label>

            <div className="flex flex-col gap-2">
              <button
                onClick={handleToggleTraining}
                className={`w-full py-2 rounded font-bold text-sm transition-colors ${
                  isTraining
                    ? 'bg-amber-600 hover:bg-amber-500 text-white'
                    : 'bg-blue-600 hover:bg-blue-500 text-white'
                }`}
              >
                {isTraining ? '⏸ Pause' : '▶ Train'}
              </button>

              <button
                onClick={handleStepEpoch}
                disabled={isTraining}
                className="w-full py-1.5 rounded text-sm bg-slate-700 hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Step 1 Epoch
              </button>

              <button
                onClick={runForwardPassAnimation}
                disabled={isTraining}
                className="w-full py-1.5 rounded text-sm bg-slate-700 hover:bg-slate-600 disabled:opacity-40"
              >
                Forward Pass
              </button>

              <button
                onClick={handleReset}
                className="w-full py-1.5 rounded text-sm bg-slate-800 hover:bg-red-900/50 text-slate-400 hover:text-red-300 border border-slate-700"
              >
                Reset
              </button>
            </div>
          </section>

          {/* Inference instructions */}
          <section className="mt-auto">
            <div className="text-xs text-slate-500 bg-slate-800/50 rounded p-2">
              <span className="text-slate-400 font-medium block mb-1">Inference Mode</span>
              Click anywhere on the decision boundary to predict that point.
            </div>
            {inferencePoint && (
              <div className="mt-2 text-xs font-mono bg-slate-800 rounded p-2">
                <div>x₁={inferencePoint.x.toFixed(2)} x₂={inferencePoint.y.toFixed(2)}</div>
                <div className={inferencePoint.prediction > 0.5 ? 'text-orange-400' : 'text-blue-400'}>
                  Class: {inferencePoint.prediction > 0.5 ? 1 : 0}
                </div>
                <div>Conf: {(Math.abs(inferencePoint.prediction - 0.5) * 2 * 100).toFixed(0)}%</div>
                <div>p(1)={inferencePoint.prediction.toFixed(3)}</div>
              </div>
            )}
          </section>
        </div>

        {/* ================================================================
            CENTER PANEL — Network Graph + Decision Boundary
        ================================================================ */}
        <div className="flex-1 flex flex-col p-4 gap-4 overflow-hidden">
          {/* Callouts */}
          {activeCallouts.length > 0 && (
            <div className="flex flex-col gap-2">
              {activeCallouts.map(type => (
                <ConceptCallout key={type} type={type} onDismiss={() => dismissCallout(type)} />
              ))}
            </div>
          )}

          {/* Network Graph */}
          <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3 flex-1 min-h-0">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-slate-300">Network</h2>
              {animatingLayer >= 0 && (
                <span className="text-xs text-blue-400 font-mono animate-pulse">
                  Animating layer {animatingLayer}…
                </span>
              )}
              {lastGradients && !animatingLayer && (
                <span className="text-xs text-violet-400 font-mono">
                  Edges colored by ∂L/∂W
                </span>
              )}
            </div>
            <NetworkGraph
              layerSizes={layerSizes}
              hiddenActivationTypes={activationTypes}
              forwardData={forwardPassDisplay || lastForwardData}
              backpropData={lastGradients}
              animatingLayer={animatingLayer}
            />
          </div>

          {/* Bottom row: Decision Boundary + Loss Curve */}
          <div className="flex gap-4 h-72">
            {/* Decision Boundary */}
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
              <h2 className="text-sm font-semibold text-slate-300 mb-2">Decision Boundary</h2>
              <p className="text-xs text-slate-500 mb-2">Click to run inference</p>
              {network ? (
                  <DecisionBoundaryCanvas
                    weights={network.weights}
                    biases={network.biases}
                    hiddenActivationTypes={activationTypes}
                    inferencePoint={inferencePoint}
                    onClick={handleCanvasClick}
                  />
              ) : (
                <div className="w-[260px] h-[260px] bg-slate-800 rounded flex items-center justify-center text-slate-500 text-sm">
                  Initializing…
                </div>
              )}
              <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500 inline-block" /> Class 0</span>
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-orange-500 inline-block" /> Class 1</span>
              </div>
            </div>

            {/* Loss Curve */}
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3 flex-1 flex flex-col">
              <h2 className="text-sm font-semibold text-slate-300 mb-2">Training Loss (BCE)</h2>
              {lossHistory.length > 1 ? (
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lossHistory} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis
                        dataKey="epoch"
                        tick={{ fill: '#64748b', fontSize: 10 }}
                        label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5, fill: '#64748b', fontSize: 10 }}
                      />
                      <YAxis
                        domain={[0, 'auto']}
                        tick={{ fill: '#64748b', fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', fontSize: 11 }}
                        labelStyle={{ color: '#94a3b8' }}
                        itemStyle={{ color: '#60a5fa' }}
                        formatter={(v) => v.toFixed(4)}
                      />
                      <Line
                        type="monotone" dataKey="loss"
                        stroke="#60a5fa" strokeWidth={1.5}
                        dot={false} isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
                  Start training to see loss curve
                </div>
              )}
              {lossHistory.length > 1 && (
                <div className="text-xs text-slate-500 mt-1 font-mono">
                  Initial: {lossHistory[0]?.loss.toFixed(4)} → Current: {latestLoss?.toFixed(4)}
                  {latestLoss < 0.05 && <span className="text-emerald-400 ml-2">✓ Converged</span>}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ================================================================
            RIGHT PANEL — PyTorch Sidebar
        ================================================================ */}
        <div className="w-72 border-l border-slate-700 p-4 flex flex-col gap-3">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">PyTorch Equivalent</h2>
          <p className="text-xs text-slate-500">Updates live as you change the architecture. Code is explanatory — it doesn't run here.</p>
          <div className="flex-1 min-h-0">
            <PyTorchSidebar
              layerSizes={layerSizes}
              hiddenActivationTypes={activationTypes}
            />
          </div>
          {/* Weight summary */}
          {network && (
            <div className="bg-slate-800 rounded p-2 text-xs font-mono">
              <div className="text-slate-400 mb-1">Parameter counts:</div>
              {network.weights.map((W, l) => {
                const params = W.length * W[0].length + W.length;
                return (
                  <div key={l} className="text-slate-300">
                    Layer {l + 1}: {W[0].length}×{W.length} + {W.length}b = {params} params
                  </div>
                );
              })}
              <div className="text-blue-400 mt-1">
                Total: {network.weights.reduce((s, W) => s + W.length * W[0].length + W.length, 0)} params
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
