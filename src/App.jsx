// =============================================================================
// Neural Network Learning Tool — Phase 1 Hardened
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
    // Derivative is 1 when x > 0, 0 otherwise (undefined at 0 — we use 0).
    derivative: x => (x > 0 ? 1 : 0),
    color: '#60a5fa',
  },
  tanh: {
    label: 'Tanh',
    // tanh maps any real to (-1, 1). Zero-centered, so gradients are better behaved.
    fn: x => Math.tanh(x),
    // Derivative: 1 - tanh²(x). Peaks at 1 when x=0, approaches 0 for large |x|.
    derivative: x => 1 - Math.tanh(x) ** 2,
    color: '#a78bfa',
  },
  sigmoid: {
    label: 'Sigmoid',
    // σ(x) = 1 / (1 + e^−x). Maps any real to (0, 1).
    fn: x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
    // Derivative: σ(x)·(1−σ(x)). Max is 0.25 — causes vanishing gradients in deep nets.
    derivative: x => {
      const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
      return s * (1 - s);
    },
    color: '#f472b6',
  },
};

// =============================================================================
// SECTION 2 — XOR DATASET
// XOR is the canonical test because it is not linearly separable.
// A network with no hidden layers cannot solve it: the four corners of the
// unit square cannot be split into two groups by a single line.
// =============================================================================
const XOR_DATA = [
  { input: [0, 0], label: 0 },
  { input: [0, 1], label: 1 },
  { input: [1, 0], label: 1 },
  { input: [1, 1], label: 0 },
];

// =============================================================================
// SECTION 3 — NETWORK INITIALIZATION
// Xavier (Glorot) initialization: scale = sqrt(2 / (fan_in + fan_out)).
// This keeps activation variance roughly constant at the start of training,
// avoiding exploding or vanishing signals before any learning has happened.
// PyTorch: torch.nn.init.xavier_uniform_(layer.weight)
// =============================================================================
function initNetwork(layerSizes) {
  const weights = [];
  const biases = [];
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const fanIn  = layerSizes[l];
    const fanOut = layerSizes[l + 1];
    const scale  = Math.sqrt(2.0 / (fanIn + fanOut));
    // W[l] is [fanOut × fanIn]. W[l][j][k] = weight from neuron k → neuron j.
    const W = Array.from({ length: fanOut }, () =>
      Array.from({ length: fanIn }, () => (Math.random() * 2 - 1) * scale)
    );
    weights.push(W);
    biases.push(new Array(fanOut).fill(0)); // PyTorch Linear bias default
  }
  return { weights, biases };
}

// =============================================================================
// SECTION 4 — FORWARD PASS
// Propagates one input vector through all layers, storing z and a at every step.
// Returns all intermediates so backprop can reuse them without re-computing.
//
// PyTorch: output = model(input_tensor)   ← model.forward() does this internally
// =============================================================================
function forwardPass(input, weights, biases, hiddenActivationTypes) {
  const activations    = [input]; // activations[0] = the raw input
  const preActivations = [null];  // preActivations[l] = z[l], null for input layer

  const L = weights.length;
  for (let l = 0; l < L; l++) {
    const W      = weights[l];
    const b      = biases[l];
    const prevA  = activations[l];

    // z = W · prevA + b  (matrix-vector product)
    const z = W.map((row, j) => {
      let sum = b[j];
      for (let k = 0; k < prevA.length; k++) sum += row[k] * prevA[k];
      return sum;
    });

    // Activation: output layer always uses sigmoid (gives probability in (0,1));
    // hidden layers use the per-layer configured function.
    const isOutput     = l === L - 1;
    const activType    = isOutput ? 'sigmoid' : (hiddenActivationTypes[l] || 'relu');
    const activationFn = ACTIVATIONS[activType].fn;

    preActivations.push(z);
    activations.push(z.map(activationFn));
  }
  return { activations, preActivations };
}

// =============================================================================
// SECTION 5 — LOSS FUNCTION
// Binary Cross-Entropy (BCE):
//   L = −(1/N) · Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]
//
// Punishes confident wrong predictions very harshly (log → −∞ near 0).
// PyTorch: nn.BCELoss() or F.binary_cross_entropy(output, target)
// =============================================================================
function computeLoss(predictions, targets) {
  const eps = 1e-12; // clamp to prevent log(0) = −Infinity
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
// Computes gradients of loss w.r.t. every weight and bias via the chain rule.
// This is what PyTorch's loss.backward() does automatically.
//
// Key recurrence:
//   δ[L]   = a[L] − y                          ← output layer (BCE+sigmoid shortcut)
//   δ[l]   = (W[l]ᵀ · δ[l+1]) ⊙ σ'(z[l])    ← hidden layers (⊙ = elementwise)
//   dW[l]  = δ[l+1] · a[l]ᵀ                   ← outer product
//   db[l]  = δ[l+1]
//
// The "BCE + sigmoid shortcut": when BCE loss is combined with a sigmoid output,
// the gradient simplifies to (ŷ − y), cancelling the sigmoid derivative. This
// avoids numerical instability from σ'(z) approaching 0 in the output layer.
// =============================================================================
function backprop(targets, activations, preActivations, weights, hiddenActivationTypes) {
  const L = weights.length;
  const dWeights = weights.map(W => W.map(row => row.map(() => 0)));
  const dBiases  = weights.map(W => new Array(W.length).fill(0));
  const deltas   = new Array(L + 1).fill(null);

  // Output layer: δ[L] = a[L] − y
  deltas[L] = activations[L].map((val, j) => val - targets[j]);

  // Backpropagate through hidden layers: δ[l] = (W[l]ᵀ · δ[l+1]) ⊙ activation'(z[l])
  for (let l = L - 1; l >= 1; l--) {
    const activType = hiddenActivationTypes[l - 1] || 'relu';
    const actDeriv  = ACTIVATIONS[activType].derivative;
    const z         = preActivations[l];
    const nextDelta = deltas[l + 1];
    const nextW     = weights[l]; // W[l] connects layer l → l+1

    deltas[l] = z.map((zVal, j) => {
      // Transposed matrix-vector product: Σ_k W[l][k][j] · δ[l+1][k]
      let sum = 0;
      for (let k = 0; k < nextDelta.length; k++) sum += nextW[k][j] * nextDelta[k];
      return sum * actDeriv(zVal); // chain rule: multiply by local derivative
    });
  }

  // Compute weight and bias gradients from deltas
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

// =============================================================================
// SECTION 7 — GRADIENT DESCENT UPDATE
// W[l] ← W[l] − lr · ∂L/∂W[l]   (step opposite to the gradient = downhill)
// PyTorch: optimizer.step()  after optimizer.zero_grad() + loss.backward()
// =============================================================================
function updateWeights(weights, biases, dWeights, dBiases, lr) {
  return {
    weights: weights.map((W, l) =>
      W.map((row, j) => row.map((w, k) => w - lr * dWeights[l][j][k]))
    ),
    biases: biases.map((b, l) =>
      b.map((val, j) => val - lr * dBiases[l][j])
    ),
  };
}

// =============================================================================
// SECTION 8 — ONE TRAINING EPOCH (FULL BATCH)
// Runs forward → loss → backward → update over all 4 XOR examples.
// "Full-batch" means all gradients are accumulated, then averaged, before the
// single weight update. This is the simplest form of gradient descent.
//
// PyTorch:
//   optimizer.zero_grad()
//   output = model(X)
//   loss = criterion(output, y)
//   loss.backward()
//   optimizer.step()
// =============================================================================
function trainOneEpoch(weights, biases, hiddenActivationTypes, lr) {
  const L = weights.length;
  const N = XOR_DATA.length;

  // Gradient accumulators (same shape as weights / biases)
  const totalDW = weights.map(W => W.map(row => row.map(() => 0)));
  const totalDB = weights.map(W => new Array(W.length).fill(0));

  const allPredictions  = [];
  const allTargets      = [];
  const allForwardData  = [];

  for (const { input, label } of XOR_DATA) {
    const { activations, preActivations } = forwardPass(
      input, weights, biases, hiddenActivationTypes
    );
    allPredictions.push(activations[L]);
    allTargets.push(label);
    allForwardData.push({ activations, preActivations });

    const { dWeights, dBiases } = backprop(
      [label], activations, preActivations, weights, hiddenActivationTypes
    );

    // Accumulate; we'll average after all samples are processed
    for (let l = 0; l < L; l++) {
      for (let j = 0; j < dWeights[l].length; j++) {
        for (let k = 0; k < dWeights[l][j].length; k++) totalDW[l][j][k] += dWeights[l][j][k];
        totalDB[l][j] += dBiases[l][j];
      }
    }
  }

  const avgDW = totalDW.map(W => W.map(row => row.map(v => v / N)));
  const avgDB = totalDB.map(b => b.map(v => v / N));

  const { weights: newWeights, biases: newBiases } = updateWeights(
    weights, biases, avgDW, avgDB, lr
  );

  return {
    weights:      newWeights,
    biases:       newBiases,
    loss:         computeLoss(allPredictions, allTargets),
    avgDW,
    avgDB,
    allForwardData,
  };
}

// Helper: derive [inputSize, ...hiddenSizes, outputSize] from weight shapes
function deriveLayerSizes(weights) {
  if (weights.length === 0) return [];
  const sizes = [weights[0][0].length];
  for (const W of weights) sizes.push(W.length);
  return sizes;
}

// =============================================================================
// SECTION 9 — DECISION BOUNDARY COMPUTATION
// Sample a gridSize×gridSize grid over [0,1]² and run a real forward pass at
// each cell. The returned probabilities are used to color the canvas.
//
// This is NOT approximated or mocked. Every cell is an actual model.forward()
// call using the current trained weights. The y-axis is flipped (1 − row/…)
// so that row=0 (canvas top) corresponds to x₂=1, matching the convention
// used everywhere else: cy = (1 − x₂) · H puts x₂=0 at the bottom.
//
// PyTorch equivalent:
//   with torch.no_grad():
//     xx, yy = torch.meshgrid(torch.linspace(0,1,grid), torch.linspace(0,1,grid))
//     Z = model(torch.stack([xx.flatten(), yy.flatten()], dim=1)).reshape(grid, grid)
// =============================================================================
function computeDecisionBoundary(weights, biases, hiddenActivationTypes, gridSize = 40) {
  const L    = weights.length;
  const grid = [];
  for (let row = 0; row < gridSize; row++) {
    const gridRow = [];
    for (let col = 0; col < gridSize; col++) {
      const x1 = col / (gridSize - 1);
      // Flip y: row=0 is canvas top; x₂=1 should be at the top so that the
      // drawn XOR training points (which use cy = (1−x₂)·H) align with the heatmap.
      const x2 = 1 - row / (gridSize - 1);
      const { activations } = forwardPass([x1, x2], weights, biases, hiddenActivationTypes);
      gridRow.push(activations[L][0]); // output probability p(class=1)
    }
    grid.push(gridRow);
  }
  return grid;
}

// =============================================================================
// SECTION 10 — PYTORCH CODE GENERATOR
// Builds a PyTorch code string reflecting the current architecture.
// Updates live as the user changes layers, neurons, or activations.
// =============================================================================
function generatePyTorchCode(layerSizes, hiddenActivationTypes) {
  const actMap = { relu: 'nn.ReLU()', tanh: 'nn.Tanh()', sigmoid: 'nn.Sigmoid()' };
  let layers = '';
  layers += `    nn.Linear(${layerSizes[0]}, ${layerSizes[1]}),  # W: [${layerSizes[1]}×${layerSizes[0]}]\n`;
  layers += `    ${actMap[hiddenActivationTypes[0]] || 'nn.ReLU()'},\n`;
  for (let i = 1; i < layerSizes.length - 2; i++) {
    layers += `    nn.Linear(${layerSizes[i]}, ${layerSizes[i+1]}),  # W: [${layerSizes[i+1]}×${layerSizes[i]}]\n`;
    layers += `    ${actMap[hiddenActivationTypes[i]] || 'nn.ReLU()'},\n`;
  }
  const lastH = layerSizes[layerSizes.length - 2];
  layers += `    nn.Linear(${lastH}, 1),   # output logit\n`;
  layers += `    nn.Sigmoid(),              # → probability\n`;

  return `import torch, torch.nn as nn

# Architecture: ${layerSizes.join(' → ')} → 1
model = nn.Sequential(
${layers})

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1)

X = torch.tensor(
    [[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)
y = torch.tensor(
    [[0],[1],[1],[0]],         dtype=torch.float)

for epoch in range(max_epochs):
    optimizer.zero_grad()    # clear ∂L/∂W
    out  = model(X)          # forward pass
    loss = criterion(out, y) # BCE loss
    loss.backward()          # backprop
    optimizer.step()         # W -= lr·dW
    # Stop when solved:
    if loss.item() < 0.001:
        break`;
}

// =============================================================================
// SECTION 11 — COLOR UTILITIES
// =============================================================================

// Activation magnitude → blue (low/negative) to orange (high/positive)
function activationColor(value) {
  const t = Math.max(0, Math.min(1, (value + 1) / 2));
  return `rgba(${Math.round(t*251+(1-t)*30)},${Math.round(t*146+(1-t)*144)},${Math.round(t*60+(1-t)*255)},1)`;
}

// Decision boundary: class-0=blue, class-1=orange, blended by output probability
function boundaryColor(prob) {
  return `rgb(${Math.round(prob*251+(1-prob)*59)},${Math.round(prob*146+(1-prob)*130)},${Math.round(prob*60+(1-prob)*246)})`;
}

// Gradient magnitude → gray (≈0) to red (large)
function gradientColor(magnitude, maxMag) {
  if (maxMag === 0) return 'rgba(100,100,100,0.3)';
  const t = Math.min(1, magnitude / maxMag);
  return `rgba(${Math.round(t*239+(1-t)*75)},${Math.round(t*68+(1-t)*85)},${Math.round(t*68+(1-t)*99)},${0.3+t*0.7})`;
}

// =============================================================================
// SECTION 12 — NETWORK GRAPH LAYOUT
// =============================================================================
function computeLayout(layerSizes, svgWidth, svgHeight) {
  const pad = { x: 60, y: 40 };
  const W   = svgWidth  - pad.x * 2;
  const H   = svgHeight - pad.y * 2;
  const L   = layerSizes.length;
  return layerSizes.map((count, li) => {
    const x = pad.x + (li / (L - 1)) * W;
    return Array.from({ length: count }, (_, ni) => ({
      x,
      y: count === 1 ? svgHeight / 2 : pad.y + (ni / (count - 1)) * H,
      layerIdx: li,
      neuronIdx: ni,
    }));
  });
}

// =============================================================================
// SECTION 13 — XOR EVALUATION
// Runs inference on all 4 XOR samples using the current weights and returns
// a verification table. This is the ground-truth check: if every sample is
// classified correctly with high confidence, XOR is solved.
//
// "Confidence" here is distance from the 0.5 decision boundary, mapped to [0,1]:
//   conf = |output − 0.5| × 2
// So an output of 0.05 (class 0, far from boundary) → conf = 0.90 (90%).
//
// PyTorch equivalent (evaluation mode):
//   model.eval()
//   with torch.no_grad():
//       preds = model(X_xor)
//       correct = ((preds > 0.5).float() == y_xor).all()
// =============================================================================
function evaluateXOR(weights, biases, hiddenActivationTypes) {
  const L = weights.length;
  return XOR_DATA.map(({ input, label }) => {
    const { activations, preActivations } = forwardPass(
      input, weights, biases, hiddenActivationTypes
    );
    const rawOutput     = activations[L][0];           // p(class=1)
    const predictedClass = rawOutput > 0.5 ? 1 : 0;
    const confidence    = Math.abs(rawOutput - 0.5) * 2; // ∈ [0,1]
    const correct       = predictedClass === label;
    const eps           = 1e-12;
    const p             = Math.max(eps, Math.min(1 - eps, rawOutput));
    const sampleLoss    = -(label * Math.log(p) + (1 - label) * Math.log(1 - p));
    return { input, label, rawOutput, predictedClass, confidence, correct, sampleLoss,
             activations, preActivations };
  });
}

// =============================================================================
// SECTION 14 — CONVERGENCE / STOP CONDITIONS
//
// Why stop automatically?
//   Once the XOR problem is solved, continuing to train serves no purpose.
//   We want the tool to give a clear "done" signal rather than running forever.
//
// Two convergence criteria (either triggers a stop):
//   1. Loss < 0.001 — the BCE loss is extremely low; the network has fit XOR.
//   2. All 4 XOR samples correctly classified with confidence > 95% for
//      CONVERGENCE_CONSECUTIVE_EPOCHS consecutive epochs — robust to cases where
//      loss is low but not quite below the 0.001 threshold.
//
// Additional stopping conditions:
//   3. Epoch count >= maxEpochs — safety ceiling, prevents infinite training.
//   4. Plateau: loss has not improved by MIN_IMPROVEMENT or more in
//      PLATEAU_PATIENCE epochs AND loss is still above CONVERGENCE_LOSS_THRESHOLD.
//      This detects a stuck network without confusing it with a converged one.
//
// The plateau check explicitly requires loss > CONVERGENCE_LOSS_THRESHOLD so we
// never surface "the model is stuck" when it has actually solved XOR.
// =============================================================================
const CONVERGENCE_LOSS_THRESHOLD      = 0.001;
const CONVERGENCE_CONSECUTIVE_EPOCHS  = 50;
const CONVERGENCE_CONFIDENCE          = 0.95;
const PLATEAU_PATIENCE                = 100;  // epochs without meaningful improvement
const MIN_IMPROVEMENT                 = 0.0005; // minimum loss delta to count as progress
const PLATEAU_MIN_LOSS                = 0.05;   // only call it a plateau if loss is still this high

function checkConvergence(loss, xorResults, consecutiveCorrect) {
  if (loss < CONVERGENCE_LOSS_THRESHOLD) {
    return { converged: true, reason: `Loss dropped below ${CONVERGENCE_LOSS_THRESHOLD} (current: ${loss.toFixed(6)})` };
  }
  const allHighConf = xorResults.every(r => r.correct && r.confidence > CONVERGENCE_CONFIDENCE);
  if (allHighConf && consecutiveCorrect >= CONVERGENCE_CONSECUTIVE_EPOCHS) {
    return { converged: true, reason: `All 4 XOR points correctly classified with >${(CONVERGENCE_CONFIDENCE*100).toFixed(0)}% confidence for ${CONVERGENCE_CONSECUTIVE_EPOCHS} consecutive epochs` };
  }
  return { converged: false, reason: '' };
}

// =============================================================================
// COMPONENT: NetworkGraph
// =============================================================================
function NetworkGraph({ layerSizes, hiddenActivationTypes, forwardData, backpropData, animatingLayer }) {
  const SVG_W = 520, SVG_H = 320, R = 18;
  const layout = computeLayout(layerSizes, SVG_W, SVG_H);

  let maxGradMag = 0;
  if (backpropData) {
    for (const W of backpropData.dWeights)
      for (const row of W)
        for (const v of row)
          if (Math.abs(v) > maxGradMag) maxGradMag = Math.abs(v);
  }

  // viewBox lets the SVG scale its coordinate system to fill the CSS width.
  // Explicit style height prevents the proportional-height calculation from
  // making the graph taller than the coordinate space (which would leave a
  // large blank area below the neurons when the panel is wide).
  // w-full: fills available width; block: removes the inline baseline gap.
  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      className="w-full block"
      style={{ height: `${SVG_H}px` }}
    >
      {/* Edges */}
      {layout.slice(0, -1).map((fromLayer, li) =>
        fromLayer.map((from, fi) =>
          layout[li + 1].map((to, ti) => {
            let stroke = 'rgba(148,163,184,0.15)', sw = 1;
            if (backpropData?.dWeights[li]) {
              const m = Math.abs(backpropData.dWeights[li][ti][fi]);
              stroke = gradientColor(m, maxGradMag);
              sw = 1 + 2 * (m / (maxGradMag || 1));
            }
            const isActive = animatingLayer >= 0 && (li === animatingLayer - 1 || li === animatingLayer);
            if (isActive) { stroke = 'rgba(96,165,250,0.5)'; sw = 1.5; }
            return (
              <line key={`e-${li}-${fi}-${ti}`}
                x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                stroke={stroke} strokeWidth={sw} className="edge-line" />
            );
          })
        )
      )}

      {/* Neurons */}
      {layout.map((layer, li) =>
        layer.map((pos, ni) => {
          const actVal  = forwardData?.activations?.[li]?.[ni];
          const hasAct  = actVal !== undefined;
          const fill    = hasAct ? activationColor(actVal) : '#1e293b';
          const dimmed  = animatingLayer >= 0 && li > animatingLayer;
          const ringCol = li === 0 ? '#64748b'
            : li === layerSizes.length - 1 ? '#f59e0b'
            : ACTIVATIONS[hiddenActivationTypes[li - 1] || 'relu'].color;
          return (
            <g key={`n-${li}-${ni}`} opacity={dimmed ? 0.3 : 1}>
              <circle cx={pos.x} cy={pos.y} r={R+2} fill="none" stroke={ringCol} strokeWidth={1} opacity={0.5} />
              <circle cx={pos.x} cy={pos.y} r={R} fill={fill}
                stroke={hasAct ? '#e2e8f0' : '#475569'} strokeWidth={1.5} className="neuron-circle" />
              {hasAct ? (
                <text x={pos.x} y={pos.y+1} textAnchor="middle" dominantBaseline="middle"
                  fill="white" fontSize={9} fontWeight="bold" fontFamily="monospace">
                  {actVal.toFixed(2)}
                </text>
              ) : (
                <text x={pos.x} y={pos.y+1} textAnchor="middle" dominantBaseline="middle"
                  fill="#94a3b8" fontSize={9}>
                  {li === 0 ? (ni === 0 ? 'x₁' : 'x₂') : `n${ni+1}`}
                </text>
              )}
            </g>
          );
        })
      )}

      {/* Layer labels */}
      {layout.map((layer, li) => (
        <text key={`lbl-${li}`} x={layer[0].x} y={14} textAnchor="middle" fill="#94a3b8" fontSize={10}>
          {li === 0 ? 'Input(2)' : li === layerSizes.length-1 ? 'Output(1)' : `Hidden${li}(${layerSizes[li]})`}
        </text>
      ))}

      {/* Gradient legend */}
      {backpropData && (
        <g transform="translate(8,285)">
          <text fill="#94a3b8" fontSize={9}>∂L/∂W:</text>
          {[0,.25,.5,.75,1].map((t,i) => (
            <rect key={i} x={48+i*12} y={-8} width={12} height={10} fill={gradientColor(t,1)} />
          ))}
          <text x={48}  y={12} fill="#94a3b8" fontSize={8}>0</text>
          <text x={92}  y={12} fill="#94a3b8" fontSize={8}>max</text>
        </g>
      )}
    </svg>
  );
}

// =============================================================================
// COMPONENT: DecisionBoundaryCanvas
// Every pixel in this canvas is colored by an ACTUAL forward pass through the
// trained model. Nothing is interpolated or mocked. When training changes the
// weights, React re-runs the effect and repaints the entire grid from scratch.
// =============================================================================
function DecisionBoundaryCanvas({ weights, biases, hiddenActivationTypes, inferencePoint, onClick }) {
  const canvasRef  = useRef(null);
  const GRID       = 40;
  const CANVAS_SZ  = 260;
  const POINT_R    = 7;

  useEffect(() => {
    if (!weights || !canvasRef.current) return;
    const ctx      = canvasRef.current.getContext('2d');
    const cellSize = CANVAS_SZ / GRID;

    // Paint the decision boundary — each cell is one forward pass
    const grid = computeDecisionBoundary(weights, biases, hiddenActivationTypes, GRID);
    ctx.clearRect(0, 0, CANVAS_SZ, CANVAS_SZ);
    for (let row = 0; row < GRID; row++) {
      for (let col = 0; col < GRID; col++) {
        ctx.fillStyle  = boundaryColor(grid[row][col]);
        ctx.globalAlpha = 0.6;
        ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
      }
    }

    // XOR training points — cy = (1 − x₂)·H so that x₂=0 is at bottom
    ctx.globalAlpha = 1;
    for (const { input, label } of XOR_DATA) {
      const cx = input[0] * CANVAS_SZ;
      const cy = (1 - input[1]) * CANVAS_SZ;
      ctx.beginPath(); ctx.arc(cx, cy, POINT_R+2, 0, Math.PI*2);
      ctx.strokeStyle = 'white'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, cy, POINT_R, 0, Math.PI*2);
      ctx.fillStyle = label === 1 ? '#f97316' : '#3b82f6'; ctx.fill();
      ctx.fillStyle = 'white'; ctx.font = 'bold 9px monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(label.toString(), cx, cy);
    }

    // Inference point (if user clicked)
    if (inferencePoint) {
      const cx = inferencePoint.x * CANVAS_SZ;
      const cy = (1 - inferencePoint.y) * CANVAS_SZ;
      ctx.beginPath(); ctx.arc(cx, cy, POINT_R+4, 0, Math.PI*2);
      ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 2;
      ctx.setLineDash([3,3]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = inferencePoint.prediction > 0.5 ? '#f97316' : '#3b82f6';
      ctx.globalAlpha = 0.85;
      ctx.beginPath(); ctx.arc(cx, cy, POINT_R, 0, Math.PI*2); ctx.fill();
      ctx.globalAlpha = 1;
    }

    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('x₁: 0 → 1', CANVAS_SZ/2, CANVAS_SZ-3);
  }, [weights, biases, hiddenActivationTypes, inferencePoint]);

  return (
    <div className="relative">
      <canvas ref={canvasRef} width={CANVAS_SZ} height={CANVAS_SZ}
        className="rounded border border-slate-700 block"
        onClick={onClick}
        style={{ cursor: onClick ? 'crosshair' : 'default' }} />
      <div className="absolute top-1 left-1 text-xs text-slate-500 font-mono pointer-events-none">x₂↑</div>
    </div>
  );
}

// =============================================================================
// COMPONENT: XorVerifyPanel
// Shows all 4 XOR inputs evaluated against the current model weights.
// Values are computed by evaluateXOR() which calls forwardPass() for each
// sample — the same function used during training. Nothing is cached from
// a prior forward pass; this is always a fresh inference call.
//
// Confidence = |p − 0.5| × 2, so:
//   p=0.97 → conf=0.94   (high confidence class 1)
//   p=0.50 → conf=0.00   (right on the boundary, uncertain)
//   p=0.05 → conf=0.90   (high confidence class 0)
//
// In PyTorch this corresponds to model.eval() + torch.no_grad() inference.
// =============================================================================
function XorVerifyPanel({ xorResults }) {
  if (!xorResults) return (
    <div className="text-xs text-slate-600 italic px-1">Train to see XOR verification</div>
  );
  const allCorrect = xorResults.every(r => r.correct);
  const allHighConf = xorResults.every(r => r.correct && r.confidence > CONVERGENCE_CONFIDENCE);
  return (
    <div>
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">XOR Verify</span>
        {allCorrect && (
          <span className={`text-xs px-1.5 py-0.5 rounded font-bold ${allHighConf ? 'bg-emerald-900/60 text-emerald-400' : 'bg-blue-900/60 text-blue-400'}`}>
            {allHighConf ? '✓ Solved' : '✓ Correct'}
          </span>
        )}
        {!allCorrect && (
          <span className="text-xs px-1.5 py-0.5 rounded bg-red-900/40 text-red-400">
            {xorResults.filter(r => r.correct).length}/4 correct
          </span>
        )}
      </div>
      <table className="w-full text-xs font-mono border-collapse">
        <thead>
          <tr className="text-slate-600">
            <th className="text-left pb-1">Input</th>
            <th className="text-center pb-1">Exp</th>
            <th className="text-right pb-1">p(1)</th>
            <th className="text-center pb-1">→</th>
            <th className="text-right pb-1">Conf</th>
            <th className="text-center pb-1 w-4">✓</th>
          </tr>
        </thead>
        <tbody>
          {xorResults.map((r, i) => (
            <tr key={i} className={`border-t border-slate-800 ${r.correct ? 'text-slate-300' : 'text-red-400'}`}>
              <td className="py-0.5 pr-1">[{r.input.join(',')}]</td>
              <td className="text-center">{r.label}</td>
              <td className="text-right">{r.rawOutput.toFixed(3)}</td>
              <td className="text-center text-slate-500">→</td>
              <td className={`text-right ${r.confidence > CONVERGENCE_CONFIDENCE ? 'text-emerald-400' : r.confidence > 0.7 ? 'text-amber-400' : 'text-red-400'}`}>
                {(r.confidence * 100).toFixed(0)}%
              </td>
              <td className="text-center">{r.correct ? '✓' : '✗'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// =============================================================================
// COMPONENT: MathAuditPanel
// Shows the full forward-pass computation for a selected XOR input:
// pre-activations (z) and activations (a) at every layer, plus the final
// output and per-sample BCE loss.
//
// These numbers MUST match what is displayed on the neurons during the
// Forward Pass animation — they are produced by the same forwardPass() call.
// If a neuron shows 0.73 in the visualization, the z and a values for that
// neuron will appear here and should be consistent.
//
// This maps to PyTorch's intermediate tensors, which you can inspect with
// hooks: model.register_forward_hook(lambda m, i, o: print(o))
// =============================================================================
function MathAuditPanel({ xorResults, hiddenActivationTypes, layerSizes }) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  if (!xorResults) return (
    <div className="text-xs text-slate-600 italic">Train to see forward-pass audit</div>
  );

  const r = xorResults[selectedIdx];
  const { activations, preActivations, input, label, rawOutput, sampleLoss } = r;

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Math Audit</span>
        <div className="flex gap-1">
          {XOR_DATA.map((d, i) => (
            <button key={i} onClick={() => setSelectedIdx(i)}
              className={`text-xs px-1.5 py-0.5 rounded font-mono transition-colors ${
                i === selectedIdx ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}>
              [{d.input.join(',')}]
            </button>
          ))}
        </div>
      </div>

      <div className="text-xs font-mono space-y-1.5 bg-slate-900/60 rounded p-2">
        {/* Input */}
        <div className="text-slate-400">
          <span className="text-slate-500">Input  </span>
          x = [{input.join(', ')}]
          <span className="text-slate-600 ml-1">(label={label})</span>
        </div>

        {/* Each layer's z and a */}
        {activations.slice(1).map((a, li) => {
          const z          = preActivations[li + 1];
          const isOutput   = li === activations.length - 2;
          const activType  = isOutput ? 'sigmoid' : (hiddenActivationTypes[li] || 'relu');
          return (
            <div key={li} className="border-t border-slate-800 pt-1.5">
              <div className="text-slate-500 text-xs mb-0.5">
                Layer {li + 1} ({isOutput ? 'output, sigmoid' : activType})
              </div>
              {/* Show z = W·x + b */}
              <div className="text-indigo-300">
                z = [{z.map(v => v.toFixed(4)).join(', ')}]
              </div>
              {/* Show a = activation(z) */}
              <div className="text-blue-300">
                a = [{a.map(v => v.toFixed(4)).join(', ')}]
                {isOutput && <span className="text-slate-500 ml-1">← p(class=1)</span>}
              </div>
            </div>
          );
        })}

        {/* Per-sample loss */}
        <div className="border-t border-slate-800 pt-1.5 text-amber-300">
          loss = −({label}·log({rawOutput.toFixed(4)}) + {1-label}·log({(1-rawOutput).toFixed(4)}))
          <span className="block text-amber-400 font-bold">     = {sampleLoss.toFixed(6)}</span>
        </div>
        <div className="text-slate-600 text-xs">
          ↑ These z/a values match the network graph neurons above.
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// COMPONENT: TrainingStatusBar
// Shows the current training state prominently so the user always knows
// whether training is running, has converged, plateaued, or hit max epochs.
// =============================================================================
function TrainingStatusBar({ status, epoch, loss, bestLoss, epochsSinceImprove, stopReason, maxEpochs }) {
  const statusConfig = {
    idle:      { label: 'Not Started',        color: 'text-slate-500',  bg: 'bg-slate-800/40',  dot: 'bg-slate-600' },
    training:  { label: 'Training',           color: 'text-blue-400',   bg: 'bg-blue-900/20',   dot: 'bg-blue-400 animate-pulse' },
    paused:    { label: 'Paused',             color: 'text-amber-400',  bg: 'bg-amber-900/20',  dot: 'bg-amber-400' },
    converged: { label: 'Converged ✓',        color: 'text-emerald-400',bg: 'bg-emerald-900/20',dot: 'bg-emerald-400' },
    plateaued: { label: 'Plateaued / Stuck',  color: 'text-orange-400', bg: 'bg-orange-900/20', dot: 'bg-orange-400' },
    maxEpochs: { label: 'Max Epochs Reached', color: 'text-slate-400',  bg: 'bg-slate-800/60',  dot: 'bg-slate-500' },
  };
  const cfg = statusConfig[status] || statusConfig.idle;
  return (
    <div className={`${cfg.bg} border-b border-slate-700 px-5 py-1.5 flex items-center gap-6 text-xs`}>
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${cfg.dot}`} />
        <span className={`font-bold ${cfg.color}`}>{cfg.label}</span>
      </div>
      <div className="flex items-center gap-5 text-slate-400 font-mono">
        <span>Epoch: <span className="text-slate-200">{epoch}</span></span>
        {loss !== null && (
          <span>Loss: <span className={`font-bold ${loss < 0.01 ? 'text-emerald-400' : loss < 0.1 ? 'text-amber-400' : 'text-red-400'}`}>{loss.toFixed(5)}</span></span>
        )}
        {bestLoss !== Infinity && (
          <span>Best: <span className="text-emerald-300">{bestLoss.toFixed(5)}</span></span>
        )}
        {epoch > 0 && (
          <span>No improv: <span className={epochsSinceImprove > 50 ? 'text-orange-400' : 'text-slate-400'}>{epochsSinceImprove}</span></span>
        )}
        {maxEpochs && (
          <span className="text-slate-600">(max {maxEpochs})</span>
        )}
      </div>
      {stopReason && (
        <div className="ml-auto text-slate-500 truncate max-w-xs">{stopReason}</div>
      )}
    </div>
  );
}

// =============================================================================
// COMPONENT: ConceptCallout
// =============================================================================
function ConceptCallout({ type, onDismiss, trainingStatus }) {
  // trainingStatus is used to adjust callouts that describe an ongoing
  // action — once training stops, their wording should reflect the past tense.
  const stopped = trainingStatus !== 'training';

  const callouts = {
    firstForward: {
      title: 'First Forward Pass', color: 'border-blue-500', icon: '→',
      body: 'Each neuron computes z = W·x + b, then a = activation(z). Values flow left → right. The output is a probability in (0,1) via sigmoid.',
      pytorch: 'output = model(input_tensor)  # calls model.forward()',
    },
    firstBackprop: {
      title: stopped ? 'Backprop Complete' : 'Backpropagation Running',
      color: 'border-violet-500', icon: '←',
      body: stopped
        ? 'Gradients were computed via the chain rule and weights were updated. The edges are now colored by the last ∂L/∂W values from the final epoch.'
        : 'Gradients flow right → left. Each weight learns how much it contributed to the error. The chain rule multiplies local derivatives at each layer.',
      pytorch: 'loss.backward()  # PyTorch autograd computes all ∂L/∂W',
    },
    lossPlateauing: {
      title: 'Loss Plateaued — Network is Stuck', color: 'border-orange-500', icon: '⚠',
      body: `Loss has not meaningfully improved in ${PLATEAU_PATIENCE} epochs and is still above ${PLATEAU_MIN_LOSS}. Training stopped. Try: increasing the learning rate, adding neurons/layers, or switching activation functions.`,
      pytorch: '# Try: optimizer = torch.optim.Adam(model.parameters(), lr=0.01)',
    },
    converged: {
      title: 'XOR Solved ✓', color: 'border-emerald-500', icon: '✓',
      body: 'The network correctly classifies all 4 XOR points with high confidence. Training stopped automatically. You can click anywhere on the boundary canvas to test inference.',
      pytorch: `model.eval()\nwith torch.no_grad():\n    pred = model(test_input)`,
    },
    vanishingGradient: {
      title: 'Vanishing Gradient Detected', color: 'border-red-500', icon: '⚠',
      body: 'Gradients near the input layer are near zero. Sigmoid derivatives max at 0.25 — deep sigmoid chains drive gradients toward 0. Try ReLU for hidden layers.',
      pytorch: '# Replace nn.Sigmoid() with nn.ReLU() in hidden layers',
    },
    inferencePoint: {
      title: 'Inference Mode', color: 'border-emerald-500', icon: '◎',
      body: 'The network ran a forward pass on your test point using the current trained weights. No gradients were computed.',
      pytorch: `model.eval()\nwith torch.no_grad():  # skip gradient tracking\n    pred = model(test_point)`,
    },
  };

  const c = callouts[type];
  if (!c) return null;
  return (
    <div className={`border-l-4 ${c.color} bg-slate-800/80 rounded-r p-3 text-sm`}>
      <div className="flex justify-between items-start">
        <span className="font-bold text-white flex items-center gap-2">
          <span>{c.icon}</span> {c.title}
        </span>
        <button onClick={onDismiss} className="text-slate-500 hover:text-white ml-2 text-xs">✕</button>
      </div>
      <p className="text-slate-300 mt-1 text-xs leading-relaxed">{c.body}</p>
      {c.pytorch && (
        <pre className="mt-2 text-xs bg-black/40 rounded p-2 text-emerald-400 overflow-x-auto">{c.pytorch}</pre>
      )}
    </div>
  );
}

// =============================================================================
// COMPONENT: PyTorchSidebar
// =============================================================================
// PyTorchSidebar must be used inside a flex-col container that gives it a
// bounded height — otherwise overflow-y-auto has nothing to constrain against
// and the <pre> expands to full content height, overflowing into sibling panels.
function PyTorchSidebar({ layerSizes, hiddenActivationTypes }) {
  return (
    <div className="bg-gray-950 rounded-lg border border-slate-700 p-3 h-full overflow-y-auto">
      <div className="flex items-center gap-1.5 mb-2">
        {['bg-red-500','bg-yellow-500','bg-green-500'].map((c,i) => (
          <div key={i} className={`w-2.5 h-2.5 rounded-full ${c}`} />
        ))}
        <span className="text-slate-400 text-xs ml-1 font-mono">model.py</span>
      </div>
      <pre className="text-xs font-mono text-slate-300 leading-relaxed whitespace-pre-wrap">
        {generatePyTorchCode(layerSizes, hiddenActivationTypes)}
      </pre>
    </div>
  );
}

// =============================================================================
// MAIN APP COMPONENT
// =============================================================================
export default function App() {
  // ── Architecture ──────────────────────────────────────────────────────────
  const [numHiddenLayers,  setNumHiddenLayers]  = useState(2);
  const [neuronsPerLayer,  setNeuronsPerLayer]   = useState([4, 4]);
  const [activationTypes,  setActivationTypes]   = useState(['relu', 'relu']);

  // ── Network weights ────────────────────────────────────────────────────────
  const [network, setNetwork] = useState(null);

  // ── Training state (React display) ────────────────────────────────────────
  const [isTraining,            setIsTraining]            = useState(false);
  const [trainingStatus,        setTrainingStatus]        = useState('idle');
  // 'idle' | 'training' | 'paused' | 'converged' | 'plateaued' | 'maxEpochs'
  const [stopReason,            setStopReason]            = useState('');
  const [epoch,                 setEpoch]                 = useState(0);
  const [lossHistory,           setLossHistory]           = useState([]);
  const [learningRate,          setLearningRate]          = useState(0.1);
  const [bestLoss,              setBestLoss]              = useState(Infinity);
  const [epochsSinceImprovement,setEpochsSinceImprovement] = useState(0);
  const [maxEpochs,             setMaxEpochs]             = useState(10000);
  const [lastGradients,         setLastGradients]         = useState(null);
  const [lastForwardData,       setLastForwardData]       = useState(null);
  const [xorResults,            setXorResults]            = useState(null);

  // ── Animation ─────────────────────────────────────────────────────────────
  const [animatingLayer,    setAnimatingLayer]    = useState(-1);
  const [forwardPassDisplay,setForwardPassDisplay] = useState(null);

  // ── Inference ─────────────────────────────────────────────────────────────
  const [inferencePoint, setInferencePoint] = useState(null);

  // ── Concept callouts ───────────────────────────────────────────────────────
  const [callouts,         setCallouts]         = useState(new Set());
  const [dismissedCallouts,setDismissedCallouts] = useState(new Set());

  // ── Refs (used inside RAF loop to avoid stale closures) ────────────────────
  const trainingRef              = useRef(false);
  const networkRef               = useRef(null);
  const epochRef                 = useRef(0);
  const lossHistoryRef           = useRef([]);
  const bestLossRef              = useRef(Infinity);
  const epochsSinceImprovRef     = useRef(0);
  const consecutiveCorrectRef    = useRef(0);
  const maxEpochsRef             = useRef(10000);
  const dismissedCalloutsRef     = useRef(new Set());
  const trainingLoopRef          = useRef(null);

  // Keep dismissedCalloutsRef in sync with React state
  useEffect(() => { dismissedCalloutsRef.current = dismissedCallouts; }, [dismissedCallouts]);
  useEffect(() => { maxEpochsRef.current = maxEpochs; }, [maxEpochs]);

  // Derived layer sizes: [2, …hidden neurons…, 1]
  const layerSizes = [2, ...neuronsPerLayer.slice(0, numHiddenLayers), 1];

  // ── Network initialization ─────────────────────────────────────────────────
  const initializeNetwork = useCallback(() => {
    if (trainingLoopRef.current) cancelAnimationFrame(trainingLoopRef.current);
    trainingRef.current          = false;
    const net                    = initNetwork(layerSizes);
    networkRef.current           = net;
    epochRef.current             = 0;
    lossHistoryRef.current       = [];
    bestLossRef.current          = Infinity;
    epochsSinceImprovRef.current = 0;
    consecutiveCorrectRef.current = 0;

    setNetwork(net);
    setIsTraining(false);
    setTrainingStatus('idle');
    setStopReason('');
    setEpoch(0);
    setLossHistory([]);
    setBestLoss(Infinity);
    setEpochsSinceImprovement(0);
    setLastGradients(null);
    setLastForwardData(null);
    setForwardPassDisplay(null);
    setInferencePoint(null);
    setXorResults(null);
    setCallouts(new Set());
  }, [layerSizes.join(',')]);

  useEffect(() => { initializeNetwork(); }, []); // mount

  // Re-init when architecture changes
  const prevArchRef = useRef(null);
  useEffect(() => {
    const key = layerSizes.join(',') + '|' + activationTypes.join(',');
    if (prevArchRef.current !== null && prevArchRef.current !== key) initializeNetwork();
    prevArchRef.current = key;
  }, [layerSizes.join(','), activationTypes.join(',')]);

  // ── Architecture handlers ──────────────────────────────────────────────────
  const handleNumHiddenLayersChange = (n) => {
    const c = Math.max(1, Math.min(4, n));
    setNumHiddenLayers(c);
    setNeuronsPerLayer(prev => { const a = [...prev]; while (a.length < c) a.push(4); return a.slice(0,c); });
    setActivationTypes(prev => { const a = [...prev]; while (a.length < c) a.push('relu'); return a.slice(0,c); });
  };
  const handleNeuronsChange    = (i, n) => setNeuronsPerLayer(prev => { const a=[...prev]; a[i]=Math.max(2,Math.min(8,n)); return a; });
  const handleActivationChange = (i, t) => setActivationTypes(prev => { const a=[...prev]; a[i]=t; return a; });

  // ── Core training step (called from RAF loop) ─────────────────────────────
  // Returns { shouldStop: bool } so the loop knows when to halt.
  const runTrainingStep = useCallback(() => {
    if (!networkRef.current) return { shouldStop: false };

    // ① Check max-epoch ceiling BEFORE running the epoch
    if (epochRef.current >= maxEpochsRef.current) {
      const reason = `Reached maximum of ${maxEpochsRef.current} epochs`;
      setTrainingStatus('maxEpochs');
      setStopReason(reason);
      return { shouldStop: true };
    }

    const { weights, biases } = networkRef.current;
    const result = trainOneEpoch(weights, biases, activationTypes, learningRate);

    // Update network weights
    networkRef.current = { weights: result.weights, biases: result.biases };
    setNetwork({ weights: result.weights, biases: result.biases });

    // Increment epoch counter
    epochRef.current += 1;
    setEpoch(epochRef.current);

    // Append to loss history (keep last 200 for performance)
    lossHistoryRef.current = [...lossHistoryRef.current,
      { epoch: epochRef.current, loss: result.loss }].slice(-200);
    setLossHistory([...lossHistoryRef.current]);

    setLastGradients({ dWeights: result.avgDW, dBiases: result.avgDB });
    setLastForwardData(result.allForwardData[0]); // first XOR sample for network graph

    // ② Track best loss and epochs-since-improvement
    // "Improvement" requires at least MIN_IMPROVEMENT decrease to filter noise.
    if (result.loss < bestLossRef.current - MIN_IMPROVEMENT) {
      bestLossRef.current          = result.loss;
      epochsSinceImprovRef.current = 0;
      setBestLoss(result.loss);
    } else {
      epochsSinceImprovRef.current += 1;
    }
    setEpochsSinceImprovement(epochsSinceImprovRef.current);

    // ③ Evaluate all 4 XOR samples with current weights (real inference calls)
    const xorEval = evaluateXOR(result.weights, result.biases, activationTypes);
    setXorResults(xorEval);

    // ④ Track consecutive epochs where all 4 points are correct + high-confidence
    const allHighConf = xorEval.every(r => r.correct && r.confidence > CONVERGENCE_CONFIDENCE);
    if (allHighConf) consecutiveCorrectRef.current += 1;
    else             consecutiveCorrectRef.current  = 0;

    // ⑤ Check convergence (either criterion)
    const { converged, reason } = checkConvergence(
      result.loss, xorEval, consecutiveCorrectRef.current
    );
    if (converged) {
      setTrainingStatus('converged');
      setStopReason(reason);
      if (!dismissedCalloutsRef.current.has('converged')) {
        setCallouts(prev => new Set([...prev, 'converged']));
      }
      return { shouldStop: true };
    }

    // ⑥ Plateau detection — only fires when loss is STILL high (not near convergence).
    // This prevents showing "stuck" when the model has actually solved XOR at a loss
    // slightly above the hard 0.001 threshold but with correct classifications.
    if (
      result.loss > PLATEAU_MIN_LOSS &&
      epochsSinceImprovRef.current >= PLATEAU_PATIENCE &&
      !dismissedCalloutsRef.current.has('lossPlateauing')
    ) {
      setTrainingStatus('plateaued');
      setStopReason(`Loss has not improved by >${MIN_IMPROVEMENT} in ${PLATEAU_PATIENCE} epochs`);
      setCallouts(prev => new Set([...prev, 'lossPlateauing']));
      return { shouldStop: true };
    }

    // ⑦ One-time informational callouts
    if (epochRef.current === 1 && !dismissedCalloutsRef.current.has('firstBackprop')) {
      setCallouts(prev => new Set([...prev, 'firstBackprop']));
    }

    // Vanishing gradient: first-layer max gradient is <1% of network-wide max gradient
    const firstMax = Math.max(...result.avgDW[0].flat().map(Math.abs));
    const anyMax   = Math.max(...result.avgDW.flat(2).map(Math.abs));
    if (anyMax > 0 && firstMax / anyMax < 0.01 && !dismissedCalloutsRef.current.has('vanishingGradient')) {
      setCallouts(prev => new Set([...prev, 'vanishingGradient']));
    }

    return { shouldStop: false };
  }, [activationTypes, learningRate]); // note: dismissedCallouts accessed via ref

  // ── RAF training loop ──────────────────────────────────────────────────────
  // Runs STEPS_PER_FRAME epochs per animation frame for speed, but checks for
  // stop conditions after every individual epoch.
  const STEPS_PER_FRAME = 5;

  useEffect(() => {
    if (!isTraining) {
      if (trainingLoopRef.current) cancelAnimationFrame(trainingLoopRef.current);
      return;
    }
    trainingRef.current = true;
    const loop = () => {
      if (!trainingRef.current) return;
      for (let i = 0; i < STEPS_PER_FRAME; i++) {
        const { shouldStop } = runTrainingStep();
        if (shouldStop) {
          trainingRef.current = false;
          setIsTraining(false);
          return; // do not schedule next frame
        }
      }
      trainingLoopRef.current = requestAnimationFrame(loop);
    };
    trainingLoopRef.current = requestAnimationFrame(loop);
    return () => { if (trainingLoopRef.current) cancelAnimationFrame(trainingLoopRef.current); };
  }, [isTraining, runTrainingStep]);

  // ── Control handlers ────────────────────────────────────────────────────────
  const handleToggleTraining = () => {
    if (isTraining) {
      trainingRef.current = false;
      setIsTraining(false);
      setTrainingStatus('paused');
    } else {
      setTrainingStatus('training');
      setStopReason('');
      setIsTraining(true);
    }
  };

  const handleStepEpoch = () => {
    if (isTraining) return;
    const { shouldStop } = runTrainingStep();
    if (shouldStop) { setIsTraining(false); }
    else { setTrainingStatus('paused'); }
  };

  const handleReset = () => { initializeNetwork(); };

  // ── Forward-pass animation ─────────────────────────────────────────────────
  const runForwardPassAnimation = useCallback(async () => {
    if (!networkRef.current || isTraining) return;
    const { weights, biases } = networkRef.current;
    const input = XOR_DATA[0].input;
    const { activations, preActivations } = forwardPass(input, weights, biases, activationTypes);
    if (!dismissedCalloutsRef.current.has('firstForward')) {
      setCallouts(prev => new Set([...prev, 'firstForward']));
    }
    for (let l = 0; l <= layerSizes.length - 1; l++) {
      setAnimatingLayer(l);
      setForwardPassDisplay({ activations: activations.slice(0, l + 1), preActivations });
      await new Promise(r => setTimeout(r, 450));
    }
    setAnimatingLayer(-1);
    setForwardPassDisplay({ activations, preActivations });
    setLastForwardData({ activations, preActivations });
  }, [activationTypes, layerSizes.length, isTraining]);

  // ── Click-to-infer on the boundary canvas ─────────────────────────────────
  const handleCanvasClick = (e) => {
    if (!networkRef.current) return;
    const canvas = e.currentTarget;
    const rect   = canvas.getBoundingClientRect();
    const x1     = (e.clientX - rect.left) / rect.width;
    const cy     = (e.clientY - rect.top)  / rect.height;
    const x2     = 1 - cy; // flip: canvas top = x₂=1
    const { weights, biases } = networkRef.current;
    const { activations }     = forwardPass([x1, x2], weights, biases, activationTypes);
    const prediction          = activations[activations.length - 1][0];
    setInferencePoint({ x: x1, y: x2, prediction });
    if (!dismissedCalloutsRef.current.has('inferencePoint')) {
      setCallouts(prev => new Set([...prev, 'inferencePoint']));
    }
  };

  const dismissCallout = (type) => {
    setCallouts(prev       => { const s = new Set(prev);        s.delete(type); return s; });
    setDismissedCallouts(prev => new Set([...prev, type]));
  };

  // Derived display values
  const latestLoss    = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].loss : null;
  const activeCallouts = [...callouts].filter(c => !dismissedCallouts.has(c)).slice(0, 2);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans flex flex-col">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-700 px-5 py-2.5 flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-base font-bold text-white leading-tight">Neural Network Learning Tool</h1>
          <p className="text-xs text-slate-500">Phase 1 Hardened — real math, XOR, self-validating</p>
        </div>
        <div className="text-xs text-slate-500 font-mono">
          {layerSizes.join(' → ')}
        </div>
      </header>

      {/* ── Training Status Bar ────────────────────────────────────────────── */}
      <TrainingStatusBar
        status={trainingStatus}
        epoch={epoch}
        loss={latestLoss}
        bestLoss={bestLoss}
        epochsSinceImprove={epochsSinceImprovement}
        stopReason={stopReason}
        maxEpochs={maxEpochs}
      />

      {/* ── Three-column body ─────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">
        {/* ═══════════════════════════════════════════════════════════════════
            LEFT PANEL — Architecture + Training Controls
        ═══════════════════════════════════════════════════════════════════ */}
        <div className="w-52 border-r border-slate-700 p-3 flex flex-col gap-3 overflow-y-auto flex-shrink-0">
          {/* Architecture */}
          <section>
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Architecture</h2>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-slate-400 flex-1">Hidden Layers</span>
              <button onClick={() => handleNumHiddenLayersChange(numHiddenLayers-1)}
                className="w-5 h-5 rounded bg-slate-700 hover:bg-slate-600 text-sm font-bold">−</button>
              <span className="font-mono text-white w-4 text-center text-sm">{numHiddenLayers}</span>
              <button onClick={() => handleNumHiddenLayersChange(numHiddenLayers+1)}
                className="w-5 h-5 rounded bg-slate-700 hover:bg-slate-600 text-sm font-bold">+</button>
            </div>
            {Array.from({ length: numHiddenLayers }, (_, i) => (
              <div key={i} className="mb-2 pl-2 border-l-2 border-slate-700">
                <div className="text-xs text-slate-500 mb-1">Layer {i+1}</div>
                <div className="flex items-center gap-1 mb-1">
                  <span className="text-xs text-slate-600 flex-1">Neurons</span>
                  <button onClick={() => handleNeuronsChange(i, neuronsPerLayer[i]-1)}
                    className="w-5 h-5 rounded bg-slate-700 hover:bg-slate-600 text-xs">−</button>
                  <span className="font-mono text-white text-xs w-4 text-center">{neuronsPerLayer[i]}</span>
                  <button onClick={() => handleNeuronsChange(i, neuronsPerLayer[i]+1)}
                    className="w-5 h-5 rounded bg-slate-700 hover:bg-slate-600 text-xs">+</button>
                </div>
                <select value={activationTypes[i]} onChange={e => handleActivationChange(i, e.target.value)}
                  className="w-full bg-slate-800 border border-slate-600 rounded text-xs text-white p-0.5">
                  <option value="relu">ReLU</option>
                  <option value="tanh">Tanh</option>
                  <option value="sigmoid">Sigmoid</option>
                </select>
              </div>
            ))}
          </section>

          {/* Training controls */}
          <section>
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Training</h2>
            <label className="block mb-2">
              <span className="text-xs text-slate-400">LR: {learningRate.toFixed(3)}</span>
              <input type="range" min="0.001" max="1" step="0.001" value={learningRate}
                onChange={e => setLearningRate(parseFloat(e.target.value))}
                className="w-full mt-0.5 accent-blue-500" />
            </label>

            {/* Max epochs */}
            <label className="block mb-2">
              <span className="text-xs text-slate-400">Max Epochs</span>
              <div className="flex gap-1 mt-0.5">
                {[1000, 5000, 10000, 50000].map(v => (
                  <button key={v} onClick={() => setMaxEpochs(v)}
                    className={`text-xs px-1.5 py-0.5 rounded flex-1 transition-colors ${
                      maxEpochs === v ? 'bg-blue-700 text-white' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                    }`}>
                    {v >= 1000 ? `${v/1000}k` : v}
                  </button>
                ))}
              </div>
            </label>

            <div className="flex flex-col gap-1.5">
              <button onClick={handleToggleTraining}
                className={`w-full py-1.5 rounded font-bold text-sm transition-colors ${
                  isTraining ? 'bg-amber-600 hover:bg-amber-500' : 'bg-blue-600 hover:bg-blue-500'
                } text-white`}>
                {isTraining ? '⏸ Pause' : trainingStatus === 'converged' ? '▶ Continue' : '▶ Train'}
              </button>
              <button onClick={handleStepEpoch} disabled={isTraining}
                className="w-full py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed">
                Step 1 Epoch
              </button>
              <button onClick={runForwardPassAnimation} disabled={isTraining}
                className="w-full py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-40">
                Forward Pass ▶
              </button>
              <button onClick={handleReset}
                className="w-full py-1 rounded text-xs bg-slate-800 hover:bg-red-900/50 text-slate-400 hover:text-red-300 border border-slate-700">
                Reset
              </button>
            </div>
          </section>

          {/* Inference result */}
          <section>
            <div className="text-xs text-slate-500 bg-slate-800/50 rounded p-2">
              <span className="text-slate-400 font-medium block mb-1">Click to Infer</span>
              Click the decision boundary canvas to run inference at that point.
            </div>
            {inferencePoint && (
              <div className="mt-1.5 text-xs font-mono bg-slate-800 rounded p-2 space-y-0.5">
                <div className="text-slate-400">x₁={inferencePoint.x.toFixed(3)} x₂={inferencePoint.y.toFixed(3)}</div>
                <div className={inferencePoint.prediction > 0.5 ? 'text-orange-400' : 'text-blue-400'}>
                  Class: {inferencePoint.prediction > 0.5 ? 1 : 0}
                </div>
                <div className="text-slate-300">p(1) = {inferencePoint.prediction.toFixed(4)}</div>
                <div className="text-slate-400">
                  conf = {(Math.abs(inferencePoint.prediction - 0.5)*2*100).toFixed(1)}%
                </div>
              </div>
            )}
          </section>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════
            CENTER PANEL — Network Graph + Boundary + Loss Curve
        ═══════════════════════════════════════════════════════════════════ */}
        <div className="flex-1 flex flex-col p-3 gap-3 min-w-0 overflow-hidden">
          {/* Callouts */}
          {activeCallouts.length > 0 && (
            <div className="flex flex-col gap-2 flex-shrink-0">
              {activeCallouts.map(type => (
                <ConceptCallout key={type} type={type} onDismiss={() => dismissCallout(type)} trainingStatus={trainingStatus} />
              ))}
            </div>
          )}

          {/* Network Graph — flex-shrink-0 so it only takes exactly the
              height the SVG needs (SVG_H + header + padding ≈ 360px).
              Previously flex-1 caused it to fill all remaining vertical
              space, leaving hundreds of pixels of empty gray below the
              neurons. */}
          <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-2 flex-shrink-0">
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-xs font-semibold text-slate-300">Network</h2>
              <div className="flex items-center gap-3 text-xs">
                {animatingLayer >= 0 && (
                  <span className="text-blue-400 font-mono animate-pulse">→ layer {animatingLayer}</span>
                )}
                {lastGradients && animatingLayer < 0 && (
                  <span className="text-violet-400 font-mono">edges = ∂L/∂W</span>
                )}
              </div>
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
          <div className="flex gap-3 flex-shrink-0" style={{ height: '280px' }}>
            {/* Decision Boundary */}
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-2 flex-shrink-0">
              <h2 className="text-xs font-semibold text-slate-300 mb-1">Decision Boundary</h2>
              {network ? (
                <DecisionBoundaryCanvas
                  weights={network.weights} biases={network.biases}
                  hiddenActivationTypes={activationTypes}
                  inferencePoint={inferencePoint} onClick={handleCanvasClick} />
              ) : (
                <div className="w-[260px] h-[260px] bg-slate-800 rounded flex items-center justify-center text-slate-600 text-xs">
                  Initializing…
                </div>
              )}
              <div className="flex gap-3 mt-1 text-xs text-slate-500">
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block"/>Class 0</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-orange-500 inline-block"/>Class 1</span>
                <span className="text-slate-600">· click = infer</span>
              </div>
            </div>

            {/* Loss Curve */}
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-2 flex-1 flex flex-col min-w-0">
              <h2 className="text-xs font-semibold text-slate-300 mb-1">BCE Loss Curve</h2>
              {lossHistory.length > 1 ? (
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lossHistory} margin={{ top:4, right:8, left:-22, bottom:4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="epoch" tick={{ fill:'#64748b', fontSize:9 }} />
                      <YAxis domain={[0,'auto']} tick={{ fill:'#64748b', fontSize:9 }} />
                      <Tooltip contentStyle={{ backgroundColor:'#1e293b', border:'1px solid #334155', fontSize:10 }}
                        labelStyle={{ color:'#94a3b8' }} itemStyle={{ color:'#60a5fa' }}
                        formatter={v => v.toFixed(5)} />
                      <Line type="monotone" dataKey="loss" stroke="#60a5fa" strokeWidth={1.5}
                        dot={false} isAnimationActive={false} />
                      {/* Visual threshold line at convergence criterion */}
                      <Line type="monotone" dataKey={() => CONVERGENCE_LOSS_THRESHOLD}
                        stroke="#10b981" strokeWidth={1} strokeDasharray="4 4"
                        dot={false} isAnimationActive={false} legendType="none" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="flex-1 flex items-center justify-center text-slate-600 text-xs">
                  Start training to see loss curve
                </div>
              )}
              {lossHistory.length > 1 && (
                <div className="text-xs text-slate-500 font-mono mt-1">
                  Initial {lossHistory[0]?.loss.toFixed(4)} → Current {latestLoss?.toFixed(4)}
                  <span className="text-slate-600 ml-2">(green line = {CONVERGENCE_LOSS_THRESHOLD} target)</span>
                  {trainingStatus === 'converged' && <span className="text-emerald-400 ml-2">✓ XOR solved</span>}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════
            RIGHT PANEL — PyTorch + XOR Verify + Math Audit
        ═══════════════════════════════════════════════════════════════════ */}
        {/*
          Right panel: flex-col with overflow-hidden on the outer div.
          Each section manages its own height so nothing bleeds into siblings.
          - PyTorch: fixed 240px, flex-col so PyTorchSidebar fills remaining
            height and scrolls internally (h-full overflow-y-auto).
          - XOR Verify: flex-shrink-0, natural height (4-row table, always fits).
          - Math Audit: flex-1 min-h-0 overflow-y-auto — grows to fill what's
            left and scrolls when the z/a listing is taller than the panel.
          - Params: flex-shrink-0, small fixed block at the bottom.
        */}
        <div className="w-80 border-l border-slate-700 flex flex-col overflow-hidden flex-shrink-0">

          {/* ── PyTorch code — 240px tall, scrollable internally ─────────── */}
          <div className="flex flex-col flex-shrink-0 p-3 border-b border-slate-700"
               style={{ height: '240px' }}>
            <div className="flex items-center justify-between mb-1.5 flex-shrink-0">
              <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">PyTorch Equivalent</h2>
              <span className="text-xs text-slate-600">explanatory only</span>
            </div>
            {/* flex-1 min-h-0: fills remaining height of the 240px parent so
                PyTorchSidebar's h-full + overflow-y-auto actually works */}
            <div className="flex-1 min-h-0">
              <PyTorchSidebar layerSizes={layerSizes} hiddenActivationTypes={activationTypes} />
            </div>
          </div>

          {/* ── XOR Verification table — natural height, never overflows ─── */}
          <div className="flex-shrink-0 p-3 border-b border-slate-700">
            <div className="bg-slate-800/40 rounded-lg border border-slate-700 p-2.5">
              <XorVerifyPanel xorResults={xorResults} />
            </div>
          </div>

          {/* ── Math Audit — fills remaining height, scrolls if needed ───── */}
          <div className="flex-1 min-h-0 overflow-y-auto p-3 border-b border-slate-700">
            <div className="bg-slate-800/40 rounded-lg border border-slate-700 p-2.5">
              <MathAuditPanel
                xorResults={xorResults}
                hiddenActivationTypes={activationTypes}
                layerSizes={layerSizes}
              />
            </div>
          </div>

          {/* ── Parameter count — fixed footer ───────────────────────────── */}
          {network && (
            <div className="flex-shrink-0 p-3">
              <div className="bg-slate-800 rounded p-2 text-xs font-mono border border-slate-700">
                <div className="text-slate-500 mb-1">Parameters</div>
                {network.weights.map((W, l) => (
                  <div key={l} className="text-slate-400">
                    L{l+1}: {W[0].length}×{W.length}+{W.length}b = {W.length*W[0].length+W.length}
                  </div>
                ))}
                <div className="text-blue-400 mt-0.5">
                  Total: {network.weights.reduce((s,W) => s+W.length*W[0].length+W.length, 0)}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
