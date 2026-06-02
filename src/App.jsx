// =============================================================================
// Neural Net Playground
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
  ReferenceLine, ReferenceDot,
} from 'recharts';

// =============================================================================
// NEURAL-NETWORK MATH CORE
// All from-scratch math (activations, forward pass, loss, backprop, gradient
// descent, evaluation, convergence and the finite-difference gradient check)
// lives in src/nn/ and is unit-tested there. No ML libraries.
// =============================================================================
import {
  ACTIVATIONS,
  XOR_DATA,
  initNetwork,
  forwardPass,
  computeLoss,
  backprop,
  updateWeights,
  trainOneEpoch,
  computeDecisionBoundary,
  evaluateXOR,
  checkConvergence,
  computeActivationCurve,
  runGradientCheck,
  CONVERGENCE_LOSS_THRESHOLD,
  CONVERGENCE_CONSECUTIVE_EPOCHS,
  CONVERGENCE_CONFIDENCE,
  PLATEAU_PATIENCE,
  MIN_IMPROVEMENT,
  PLATEAU_MIN_LOSS,
} from './nn/index.js';

// =============================================================================
// SECTION 10 — PYTORCH CODE GENERATOR
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
// SECTION 10b — EXPORT UTILITIES
// generateFullScript  → runnable .py file (clipboard copy)
// generateNotebook    → .ipynb JSON (browser download, no backend)
//
// Neither function exports the current trained weights; both reinitialize
// randomly. The architecture and hyper-parameters do match the current UI.
// =============================================================================

function generateFullScript(layerSizes, hiddenActivationTypes, learningRate) {
  const actMap = { relu: 'nn.ReLU()', tanh: 'nn.Tanh()', sigmoid: 'nn.Sigmoid()' };
  const paramCount = layerSizes.slice(0, -1).reduce(
    (s, n, i) => s + n * layerSizes[i + 1] + layerSizes[i + 1], 0
  );
  const modelLines = [];
  modelLines.push(`    nn.Linear(${layerSizes[0]}, ${layerSizes[1]}),`);
  modelLines.push(`    ${actMap[hiddenActivationTypes[0]] || 'nn.ReLU()'},`);
  for (let i = 1; i < layerSizes.length - 2; i++) {
    modelLines.push(`    nn.Linear(${layerSizes[i]}, ${layerSizes[i + 1]}),`);
    modelLines.push(`    ${actMap[hiddenActivationTypes[i]] || 'nn.ReLU()'},`);
  }
  modelLines.push(`    nn.Linear(${layerSizes[layerSizes.length - 2]}, 1),`);
  modelLines.push(`    nn.Sigmoid(),`);

  return `#!/usr/bin/env python3
"""
XOR Neural Network — exported from Neural Net Playground
Architecture : ${layerSizes.slice(0, -1).join(' -> ')} -> 1
Activations  : ${hiddenActivationTypes.join(', ')} + Sigmoid (output)
Parameters   : ${paramCount}
Learning rate: ${learningRate}

Note: weights are randomly re-initialized, not copied from the simulator.
      Re-run until convergence — this architecture reliably solves XOR.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)  # remove for random init each run

# -- Dataset ------------------------------------------------------------------
# XOR: output is 1 iff exactly one input is 1 (not linearly separable).
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = torch.tensor([[0],    [1],    [1],    [0]],     dtype=torch.float)

# -- Model --------------------------------------------------------------------
# nn.Linear(in, out) creates weight matrix W [out x in] and bias b [out].
# Each layer computes: a = activation(W*x + b)
model = nn.Sequential(
${modelLines.join('\n')}
)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# -- Loss and optimizer -------------------------------------------------------
# BCELoss: L = -(1/N) * sum[y*log(y_hat) + (1-y)*log(1-y_hat)]
# SGD:     W <- W - lr * dL/dW   (same update rule as the simulator)
criterion = nn.BCELoss()
optimizer  = torch.optim.SGD(model.parameters(), lr=${learningRate})

# -- Training loop ------------------------------------------------------------
losses = []
for epoch in range(10_000):
    optimizer.zero_grad()        # clear accumulated gradients
    out  = model(X)              # forward pass (activations left -> right)
    loss = criterion(out, y)     # BCE loss over all 4 XOR samples
    loss.backward()              # backprop: dL/dW via chain rule
    optimizer.step()             # W <- W - lr * dL/dW
    losses.append(loss.item())
    if loss.item() < 0.001:      # same convergence threshold as simulator
        print(f"Converged at epoch {epoch + 1},  loss = {loss.item():.6f}")
        break
else:
    print(f"Max epochs reached,  final loss = {losses[-1]:.6f}")

# -- Loss curve ---------------------------------------------------------------
plt.figure(figsize=(8, 3))
plt.plot(losses, lw=1, color="#60a5fa", label="BCE Loss")
plt.axhline(0.001, color="#10b981", ls="--", lw=1, label="convergence (0.001)")
plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title("XOR Training Loss")
plt.legend(); plt.tight_layout(); plt.show()

# -- XOR verification ---------------------------------------------------------
# model.eval() disables training-only layers (none here, but good practice).
# torch.no_grad() skips gradient tracking during inference.
model.eval()
with torch.no_grad():
    preds = model(X)

print("\\nXOR Verification:")
print(f"{'Input':<12} {'Expected':<10} {'p(class=1)':<14} {'Class':<8} Correct?")
print("-" * 55)
all_correct = True
for xi, yi, pi in zip(X, y, preds):
    p    = pi.item()
    pred = 1 if p > 0.5 else 0
    conf = abs(p - 0.5) * 2
    ok   = pred == int(yi.item())
    all_correct = all_correct and ok
    print(f"{str(xi.tolist()):<12} {int(yi.item()):<10} {p:.4f}         {pred:<8} {'OK' if ok else 'WRONG'}  ({conf:.0%})")
print("\\nAll correct!" if all_correct else "\\nNot all correct -- try re-running.")

# -- Decision boundary --------------------------------------------------------
grid_n = 60
xs     = torch.linspace(0, 1, grid_n)
xx, yy = torch.meshgrid(xs, xs, indexing="xy")
grid   = torch.stack([xx.flatten(), yy.flatten()], dim=1)
with torch.no_grad():
    grid_p = model(grid).reshape(grid_n, grid_n).numpy()

fig, ax = plt.subplots(figsize=(4, 4))
ax.contourf(xs.numpy(), xs.numpy(), grid_p,
            levels=[0, 0.5, 1], colors=["#3b82f6", "#f97316"], alpha=0.35)
ax.contour(xs.numpy(), xs.numpy(), grid_p, levels=[0.5], colors="white", linewidths=1)
for xi, yi in zip(X, y):
    c = "#f97316" if yi.item() == 1 else "#3b82f6"
    ax.scatter(xi[0].item(), xi[1].item(), color=c, s=120, zorder=5,
               edgecolors="white", linewidths=1.5)
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title("Decision Boundary")
plt.tight_layout(); plt.show()

# -- Custom inference ---------------------------------------------------------
test_pt = torch.tensor([[0.2, 0.8]], dtype=torch.float)
with torch.no_grad():
    prob = model(test_pt).item()
conf = abs(prob - 0.5) * 2
print(f"\\nInference: {test_pt.tolist()[0]} -> p(1)={prob:.4f}, "
      f"class={1 if prob > 0.5 else 0}, conf={conf:.0%}")
`;
}

// Generates an .ipynb JSON that Jupyter can open directly.
// Architecture, activations, and learning rate match the current UI state.
// Each cell's source is an array of line-strings as the nbformat v4 spec requires.
function generateNotebook(layerSizes, hiddenActivationTypes, learningRate) {
  const actMap = { relu: 'nn.ReLU()', tanh: 'nn.Tanh()', sigmoid: 'nn.Sigmoid()' };
  const paramCount = layerSizes.slice(0, -1).reduce(
    (s, n, i) => s + n * layerSizes[i + 1] + layerSizes[i + 1], 0
  );

  // nbformat v4: source is an array where every line except the last ends with '\n'
  const toSrc = str => {
    const lines = str.replace(/^\n/, '').replace(/\n$/, '').split('\n');
    return lines.map((l, i) => (i < lines.length - 1 ? l + '\n' : l));
  };
  const md   = s => ({ cell_type: 'markdown', metadata: {}, source: toSrc(s) });
  const code = s => ({
    cell_type: 'code', execution_count: null, metadata: {}, outputs: [],
    source: toSrc(s),
  });

  const modelLines = [];
  modelLines.push(`    nn.Linear(${layerSizes[0]}, ${layerSizes[1]}),`);
  modelLines.push(`    ${actMap[hiddenActivationTypes[0]] || 'nn.ReLU()'},`);
  for (let i = 1; i < layerSizes.length - 2; i++) {
    modelLines.push(`    nn.Linear(${layerSizes[i]}, ${layerSizes[i + 1]}),`);
    modelLines.push(`    ${actMap[hiddenActivationTypes[i]] || 'nn.ReLU()'},`);
  }
  modelLines.push(`    nn.Linear(${layerSizes[layerSizes.length - 2]}, 1),`);
  modelLines.push(`    nn.Sigmoid(),`);
  const modelBlock = modelLines.join('\n');

  const archStr = layerSizes.slice(0, -1).join(' → ') + ' → 1';
  const actStr  = hiddenActivationTypes.join(', ');

  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
      language_info: { name: 'python', version: '3.9.0' },
    },
    cells: [
      md(`# XOR Neural Network — Neural Net Playground Export

This notebook was exported from **Neural Net Playground** — an interactive MLP
simulator that trains on XOR using real backpropagation (no ML libraries).

## Configuration

| Property | Value |
|----------|-------|
| Architecture | ${archStr} |
| Hidden activations | ${actStr} |
| Output activation | Sigmoid |
| Loss function | Binary Cross-Entropy (BCELoss) |
| Optimizer | SGD, lr = ${learningRate} |
| Total parameters | ${paramCount} |

> **Note:** Weights are randomly re-initialized here, not copied from the simulator.
> Re-run cells until convergence — this architecture reliably solves XOR.`),

      code(`import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)  # remove for random init each run
print("PyTorch:", torch.__version__)`),

      md(`## 1 · XOR Dataset

XOR outputs **1** iff exactly one input is 1. It is *not linearly separable* —
a single-layer perceptron cannot solve it, but a two-layer MLP can.

| x₁ | x₂ | y |
|:--:|:--:|:-:|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |`),

      code(`X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = torch.tensor([[0],    [1],    [1],    [0]],     dtype=torch.float)
print("X:", X.shape, " y:", y.shape)`),

      md(`## 2 · Model Definition

\`nn.Linear(in, out)\` creates a weight matrix **W** [out × in] and bias **b** [out].
Each layer's forward computation: **z = W·x + b**, then **a = activation(z)**.

This matches the network graph in the simulator — every node is one activation value.`),

      code(`model = nn.Sequential(
${modelBlock}
)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
print(model)`),

      md(`## 3 · Loss Function and Optimizer

**Binary Cross-Entropy (BCELoss):**

$$L = -\\frac{1}{N} \\sum_{i=1}^{N} \\bigl[ y_i \\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i) \\bigr]$$

**SGD** updates weights via gradient descent:

$$W \\leftarrow W - \\eta \\cdot \\frac{\\partial L}{\\partial W}, \\quad \\eta = ${learningRate}$$`),

      code(`criterion = nn.BCELoss()
optimizer  = torch.optim.SGD(model.parameters(), lr=${learningRate})
print(criterion)
print(optimizer)`),

      md(`## 4 · Training Loop

Each epoch is a full pass over all 4 XOR samples. The five lines map directly
to the **Explained Step** mode in the simulator:

| Step | Code | Simulator stage |
|------|------|-----------------|
| Clear gradients | \`optimizer.zero_grad()\` | — |
| Forward pass | \`out = model(X)\` | Stage 1 |
| Compute loss | \`loss = criterion(out, y)\` | Stage 2 |
| Backpropagation | \`loss.backward()\` | Stage 3 |
| Weight update | \`optimizer.step()\` | Stage 4 |`),

      code(`losses = []
for epoch in range(10_000):
    optimizer.zero_grad()        # clear accumulated gradients
    out  = model(X)              # forward pass: activations left -> right
    loss = criterion(out, y)     # BCE loss over all 4 samples
    loss.backward()              # backprop: dL/dW via chain rule
    optimizer.step()             # W <- W - lr * dL/dW
    losses.append(loss.item())
    if loss.item() < 0.001:      # same convergence threshold as simulator
        print(f"Converged at epoch {epoch + 1},  loss = {loss.item():.6f}")
        break
else:
    print(f"Max epochs reached.  Final loss = {losses[-1]:.6f}")`),

      md(`## 5 · Loss Curve

A healthy training run shows BCE loss falling monotonically toward the convergence
threshold (0.001). If it plateaus, try reinitializing with a new seed, switching
to **Tanh** activations, or using **Adam** (lr=0.01).`),

      code(`fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(losses, lw=1, color="#60a5fa", label="BCE Loss")
ax.axhline(0.001, color="#10b981", ls="--", lw=1, label="convergence (0.001)")
ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss"); ax.set_title("XOR Training Loss")
ax.legend(); plt.tight_layout(); plt.show()`),

      md(`## 6 · XOR Verification

Inference best practice:
- \`model.eval()\` — disables dropout / batch-norm (none here, good habit)
- \`torch.no_grad()\` — skips gradient computation → faster, less memory

**Confidence** = \`|p − 0.5| × 2\`: 0 = on the decision boundary, 1 = fully certain.`),

      code(`model.eval()
with torch.no_grad():
    preds = model(X)

print(f"{'Input':<12} {'Expected':<10} {'p(1)':<10} {'Class':<8} OK?")
print("-" * 45)
all_ok = True
for xi, yi, pi in zip(X, y, preds):
    p  = pi.item(); pred = 1 if p > 0.5 else 0
    ok = pred == int(yi.item()); all_ok = all_ok and ok
    print(f"{str(xi.tolist()):<12} {int(yi.item()):<10} {p:.4f}     {pred:<8} {'V' if ok else 'X'}  ({abs(p-.5)*2:.0%})")
print("\\nAll correct!" if all_ok else "\\nNot all correct -- re-run.")`),

      md(`## 7 · Decision Boundary

Grid the input space [0, 1]² and run a forward pass at every point.
Blue = class 0, orange = class 1, white contour = decision boundary (p = 0.5).
This matches the **Decision Boundary** canvas in the simulator.`),

      code(`grid_n = 60
xs     = torch.linspace(0, 1, grid_n)
xx, yy = torch.meshgrid(xs, xs, indexing="xy")
grid   = torch.stack([xx.flatten(), yy.flatten()], dim=1)
model.eval()
with torch.no_grad():
    grid_p = model(grid).reshape(grid_n, grid_n).numpy()

fig, ax = plt.subplots(figsize=(4, 4))
ax.contourf(xs.numpy(), xs.numpy(), grid_p,
            levels=[0, 0.5, 1], colors=["#3b82f6", "#f97316"], alpha=0.4)
ax.contour(xs.numpy(), xs.numpy(), grid_p, levels=[0.5], colors="white", linewidths=1.2)
for xi, yi in zip(X, y):
    c = "#f97316" if yi.item() == 1 else "#3b82f6"
    ax.scatter(xi[0].item(), xi[1].item(), color=c, s=150, zorder=5,
               edgecolors="white", linewidths=1.5)
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title("Decision Boundary")
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
plt.tight_layout(); plt.show()`),

      md(`## 8 · Custom Inference

Run the trained network on any point in [0, 1]².
Always use \`model.eval()\` + \`torch.no_grad()\` for inference.`),

      code(`test_pt = torch.tensor([[0.2, 0.8]], dtype=torch.float)
model.eval()
with torch.no_grad():
    prob = model(test_pt).item()
conf = abs(prob - 0.5) * 2
print(f"Input:      {test_pt.tolist()[0]}")
print(f"p(class=1): {prob:.4f}")
print(f"Class:      {1 if prob > 0.5 else 0}")
print(f"Confidence: {conf:.1%}  (= |p - 0.5| * 2)")`),
    ],
  };
}

// =============================================================================
// SECTION 11 — COLOR UTILITIES
// =============================================================================

// Activation magnitude → blue (low/negative) to orange (high/positive)
function activationColor(value) {
  const t = Math.max(0, Math.min(1, (value + 1) / 2));
  return `rgba(${Math.round(t*251+(1-t)*30)},${Math.round(t*146+(1-t)*144)},${Math.round(t*60+(1-t)*255)},1)`;
}

// Decision boundary: class-0=blue, class-1=orange
function boundaryColor(prob) {
  return `rgb(${Math.round(prob*251+(1-prob)*59)},${Math.round(prob*146+(1-prob)*130)},${Math.round(prob*60+(1-prob)*246)})`;
}

// Gradient magnitude → gray (≈0) to red (large)
function gradientColor(magnitude, maxMag) {
  if (maxMag === 0) return 'rgba(100,100,100,0.3)';
  const t = Math.min(1, magnitude / maxMag);
  return `rgba(${Math.round(t*239+(1-t)*75)},${Math.round(t*68+(1-t)*85)},${Math.round(t*68+(1-t)*99)},${0.3+t*0.7})`;
}

// Confidence heatmap: dark slate (uncertain, near boundary) → amber (very confident)
// conf = |p − 0.5| × 2 maps to [0,1]: 0 = on decision boundary, 1 = fully certain
function confidenceColor(prob) {
  const conf = Math.abs(prob - 0.5) * 2;
  const r = Math.round(30  + conf * (251 - 30));
  const g = Math.round(41  + conf * (191 - 41));
  const b = Math.round(59  + conf * (36  - 59));
  return `rgb(${r},${g},${b})`;
}

// Weight value → cool (negative) or warm (positive), intensity proportional to magnitude.
// maxMag is the largest |weight| in the layer (for per-layer normalization).
function weightColor(v, maxMag) {
  if (maxMag < 1e-8) return 'rgba(71,85,105,0.2)';
  const t = Math.max(-1, Math.min(1, v / maxMag));
  const a = (0.12 + Math.abs(t) * 0.78).toFixed(2);
  return t >= 0
    ? `rgba(245,158,11,${a})`   // amber — positive weight
    : `rgba(59,130,246,${a})`;  // blue  — negative weight
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
// COMPONENT: NetworkGraph
//
// Props:
//   backpropData       — { dWeights, dBiases } from last backprop; edges colored
//                        by ∂L/∂W magnitude when present
//   animatingBackward  — true during the backprop stage of an explained epoch;
//                        uses violet highlights instead of blue so the direction
//                        of gradient flow (right→left) is visually distinct
//   showGradientLabels — show numeric ∂L/∂W values on edges when backpropData present
// =============================================================================
function NetworkGraph({ layerSizes, hiddenActivationTypes, forwardData, backpropData,
                        animatingLayer, animatingBackward, showGradientLabels }) {
  const SVG_W = 520, SVG_H = 320, R = 18;
  const layout = computeLayout(layerSizes, SVG_W, SVG_H);

  let maxGradMag = 0;
  if (backpropData) {
    for (const W of backpropData.dWeights)
      for (const row of W)
        for (const v of row)
          if (Math.abs(v) > maxGradMag) maxGradMag = Math.abs(v);
  }

  // When there are many edges, only label the top-8 by magnitude to avoid clutter.
  // With <= 16 total edges, all are labeled.
  const totalEdges = layerSizes.slice(0,-1).reduce(
    (sum, sz, li) => sum + sz * layerSizes[li + 1], 0
  );
  let labeledEdgeKeys = null; // null = label all
  if (backpropData && showGradientLabels && totalEdges > 16) {
    const allEdgeMags = [];
    layout.slice(0,-1).forEach((fromLayer, li) => {
      fromLayer.forEach((_, fi) => {
        layout[li+1].forEach((_, ti) => {
          const v = backpropData.dWeights[li]?.[ti]?.[fi];
          if (v !== undefined) allEdgeMags.push({ key: `${li}-${fi}-${ti}`, m: Math.abs(v) });
        });
      });
    });
    allEdgeMags.sort((a, b) => b.m - a.m);
    labeledEdgeKeys = new Set(allEdgeMags.slice(0, 8).map(e => e.key));
  }

  return (
    <div>
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="w-full block"
        style={{ height: `${SVG_H}px` }}
      >
      {/* ── Edges ──────────────────────────────────────────────────────────── */}
      {layout.slice(0, -1).map((fromLayer, li) =>
        fromLayer.map((from, fi) =>
          layout[li + 1].map((to, ti) => {
            let stroke = 'rgba(148,163,184,0.15)', sw = 1;

            const isActive = animatingLayer >= 0 &&
              (li === animatingLayer - 1 || li === animatingLayer);

            if (backpropData?.dWeights[li]) {
              // Color edges by gradient magnitude
              const m = Math.abs(backpropData.dWeights[li][ti][fi]);
              stroke = gradientColor(m, maxGradMag);
              sw = 1 + 2 * (m / (maxGradMag || 1));
              // During backward animation: violet highlight for the active layer
              if (isActive && animatingBackward) {
                stroke = 'rgba(167,139,250,0.9)';
                sw = 2.5;
              }
            } else if (isActive && !animatingBackward) {
              // Forward animation highlight (no gradient data yet)
              stroke = 'rgba(96,165,250,0.5)';
              sw = 1.5;
            }

            return (
              <line key={`e-${li}-${fi}-${ti}`}
                x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                stroke={stroke} strokeWidth={sw} />
            );
          })
        )
      )}

      {/* ── Gradient magnitude labels on edges ─────────────────────────────
          Show numeric ∂L/∂W values at each edge midpoint so the user can see
          the actual numbers behind the color encoding. For dense networks,
          only the 8 highest-magnitude gradients are labeled to avoid overlap. */}
      {backpropData && showGradientLabels &&
        layout.slice(0, -1).map((fromLayer, li) =>
          fromLayer.map((from, fi) =>
            layout[li + 1].map((to, ti) => {
              const v = backpropData.dWeights[li]?.[ti]?.[fi];
              if (v === undefined) return null;
              // Skip label if we're in "top-N only" mode and this edge isn't included
              if (labeledEdgeKeys && !labeledEdgeKeys.has(`${li}-${fi}-${ti}`)) return null;
              const mx = (from.x + to.x) / 2;
              const my = (from.y + to.y) / 2;
              // Format: 3 decimal places; very small values get scientific notation
              const label = Math.abs(v) < 0.0005
                ? Math.abs(v).toExponential(1)
                : Math.abs(v).toFixed(3);
              const w = label.length * 3.8 + 4;
              return (
                <g key={`gl-${li}-${fi}-${ti}`}>
                  {/* Semi-transparent background for readability */}
                  <rect x={mx - w/2} y={my - 5} width={w} height={9}
                    fill="rgba(15,23,42,0.75)" rx={1} />
                  <text x={mx} y={my + 1} textAnchor="middle" dominantBaseline="middle"
                    fill="#e2e8f0" fontSize={6.5} fontFamily="monospace"
                    style={{ userSelect: 'none' }}>
                    {label}
                  </text>
                </g>
              );
            })
          )
        )
      }

      {/* ── Neurons ────────────────────────────────────────────────────────── */}
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
                stroke={hasAct ? '#e2e8f0' : '#475569'} strokeWidth={1.5} />
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

      {/* ── Layer labels ───────────────────────────────────────────────────── */}
      {layout.map((layer, li) => (
        <text key={`lbl-${li}`} x={layer[0].x} y={14} textAnchor="middle" fill="#94a3b8" fontSize={10}>
          {li === 0 ? 'Input(2)' : li === layerSizes.length-1 ? 'Output(1)' : `Hidden${li}(${layerSizes[li]})`}
        </text>
      ))}

    </svg>
    {/* Gradient legend in HTML so it can never overlap SVG neurons */}
    {backpropData && (
      <div className="flex items-center gap-1.5 px-1 pt-1 font-mono text-slate-500 flex-wrap"
           style={{ fontSize: '7.5px' }}>
        <span>Edge color = |∂L/∂w| (absolute magnitude · avg over 4 XOR samples)</span>
        <div className="flex shrink-0">
          {[0, .17, .33, .5, .67, .83, 1].map((t, i) => (
            <div key={i} style={{ width: 14, height: 8, backgroundColor: gradientColor(t, 1) }} />
          ))}
        </div>
        <span className="text-slate-600 shrink-0">0</span>
        <span className="text-slate-400 shrink-0">
          {maxGradMag > 0 ? `max = ${maxGradMag.toFixed(4)}` : 'max = n/a'}
        </span>
        <span className="text-slate-700 shrink-0">
          {totalEdges > 16 ? '(top 8 edges labeled)' : '(all edges labeled)'}
        </span>
      </div>
    )}
    </div>
  );
}

// =============================================================================
// COMPONENT: DecisionBoundaryCanvas
//
// showConfidence: when true, colors each grid cell by confidence instead of class.
//   Class mode:      blue=class 0, orange=class 1
//   Confidence mode: dark=near decision boundary (uncertain), amber=confident
//   Both use ACTUAL forward passes — nothing is approximated.
// =============================================================================
function DecisionBoundaryCanvas({ weights, biases, hiddenActivationTypes,
                                   inferencePoint, onClick, showConfidence }) {
  const canvasRef = useRef(null);
  const GRID      = 40;
  const CANVAS_SZ = 260;
  const POINT_R   = 7;

  useEffect(() => {
    if (!weights || !canvasRef.current) return;
    const ctx      = canvasRef.current.getContext('2d');
    const cellSize = CANVAS_SZ / GRID;

    const grid = computeDecisionBoundary(weights, biases, hiddenActivationTypes, GRID);
    ctx.clearRect(0, 0, CANVAS_SZ, CANVAS_SZ);
    for (let row = 0; row < GRID; row++) {
      for (let col = 0; col < GRID; col++) {
        // showConfidence: brightness encodes |p−0.5|×2 regardless of class
        ctx.fillStyle   = showConfidence
          ? confidenceColor(grid[row][col])
          : boundaryColor(grid[row][col]);
        ctx.globalAlpha = 0.65;
        ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
      }
    }

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
  }, [weights, biases, hiddenActivationTypes, inferencePoint, showConfidence]);

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
// =============================================================================
function XorVerifyPanel({ xorResults }) {
  if (!xorResults) return (
    <div className="text-xs text-slate-600 italic px-1">Train to see XOR verification</div>
  );
  const allCorrect  = xorResults.every(r => r.correct);
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
        <div className="text-slate-400">
          <span className="text-slate-500">Input  </span>
          x = [{input.join(', ')}]
          <span className="text-slate-600 ml-1">(label={label})</span>
        </div>

        {activations.slice(1).map((a, li) => {
          const z         = preActivations[li + 1];
          const isOutput  = li === activations.length - 2;
          const activType = isOutput ? 'sigmoid' : (hiddenActivationTypes[li] || 'relu');
          return (
            <div key={li} className="border-t border-slate-800 pt-1.5">
              <div className="text-slate-500 text-xs mb-0.5">
                Layer {li + 1} ({isOutput ? 'output, sigmoid' : activType})
              </div>
              <div className="text-indigo-300">z = [{z.map(v => v.toFixed(4)).join(', ')}]</div>
              <div className="text-blue-300">
                a = [{a.map(v => v.toFixed(4)).join(', ')}]
                {isOutput && <span className="text-slate-500 ml-1">← p(class=1)</span>}
              </div>
            </div>
          );
        })}

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
// COMPONENT: GradientCheckPanel
//
// Lets the user select any single weight W[l][j][k] and verify that the
// backprop gradient matches the centered finite-difference approximation.
// Shows both values side by side with the absolute and relative error so the
// user can see for themselves that the math is correct.
// =============================================================================
function GradientCheckPanel({ network, hiddenActivationTypes, layerSizes, lastGradients }) {
  const [layerIdx, setLayerIdx] = useState(0);
  const [rowIdx,   setRowIdx]   = useState(0);
  const [colIdx,   setColIdx]   = useState(0);
  const [result,   setResult]   = useState(null);

  if (!network) return (
    <div className="text-xs text-slate-600 italic">Initialize or train the network first</div>
  );

  // Clamp j/k indices whenever the layer selector changes
  const maxRow = (network.weights[layerIdx]?.length       ?? 1) - 1;
  const maxCol = (network.weights[layerIdx]?.[0]?.length  ?? 1) - 1;
  const safeRow = Math.min(rowIdx, maxRow);
  const safeCol = Math.min(colIdx, maxCol);

  const currentW = network.weights[layerIdx]?.[safeRow]?.[safeCol] ?? 0;

  const handleLayerChange = (v) => { setLayerIdx(+v); setResult(null); };
  const handleRowChange   = (v) => { setRowIdx(+v);   setResult(null); };
  const handleColChange   = (v) => { setColIdx(+v);   setResult(null); };

  // Scan lastGradients for the weight with the largest |∂L/∂w| and jump to it.
  // Picking the highest-magnitude gradient gives the most interesting check:
  // a near-zero finite-difference on a near-zero gradient tells you very little,
  // but exact agreement on a large gradient is strong evidence backprop is right.
  const handleAutoPick = () => {
    if (!lastGradients) return;
    let bestL = 0, bestJ = 0, bestK = 0, bestMag = -1;
    lastGradients.dWeights.forEach((W, l) => {
      W.forEach((row, j) => {
        row.forEach((v, k) => {
          if (Math.abs(v) > bestMag) {
            bestMag = Math.abs(v);
            bestL = l; bestJ = j; bestK = k;
          }
        });
      });
    });
    setLayerIdx(bestL);
    setRowIdx(bestJ);
    setColIdx(bestK);
    setResult(null);
  };

  const handleRun = () => {
    const r = runGradientCheck(
      network.weights, network.biases, hiddenActivationTypes,
      layerIdx, safeRow, safeCol
    );
    setResult(r);
  };

  // Quality tier based on relative error
  const quality = result
    ? result.relError < 1e-4
      ? { text: '✓ Excellent match  (rel err < 1e-4)', cls: 'text-emerald-400' }
      : result.relError < 1e-2
      ? { text: '~ Acceptable  (rel err < 1e-2)', cls: 'text-amber-400' }
      : { text: '⚠ Mismatch — possible backprop bug', cls: 'text-red-400' }
    : null;

  return (
    <div>
      <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        Gradient Check
      </div>

      {/* Weight selector ─────────────────────────────────────────────────── */}
      <div className="text-xs font-mono bg-slate-900/60 rounded p-2 space-y-1.5 mb-2">
        <div className="flex items-center gap-2">
          <span className="text-slate-500 w-24 shrink-0">Layer W[l]</span>
          <select value={layerIdx} onChange={e => handleLayerChange(e.target.value)}
            className="bg-slate-800 border border-slate-600 rounded text-white text-xs p-0.5 flex-1">
            {network.weights.map((_, li) => (
              <option key={li} value={li}>
                W[{li}]  {layerSizes[li]}→{layerSizes[li + 1]}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-500 w-24 shrink-0">To neuron j</span>
          <select value={safeRow} onChange={e => handleRowChange(e.target.value)}
            className="bg-slate-800 border border-slate-600 rounded text-white text-xs p-0.5 flex-1">
            {Array.from({ length: maxRow + 1 }, (_, i) => (
              <option key={i} value={i}>{i}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-500 w-24 shrink-0">From neuron k</span>
          <select value={safeCol} onChange={e => handleColChange(e.target.value)}
            className="bg-slate-800 border border-slate-600 rounded text-white text-xs p-0.5 flex-1">
            {Array.from({ length: maxCol + 1 }, (_, i) => (
              <option key={i} value={i}>{i}</option>
            ))}
          </select>
        </div>
        <div className="text-slate-600 border-t border-slate-800 pt-1.5 text-xs">
          W[{layerIdx}][{safeRow}][{safeCol}] ={' '}
          <span className="text-slate-300">{currentW.toFixed(6)}</span>
        </div>
      </div>

      <div className="flex gap-1.5 mb-3">
        <button onClick={handleAutoPick} disabled={!lastGradients}
          title="Select the weight with the largest |∂L/∂w| from the last backward pass"
          className="flex-1 py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 border border-slate-600 disabled:opacity-40 disabled:cursor-not-allowed">
          Auto-pick max |∂w|
        </button>
        <button onClick={handleRun}
          className="flex-1 py-1 rounded text-xs bg-indigo-900/60 hover:bg-indigo-800 text-indigo-300 border border-indigo-700/50">
          Run Check  (ε = 1e-4)
        </button>
      </div>

      {/* Results ─────────────────────────────────────────────────────────── */}
      {result && (
        <div className="text-xs font-mono bg-slate-900/60 rounded p-2 space-y-1">
          {/* Intermediate loss values so the user can see the raw computation */}
          <div className="text-slate-500 border-b border-slate-800 pb-1 mb-1">
            ε = {result.epsilon}  ·  full-batch (N = 4 XOR samples)
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Loss(w + ε)</span>
            <span className="text-slate-300">{result.lossPlus.toFixed(8)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Loss(w − ε)</span>
            <span className="text-slate-300">{result.lossMinus.toFixed(8)}</span>
          </div>

          {/* The two gradient estimates side by side */}
          <div className="border-t border-slate-800 pt-1 mt-0.5 space-y-1">
            <div className="flex justify-between">
              <span className="text-violet-400">Backprop  ∂L/∂w</span>
              <span className="text-violet-300 font-bold">{result.backpropGrad >= 0 ? ' ' : ''}{result.backpropGrad.toFixed(8)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-400">Finite diff  ∂L/∂w</span>
              <span className="text-blue-300 font-bold">{result.fdGrad >= 0 ? ' ' : ''}{result.fdGrad.toFixed(8)}</span>
            </div>
          </div>

          {/* Error metrics */}
          <div className="border-t border-slate-800 pt-1 mt-0.5 space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">|error|  (absolute)</span>
              <span className="text-slate-200">{result.absError.toExponential(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">rel error</span>
              <span className="text-slate-200">{result.relError.toExponential(3)}</span>
            </div>
          </div>

          <div className={`border-t border-slate-800 pt-1.5 font-bold ${quality.cls}`}>
            {quality.text}
          </div>

          <div className="text-slate-600 text-xs pt-0.5">
            Relative error = |err| / (|backprop| + |fd|)
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// COMPONENT: ChainRuleTracer
//
// For a selected weight W[l][j][k], shows the full chain-rule derivation using
// actual values from a fresh per-sample backprop (not the stored average).
//
// Math:  ∂L/∂w_{jk}^{(l)} = δⱼ · aₖ
//   where δⱼ = ∂L/∂zⱼ  (error signal at destination neuron's pre-activation)
//         aₖ = activations[l][k]  (incoming activation through this weight)
//
// The stored lastGradients.dWeights are *averaged* over all 4 XOR samples.
// The trace below shows per-sample values for the selected input, with a
// comparison to the batch-average gradient that actually drives training.
// =============================================================================
function ChainRuleTracer({ network, layerSizes, hiddenActivationTypes, lastGradients }) {
  const [selLayer,  setSelLayer]  = useState(0);
  const [selOutN,   setSelOutN]   = useState(0);
  const [selInN,    setSelInN]    = useState(0);
  const [selSample, setSelSample] = useState(0);

  if (!network || !lastGradients) {
    return (
      <div className="text-slate-500 text-xs italic text-center py-6 px-2">
        Run at least one training step or Explained Step to trace a gradient.
      </div>
    );
  }

  const L       = network.weights.length;
  const safeL   = Math.min(selLayer, L - 1);
  const safeJ   = Math.min(selOutN, layerSizes[safeL + 1] - 1);
  const safeK   = Math.min(selInN,  layerSizes[safeL]     - 1);

  // Re-run forward + backprop on the selected XOR sample to get per-sample deltas.
  // backprop() returns deltas[] where deltas[l+1][j] = ∂L/∂z_j for layer l+1.
  const sample = XOR_DATA[selSample];
  const { activations, preActivations } = forwardPass(
    sample.input, network.weights, network.biases, hiddenActivationTypes
  );
  const { dWeights: perDW, deltas } = backprop(
    [sample.label], activations, preActivations, network.weights, hiddenActivationTypes
  );

  const l = safeL, j = safeJ, k = safeK;
  const isOut    = l === L - 1;
  const actType  = isOut ? 'sigmoid' : (hiddenActivationTypes[l] || 'relu');
  const actLabel = ACTIVATIONS[actType].label;

  // The three quantities that compose the gradient
  const a_in    = activations[l][k];         // aₖ — incoming activation
  const z_j     = preActivations[l + 1][j];  // zⱼ — destination pre-activation
  const a_j     = activations[l + 1][j];     // f(zⱼ) — destination activation
  const delta_j = deltas[l + 1][j];          // δⱼ = ∂L/∂zⱼ
  const dAdZ_j  = ACTIVATIONS[actType].derivative(z_j);  // f′(zⱼ)

  // ∂L/∂aⱼ = δⱼ / f′(zⱼ) — only meaningful when f′ ≠ 0
  const dLdA_j  = Math.abs(dAdZ_j) > 1e-8 ? delta_j / dAdZ_j : null;

  const sampleGrad = perDW[l][j][k];             // δⱼ · aₖ for this sample
  const avgGrad    = lastGradients.dWeights[l][j][k]; // batch-averaged gradient
  const currentW   = network.weights[l][j][k];

  const dead      = actType === 'relu' && z_j <= 0;
  const saturated = actType !== 'relu' && Math.abs(dAdZ_j) < 0.05;

  const f = v => {
    if (!isFinite(v)) return '—';
    return Math.abs(v) < 0.0001 && v !== 0 ? v.toExponential(3) : v.toFixed(5);
  };

  return (
    <div className="space-y-3 text-xs">
      {/* ── Selectors ───────────────────────────────────────────────────────── */}
      <div className="space-y-1.5">
        <div className="text-slate-500 mb-0.5" style={{ fontSize: '9px' }}>
          Weight W[layer][j][k] to trace:
        </div>
        <div className="flex gap-1.5">
          {[
            {
              label: 'Layer',
              value: safeL,
              set: v => setSelLayer(+v),
              opts: Array.from({ length: L }, (_, i) => ({
                v: i, t: i === L - 1 ? `L${i + 1} (out)` : `L${i + 1}`,
              })),
            },
            {
              label: 'Out j',
              value: safeJ,
              set: v => setSelOutN(+v),
              opts: Array.from({ length: layerSizes[safeL + 1] }, (_, n) => ({ v: n, t: `n${n}` })),
            },
            {
              label: 'In k',
              value: safeK,
              set: v => setSelInN(+v),
              opts: Array.from({ length: layerSizes[safeL] }, (_, n) => ({ v: n, t: `n${n}` })),
            },
          ].map(({ label, value, set, opts }) => (
            <div key={label} className="flex-1">
              <div className="text-slate-600 mb-0.5" style={{ fontSize: '8px' }}>{label}</div>
              <select value={value} onChange={e => set(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-200 text-xs">
                {opts.map(o => <option key={o.v} value={o.v}>{o.t}</option>)}
              </select>
            </div>
          ))}
        </div>

        {/* XOR sample picker */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-slate-600" style={{ fontSize: '8px' }}>Input sample:</span>
          {XOR_DATA.map((s, i) => (
            <button key={i} onClick={() => setSelSample(i)}
              className={`px-1.5 py-0.5 rounded font-mono border transition-colors ${
                selSample === i
                  ? 'bg-blue-900/50 text-blue-300 border-blue-700'
                  : 'bg-slate-800 text-slate-500 border-slate-700 hover:text-slate-300'
              }`} style={{ fontSize: '9px' }}>
              [{s.input}]→{s.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Symbolic formula ─────────────────────────────────────────────────── */}
      <div className="bg-slate-900/70 border border-slate-700 rounded p-2 space-y-1">
        <div className="text-slate-600 uppercase tracking-wider" style={{ fontSize: '8px' }}>
          Symbolic formula
        </div>
        <div className="font-mono text-slate-200">
          ∂L/∂w = δⱼ · aₖ
        </div>
        {isOut ? (
          <>
            <div className="font-mono text-slate-400 text-xs">
              where δⱼ = ŷ − y
              <span className="text-slate-600 font-sans ml-1.5">(BCE + σ shortcut)</span>
            </div>
            <div className="text-slate-600 leading-tight" style={{ fontSize: '9px' }}>
              Output layer: ∂L/∂ŷ · ∂ŷ/∂z = [−y/ŷ+(1−y)/(1−ŷ)] · ŷ(1−ŷ) simplifies to ŷ−y.
              No separate σ′ term needed.
            </div>
          </>
        ) : (
          <>
            <div className="font-mono text-slate-400 text-xs">
              where δⱼ = (Σ w·δ<sub>next</sub>) · {actLabel}′(zⱼ)
            </div>
            <div className="text-slate-600 leading-tight" style={{ fontSize: '9px' }}>
              Hidden layer: backpropagated error from the next layer is multiplied by
              the local derivative — this is the chain rule applied recursively.
            </div>
          </>
        )}
      </div>

      {/* ── Numeric term cards ───────────────────────────────────────────────── */}
      <div className="space-y-1.5">
        <div className="text-slate-500 uppercase tracking-wider" style={{ fontSize: '8px' }}>
          Numeric substitution — sample [{sample.input}]→{sample.label}
        </div>

        {/* aₖ */}
        <div className="bg-slate-800/50 border border-slate-700/60 rounded p-1.5 flex items-start gap-2">
          <span className="font-mono text-amber-400 w-10 shrink-0 mt-px">aₖ</span>
          <div className="flex-1 text-slate-500 leading-tight" style={{ fontSize: '9px' }}>
            Activation of neuron {k} in {l === 0 ? 'the input layer' : `hidden layer ${l}`}.
            This is the signal flowing <em>into</em> weight w[{l}][{j}][{k}].
          </div>
          <span className="font-mono text-amber-300 shrink-0">{f(a_in)}</span>
        </div>

        {/* zⱼ + f′(zⱼ) */}
        <div className="bg-slate-800/50 border border-slate-700/60 rounded p-1.5 space-y-1">
          <div className="flex items-start gap-2">
            <span className="font-mono text-blue-400 w-10 shrink-0">zⱼ</span>
            <div className="flex-1 text-slate-500 leading-tight" style={{ fontSize: '9px' }}>
              Pre-activation (Σ w·a + b) at neuron {j} in layer {l + 1}.
            </div>
            <span className="font-mono text-blue-300 shrink-0">{f(z_j)}</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="font-mono text-blue-300 w-10 shrink-0">{actLabel}′</span>
            <div className="flex-1 text-slate-500 leading-tight" style={{ fontSize: '9px' }}>
              {isOut
                ? 'Shown for intuition only — not a separate factor in δ.'
                : 'Local derivative f′(zⱼ) — how much the activation changes per unit of z.'}
            </div>
            <span className={`font-mono shrink-0 ${!isOut && (dead || saturated) ? 'text-red-400' : 'text-slate-400'}`}>
              {f(dAdZ_j)}
            </span>
          </div>
          {/* Output layer: informational note about BCE+σ shortcut */}
          {isOut && (
            <div className="text-slate-600 leading-tight" style={{ fontSize: '9px' }}>
              Sigmoid′ shown for intuition only; BCE+sigmoid simplifies δ to ŷ−y, so this
              derivative is already accounted for. Saturation affects output confidence and
              model calibration, but δ is computed directly as ŷ−y regardless.
            </div>
          )}
          {/* Hidden-layer dead / saturated warnings only */}
          {!isOut && dead && (
            <div className="text-red-400 leading-tight" style={{ fontSize: '9px' }}>
              ⚠ Dead ReLU: z≤0 → f′=0. Gradient through this neuron is zero.
              For this sample, this path contributes no gradient. If the neuron is inactive
              for every training sample, its incoming weights will not learn until an update
              elsewhere makes z positive again.
            </div>
          )}
          {!isOut && !dead && saturated && (
            <div className="text-amber-400 leading-tight" style={{ fontSize: '9px' }}>
              ⚠ Saturated {actLabel}: f′≈0. Gradient barely flows — learning slows here.
            </div>
          )}
        </div>

        {/* δⱼ */}
        <div className="bg-slate-800/50 border border-slate-700/60 rounded p-1.5 flex items-start gap-2">
          <span className="font-mono text-violet-400 w-10 shrink-0 mt-px">δⱼ</span>
          <div className="flex-1 text-slate-500 leading-tight" style={{ fontSize: '9px' }}>
            {isOut
              ? `ŷ − y = ${f(a_j)} − ${sample.label}. Combined BCE+σ error signal.`
              : dLdA_j !== null
                ? `Backpropagated error ${f(dLdA_j)} × ${actLabel}′ ${f(dAdZ_j)} = chain-rule product.`
                : `f′(z)=0 — dead neuron, δⱼ=0, gradient blocked.`
            }
          </div>
          <span className="font-mono text-violet-300 shrink-0">{f(delta_j)}</span>
        </div>
      </div>

      {/* ── Result ──────────────────────────────────────────────────────────── */}
      <div className="bg-slate-900/80 border border-slate-600 rounded p-2 space-y-1.5">
        <div className="text-slate-500 uppercase tracking-wider" style={{ fontSize: '8px' }}>Result</div>
        <div className="font-mono flex items-center gap-1 flex-wrap">
          <span className="text-violet-300">{f(delta_j)}</span>
          <span className="text-slate-600">×</span>
          <span className="text-amber-300">{f(a_in)}</span>
          <span className="text-slate-600">=</span>
          <span className="text-emerald-400 font-bold">{f(sampleGrad)}</span>
        </div>
        <div className="flex gap-3 font-mono" style={{ fontSize: '9px' }}>
          <span className="text-slate-500">this sample:</span>
          <span className="text-emerald-400">{f(sampleGrad)}</span>
          <span className="text-slate-700">|</span>
          <span className="text-slate-500">batch avg (×4):</span>
          <span className="text-blue-400">{f(avgGrad)}</span>
        </div>
        <div className="text-slate-600 leading-tight" style={{ fontSize: '9px' }}>
          Training uses the batch average. Each sample contributes a different
          gradient; averaging reduces noise. Update: w ← w − lr·{f(avgGrad)}
          {' '}→ w will {avgGrad > 0 ? 'decrease' : avgGrad < 0 ? 'increase' : 'not change'}.
        </div>
        <div className="font-mono text-slate-500 border-t border-slate-800 pt-1" style={{ fontSize: '9px' }}>
          w[{l}][{j}][{k}] current value: {currentW >= 0 ? '+' : ''}{currentW.toFixed(5)}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// COMPONENT: ActivationExplorer
//
// Plots the activation function for a selected layer/neuron using actual z
// values from a real forward pass on the selected XOR input.
// Shows: f(z) curve, f′(z) derivative curve, tangent line at current z,
// vertical marker at current z, and numeric readout.
// =============================================================================
function ActivationExplorer({ network, layerSizes, hiddenActivationTypes, lastForwardData }) {
  const [selLayer,  setSelLayer]  = useState(1);
  const [selNeuron, setSelNeuron] = useState(0);
  const [selSample, setSelSample] = useState(0);
  const [showDeriv, setShowDeriv] = useState(true);

  if (!network) {
    return (
      <div className="text-slate-500 text-xs italic text-center py-6 px-2">
        Initialize the network first.
      </div>
    );
  }

  const numLayers = layerSizes.length;   // includes input layer
  const safeLayer = Math.max(1, Math.min(selLayer, numLayers - 1));
  const safeN     = Math.min(selNeuron, layerSizes[safeLayer] - 1);
  const isOut     = safeLayer === numLayers - 1;
  const actType   = isOut ? 'sigmoid' : (hiddenActivationTypes[safeLayer - 1] || 'relu');
  const actColor  = ACTIVATIONS[actType].color;
  const actLabel  = ACTIVATIONS[actType].label;

  // Compute actual z and activation values for the selected sample
  const sample = XOR_DATA[selSample];
  const { activations: sActs, preActivations: sPre } = forwardPass(
    sample.input, network.weights, network.biases, hiddenActivationTypes
  );
  const zVal  = sPre[safeLayer]?.[safeN] ?? null;
  const aVal  = sActs[safeLayer]?.[safeN] ?? null;
  const dAdZ  = zVal !== null ? ACTIVATIONS[actType].derivative(zVal) : null;

  const dead      = actType === 'relu' && zVal !== null && zVal <= 0;
  const saturated = actType !== 'relu' && dAdZ !== null && Math.abs(dAdZ) < 0.05;

  // Curve data for Recharts (tangent only rendered within ±1.5 of current z)
  const curveData = computeActivationCurve(actType, -4, 4, 80, zVal);
  const yDomain   = actType === 'relu' ? [-0.3, 1.3] : [-1.3, 1.3];
  const f4 = v => (v !== null && isFinite(v)) ? v.toFixed(4) : '—';

  return (
    <div className="space-y-2.5 text-xs">
      {/* ── Selectors ───────────────────────────────────────────────────────── */}
      <div className="space-y-1.5">
        <div className="flex gap-1.5">
          <div className="flex-1">
            <div className="text-slate-600 mb-0.5" style={{ fontSize: '8px' }}>Layer</div>
            <select value={safeLayer} onChange={e => setSelLayer(+e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-200 text-xs">
              {Array.from({ length: numLayers - 1 }, (_, i) => i + 1).map(li => (
                <option key={li} value={li}>
                  {li === numLayers - 1 ? `Layer ${li} (output)` : `Layer ${li} (hidden)`}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <div className="text-slate-600 mb-0.5" style={{ fontSize: '8px' }}>Neuron</div>
            <select value={safeN} onChange={e => setSelNeuron(+e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-200 text-xs">
              {Array.from({ length: layerSizes[safeLayer] }, (_, n) => (
                <option key={n} value={n}>n{n}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-slate-600" style={{ fontSize: '8px' }}>Input:</span>
          {XOR_DATA.map((s, i) => (
            <button key={i} onClick={() => setSelSample(i)}
              className={`px-1.5 py-0.5 rounded font-mono border transition-colors ${
                selSample === i
                  ? 'bg-blue-900/50 text-blue-300 border-blue-700'
                  : 'bg-slate-800 text-slate-500 border-slate-700 hover:text-slate-300'
              }`} style={{ fontSize: '9px' }}>
              [{s.input}]
            </button>
          ))}
          <button onClick={() => setShowDeriv(v => !v)}
            className={`ml-auto px-1.5 py-0.5 rounded border transition-colors ${
              showDeriv
                ? 'bg-slate-700 text-slate-300 border-slate-600'
                : 'bg-slate-800 text-slate-500 border-slate-700 hover:text-slate-300'
            }`} style={{ fontSize: '9px' }}>
            show f′(z)
          </button>
        </div>
      </div>

      {/* ── Stats readout ────────────────────────────────────────────────────── */}
      <div className="bg-slate-800/50 border border-slate-700 rounded p-2">
        <div className="flex items-center justify-between mb-1">
          <span className="font-semibold text-sm" style={{ color: actColor }}>{actLabel}</span>
          <span className="text-slate-500 font-mono" style={{ fontSize: '9px' }}>
            layer {safeLayer}, n{safeN}
          </span>
        </div>
        {zVal !== null && (
          <div className="grid grid-cols-3 gap-x-2 font-mono" style={{ fontSize: '9px' }}>
            <div><span className="text-slate-500">z =</span> <span className="text-blue-300">{f4(zVal)}</span></div>
            <div><span className="text-slate-500">f(z) =</span> <span style={{ color: actColor }}>{f4(aVal)}</span></div>
            <div>
              <span className="text-slate-500">f′(z) =</span>
              <span className={`ml-0.5 ${(dead || saturated) ? 'text-red-400' : 'text-slate-200'}`}>
                {f4(dAdZ)}
              </span>
            </div>
          </div>
        )}
        {dead && (
          <div className="text-red-400 mt-1 leading-tight" style={{ fontSize: '9px' }}>
            ⚠ Dead ReLU: z≤0, output=0, gradient=0. This neuron contributes nothing to the
            forward pass or to learning for this input.
          </div>
        )}
        {!dead && saturated && (
          <div className="text-amber-400 mt-1 leading-tight" style={{ fontSize: '9px' }}>
            ⚠ Saturated {actLabel}: f′≈0. Gradient barely flows through this neuron —
            learning slows in this region.
          </div>
        )}
      </div>

      {/* ── Recharts plot ────────────────────────────────────────────────────── */}
      <div style={{ height: '170px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={curveData} margin={{ top: 8, right: 8, left: -22, bottom: 2 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="z" type="number" domain={[-4, 4]} tickCount={5}
              tick={{ fill: '#475569', fontSize: 8 }} />
            <YAxis domain={yDomain} tick={{ fill: '#475569', fontSize: 8 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', fontSize: 9, padding: '3px 6px' }}
              labelFormatter={v => `z = ${(+v).toFixed(3)}`}
              formatter={(v, name) => [v !== null ? (+v).toFixed(4) : '—', name]}
            />

            {/* f(z) curve */}
            <Line type="monotone" dataKey="a" stroke={actColor} strokeWidth={2}
              dot={false} isAnimationActive={false} name={`${actLabel}(z)`} />

            {/* f′(z) curve */}
            {showDeriv && (
              <Line type="monotone" dataKey="dAdZ" stroke="#64748b" strokeWidth={1.5}
                strokeDasharray="4 2" dot={false} isAnimationActive={false} name={`${actLabel}′(z)`} />
            )}

            {/* Tangent line segment (only rendered where non-null, ±1.5 of current z) */}
            {zVal !== null && (
              <Line type="linear" dataKey="tangent" stroke="#f59e0b" strokeWidth={1.5}
                strokeDasharray="3 2" dot={false} isAnimationActive={false}
                connectNulls={false} name="tangent" />
            )}

            {/* Vertical reference line at current z */}
            {zVal !== null && (
              <ReferenceLine x={zVal} stroke="#f59e0b" strokeWidth={1} strokeOpacity={0.7}
                strokeDasharray="2 2" />
            )}

            {/* Dot at f(z) */}
            {zVal !== null && aVal !== null && (
              <ReferenceDot x={zVal} y={aVal} r={4} fill={actColor} stroke="#0f172a" strokeWidth={1} />
            )}

            {/* Dot at f′(z) */}
            {showDeriv && zVal !== null && dAdZ !== null && (
              <ReferenceDot x={zVal} y={dAdZ} r={3} fill="#64748b" stroke="#0f172a" strokeWidth={1} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Amber z-label below chart ─────────────────────────────────────────── */}
      {zVal !== null && (
        <div className="text-center font-mono text-amber-500/80" style={{ fontSize: '8px' }}>
          current z = {f4(zVal)}
          {' · '}f(z) = {f4(aVal)}
          {showDeriv && ` · f′(z) = ${f4(dAdZ)}`}
          {' '}(most recent forward pass, sample [{sample.input}])
        </div>
      )}

      {/* ── Educational notes ────────────────────────────────────────────────── */}
      <div className="space-y-1.5 text-slate-500 leading-tight border-t border-slate-800 pt-2"
           style={{ fontSize: '9px' }}>
        {actType === 'relu' && (
          <>
            <p>
              <span className="text-blue-400 font-semibold">ReLU(z)</span> = max(0, z).
              Derivative is 1 for z&gt;0 and 0 for z&lt;0. At z=0 ReLU is nondifferentiable;
              this simulator and PyTorch use the convention f′(0)=0.
            </p>
            <p>
              <span className="text-red-400/80">Dead ReLU</span>: if z stays below 0 for all
              training samples, the neuron outputs 0 across the entire dataset and its incoming
              weights receive no gradient — recovery through gradient descent alone is impossible.
              Reset weights or switch activation to fix it.
            </p>
          </>
        )}
        {actType === 'tanh' && (
          <>
            <p>
              <span className="text-violet-400 font-semibold">Tanh(z)</span> ∈ (−1, 1), zero-centered.
              Derivative peaks at 1.0 (z=0) and decays toward 0 in both tails.
            </p>
            <p>
              <span className="text-amber-400/80">Saturation</span>: for |z|≳2, tanh′→0.
              Gradient slows but does not fully stop. Tanh saturates more gracefully than sigmoid
              and stays zero-centered — often preferable for hidden layers.
            </p>
          </>
        )}
        {actType === 'sigmoid' && (
          <>
            <p>
              <span className="text-pink-400 font-semibold">Sigmoid(z)</span> ∈ (0, 1).
              Derivative peaks at 0.25 (at z=0). Output is not zero-centered.
            </p>
            <p>
              <span className="text-amber-400/80">Saturation + vanishing gradients</span>: for |z|≳3,
              σ′→0. In deep nets, gradients multiply through each layer — a chain of near-zero
              derivatives shrinks the signal exponentially, making early layers very slow to train.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// COMPONENT: CalcPanel  (Phase 3 Calculus Panel)
// =============================================================================
function CalcPanel({ network, layerSizes, hiddenActivationTypes, lastGradients, lastForwardData }) {
  const [section, setSection] = useState('chain');

  return (
    <div className="space-y-2.5">
      {/* Inner tab strip */}
      <div className="flex border border-slate-700 rounded overflow-hidden">
        {[
          { id: 'chain', label: '∂w Trace' },
          { id: 'activ', label: 'f(z) Plot' },
        ].map(({ id, label }) => (
          <button key={id} onClick={() => setSection(id)}
            className={`flex-1 py-1 text-xs font-medium transition-colors ${
              section === id
                ? 'bg-slate-700 text-white'
                : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
            }`}>
            {label}
          </button>
        ))}
      </div>

      {section === 'chain' && (
        <ChainRuleTracer
          network={network}
          layerSizes={layerSizes}
          hiddenActivationTypes={hiddenActivationTypes}
          lastGradients={lastGradients}
        />
      )}
      {section === 'activ' && (
        <ActivationExplorer
          network={network}
          layerSizes={layerSizes}
          hiddenActivationTypes={hiddenActivationTypes}
          lastForwardData={lastForwardData}
        />
      )}
    </div>
  );
}

// =============================================================================
// COMPONENT: WeightsInspector
//
// Shows every layer's weight matrix and bias vector with amber/blue color coding
// (positive = amber, negative = blue, brightness = relative magnitude).
// Also provides JSON export of all parameters (copy + download).
// =============================================================================
function WeightsInspector({ network, layerSizes, hiddenActivationTypes, epoch, trainingStatus, latestLoss, xorResults }) {
  const [llmExpanded, setLlmExpanded] = useState(false);
  const [ptExpanded,  setPtExpanded]  = useState(false);
  const [copyDone,    setCopyDone]    = useState(false);

  if (!network) {
    return (
      <div className="text-slate-600 text-xs text-center py-4 italic">
        Initialize network to inspect parameters.
      </div>
    );
  }

  const { weights, biases } = network;
  const layerMaxMag = weights.map(W =>
    Math.max(...W.map(row => Math.max(...row.map(Math.abs))))
  );

  const buildExport = () => {
    // xor_solved: same logic as XorVerifyPanel — all 4 points predicted correctly
    const xorSolved = xorResults ? xorResults.every(r => r.correct) : false;

    // Derive which convergence criterion(a) fired, consistent with checkConvergence()
    const convergenceReasons = [];
    if (trainingStatus === 'converged') {
      if (latestLoss !== null && latestLoss < CONVERGENCE_LOSS_THRESHOLD) {
        convergenceReasons.push('loss_threshold');
      }
      if (xorResults && xorResults.every(r => r.correct && r.confidence > CONVERGENCE_CONFIDENCE)) {
        convergenceReasons.push('xor_verified');
      }
    }

    // Per-point XOR verification detail
    const xorVerification = xorResults
      ? xorResults.map(r => ({
          input:                 r.input,
          expected:              r.label,
          predicted_probability: r.rawOutput,
          predicted_class:       r.predictedClass,
          confidence:            r.confidence,
          correct:               r.correct,
        }))
      : null;

    const layers = weights.map((W, l) => {
      const inSize  = layerSizes[l];
      const outSize = layerSizes[l + 1];
      const isOutput = l === weights.length - 1;
      return {
        name:              isOutput ? 'output' : `hidden_${l + 1}`,
        index:             l,
        activation:        isOutput ? 'sigmoid' : (hiddenActivationTypes[l] || 'relu'),
        weight_shape:      [outSize, inSize],
        weight_convention: 'weight[out_feature][in_feature] — same as nn.Linear.weight; no transposition needed',
        bias_shape:        [outSize],
        weight:            W,
        bias:              biases[l],
        parameter_count:   outSize * inSize + outSize,
      };
    });

    const trainingObj = { epoch, loss: latestLoss, status: trainingStatus, xor_solved: xorSolved };
    if (convergenceReasons.length === 1) trainingObj.convergence_reason = convergenceReasons[0];
    if (convergenceReasons.length  > 1) trainingObj.convergence_reason = convergenceReasons;
    // Explicitly note the "high-confidence, loss still above threshold" case so readers aren't confused
    if (
      trainingStatus === 'converged' &&
      latestLoss !== null &&
      latestLoss >= CONVERGENCE_LOSS_THRESHOLD &&
      convergenceReasons.includes('xor_verified')
    ) {
      trainingObj.note =
        `Converged via XOR confidence criterion (all 4 points correct with >${(CONVERGENCE_CONFIDENCE*100).toFixed(0)}% confidence ` +
        `for ${CONVERGENCE_CONSECUTIVE_EPOCHS} consecutive epochs). Loss ${latestLoss.toFixed(5)} is above the ` +
        `${CONVERGENCE_LOSS_THRESHOLD} threshold — that is expected for this convergence path.`;
    }
    if (xorVerification) trainingObj.xor_verification = xorVerification;

    return {
      exported_at:      new Date().toISOString(),
      source:           'Neural Net Playground — neural-viz',
      architecture: {
        layers:             layerSizes,
        hidden_activations: hiddenActivationTypes,
        output_activation:  'sigmoid',
      },
      training: trainingObj,
      layers,
      total_parameters: layers.reduce((s, l) => s + l.parameter_count, 0),
    };
  };

  const handleCopyJSON = () => {
    const json = JSON.stringify(buildExport(), null, 2);
    const fallback = () => {
      const ta = document.createElement('textarea');
      ta.value = json; ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta); ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(json).catch(fallback);
    } else { fallback(); }
    setCopyDone(true);
    setTimeout(() => setCopyDone(false), 2000);
  };

  const handleDownloadJSON = () => {
    const blob = new Blob([JSON.stringify(buildExport(), null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'neural-viz-params.json'; a.click();
    URL.revokeObjectURL(url);
  };

  const ptSnippet =
`import json, torch, torch.nn as nn

with open('neural-viz-params.json') as f:
    p = json.load(f)

linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
for i, ldata in enumerate(p['layers']):
    with torch.no_grad():
        linears[i].weight.copy_(
            torch.tensor(ldata['weight'], dtype=torch.float32))
        linears[i].bias.copy_(
            torch.tensor(ldata['bias'], dtype=torch.float32))
print("Weights loaded from neural-viz-params.json")`;

  return (
    <div className="space-y-3">
      {/* Educational callout */}
      <div className="bg-blue-900/20 border border-blue-800/40 rounded p-2 text-xs text-blue-200/80 leading-relaxed">
        <p className="font-semibold text-blue-300 mb-1">What these numbers mean</p>
        <p>Each <span className="text-amber-300">weight</span> is a learned multiplier on an incoming signal. Large positive → amplifies toward class 1; large negative → suppresses it. <span className="text-slate-400">Biases shift the activation threshold independently of inputs.</span></p>
        <p className="mt-1"><span className="text-amber-400 font-semibold">Amber = positive</span> · <span className="text-blue-400 font-semibold">blue = negative</span> · brightness = magnitude relative to layer max.</p>
      </div>

      {/* Per-layer weight matrices + bias vectors */}
      {weights.map((W, l) => {
        const inSize   = layerSizes[l];
        const outSize  = layerSizes[l + 1];
        const isOutput = l === weights.length - 1;
        const actName  = isOutput ? 'Sigmoid' : (ACTIVATIONS[hiddenActivationTypes[l]]?.label || hiddenActivationTypes[l]);
        const maxMag   = layerMaxMag[l];

        return (
          <div key={l} className="bg-slate-800/50 border border-slate-700 rounded p-2">
            <div className="flex items-baseline justify-between mb-1.5">
              <span className="text-xs font-semibold text-slate-300">
                {isOutput ? 'Output layer' : `Hidden layer ${l + 1}`}
              </span>
              <span className="text-xs font-mono text-slate-500">
                {inSize}→{outSize} · <span className="text-slate-400">{actName}</span>
              </span>
            </div>

            {/* Weight matrix */}
            <div className="text-slate-500 mb-1 font-mono" style={{ fontSize: '9px' }}>
              W [{outSize}×{inSize}]<span className="text-slate-700 ml-1">rows=out neuron, cols=in neuron</span>
            </div>
            {/* Storage/PyTorch orientation note */}
            <div className="bg-slate-900/60 border border-slate-700/50 rounded px-2 py-1 mb-1.5 text-slate-500 leading-snug" style={{ fontSize: '8.5px' }}>
              Shape: <span className="text-slate-400 font-mono">[out_features, in_features]</span> — same as <span className="text-emerald-500/80 font-mono">nn.Linear.weight</span>.
              <span className="block mt-0.5 text-slate-600">W[j][k] = weight from input k → output j. Loads into PyTorch directly; no transposition needed.</span>
            </div>
            <div className="overflow-x-auto">
              <table className="border-collapse">
                <thead>
                  <tr>
                    <td className="pr-1 text-slate-700" style={{ fontSize: '8px', fontFamily: 'monospace' }}>↓out\in→</td>
                    {Array.from({ length: inSize }, (_, j) => (
                      <td key={j} className="text-center text-slate-600 pb-0.5 px-0.5"
                          style={{ fontSize: '8px', fontFamily: 'monospace', minWidth: '38px' }}>
                        x{j}
                      </td>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {W.map((row, i) => (
                    <tr key={i}>
                      <td className="pr-1 text-slate-600 text-right"
                          style={{ fontSize: '8px', fontFamily: 'monospace' }}>n{i}</td>
                      {row.map((v, j) => (
                        <td key={j} className="px-0.5 py-px">
                          <div className="text-center rounded text-slate-100 leading-tight"
                               style={{
                                 background: weightColor(v, maxMag),
                                 padding: '2px 3px',
                                 fontSize: '9px',
                                 fontFamily: 'monospace',
                                 minWidth: '36px',
                               }}>
                            {v >= 0 ? '+' : ''}{v.toFixed(3)}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Bias vector */}
            <div className="mt-1.5">
              <div className="text-slate-500 mb-0.5 font-mono" style={{ fontSize: '9px' }}>
                b [{outSize}]
              </div>
              <div className="flex flex-wrap gap-1">
                {biases[l].map((v, i) => (
                  <div key={i} className="text-center rounded text-slate-100 leading-tight"
                       style={{
                         background: weightColor(v, maxMag),
                         padding: '2px 4px',
                         fontSize: '9px',
                         fontFamily: 'monospace',
                         minWidth: '36px',
                       }}>
                    {v >= 0 ? '+' : ''}{v.toFixed(3)}
                  </div>
                ))}
              </div>
            </div>

            <div className="text-slate-700 mt-1 font-mono" style={{ fontSize: '8px' }}>
              max |w| = {maxMag.toFixed(4)}
            </div>
          </div>
        );
      })}

      {/* LLM analogy (collapsible) */}
      <div className="border border-slate-700 rounded overflow-hidden">
        <button
          onClick={() => setLlmExpanded(v => !v)}
          className="w-full flex items-center justify-between px-2.5 py-1.5 text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 transition-colors">
          <span className="font-semibold">LLM analogy</span>
          <span className="font-mono text-slate-600">{llmExpanded ? '▲' : '▼'}</span>
        </button>
        {llmExpanded && (
          <div className="bg-slate-800/40 border-t border-slate-700 px-2.5 py-2 text-xs text-slate-300 leading-relaxed space-y-1.5">
            <p>Frontier LLMs may contain <span className="text-amber-300 font-semibold">hundreds of billions to trillions of parameters</span> — the same broad kind of learned numbers you see here, just at vastly larger scale.</p>
            <p>In a transformer, learned <span className="text-blue-300">projection matrices</span> help compute attention: each token is transformed into query, key, and value vectors, and attention scores are computed dynamically from those vectors. That is much more complex than this XOR net, but it still relies on learned weight matrices.</p>
            <p>When you download a model checkpoint (e.g. a <code className="text-emerald-400 bg-slate-900/60 rounded px-0.5">.safetensors</code> file), you're downloading trained tensors: embeddings, attention weights, MLP/feed-forward weights, normalization parameters, and more. The architecture is fixed; the learned values are what training produces.</p>
            <p className="text-slate-400">
              Your net: {network.weights.reduce((s, W) => s + W.length * W[0].length + W.length, 0)} params.
              Frontier LLMs: hundreds of billions to trillions. Same broad idea — learned tensors — scaled up enormously and arranged in a much more complex architecture.
            </p>
            <p className="text-slate-400 border-t border-slate-700/60 pt-1.5 mt-0.5 italic">
              This MLP is not architecturally equivalent to an LLM. The analogy is about learned tensors/parameters, not model structure.
            </p>
          </div>
        )}
      </div>

      {/* Parameter JSON export */}
      <div>
        <div className="text-xs text-slate-500 mb-1.5">Export Parameters (JSON)</div>
        <div className="flex gap-1.5">
          <button onClick={handleCopyJSON}
            className={`flex-1 py-1 rounded text-xs border transition-colors ${
              copyDone
                ? 'bg-emerald-900/40 text-emerald-400 border-emerald-700/50'
                : 'bg-slate-700 hover:bg-slate-600 text-slate-300 border-slate-600'
            }`}>
            {copyDone ? '✓ Copied!' : '📋 Copy JSON'}
          </button>
          <button onClick={handleDownloadJSON}
            className="flex-1 py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 border border-slate-600 transition-colors">
            ⬇ Download JSON
          </button>
        </div>
        <div className="text-slate-700 mt-1 font-mono" style={{ fontSize: '9px' }}>
          neural-viz-params.json · weights, biases, arch &amp; training state
        </div>
      </div>

      {/* PyTorch weight-loading snippet (collapsible) */}
      <div className="border border-slate-700 rounded overflow-hidden">
        <button
          onClick={() => setPtExpanded(v => !v)}
          className="w-full flex items-center justify-between px-2.5 py-1.5 text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 transition-colors">
          <span className="font-mono">PyTorch: load weights</span>
          <span className="font-mono text-slate-600">{ptExpanded ? '▲' : '▼'}</span>
        </button>
        {ptExpanded && (
          <div className="bg-gray-950 border-t border-slate-800 overflow-y-auto" style={{ maxHeight: '180px' }}>
            <pre className="text-xs font-mono text-slate-300 leading-relaxed whitespace-pre-wrap p-2.5">
              {ptSnippet}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// COMPONENT: TrainingStatusBar
// =============================================================================
function TrainingStatusBar({ status, epoch, loss, bestLoss, epochsSinceImprove, stopReason, maxEpochs }) {
  const statusConfig = {
    idle:      { label: 'Not Started',        color: 'text-slate-500',  bg: 'bg-slate-800/40',  dot: 'bg-slate-600' },
    training:  { label: 'Training',           color: 'text-blue-400',   bg: 'bg-blue-900/20',   dot: 'bg-blue-400 animate-pulse' },
    stepping:  { label: 'Explained Step',     color: 'text-violet-400', bg: 'bg-violet-900/20', dot: 'bg-violet-400 animate-pulse' },
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
function ConceptCallout({ type, onDismiss, trainingStatus, hiddenActivationTypes }) {
  // 'stepping' means the explained-step animation is mid-backward-pass — backprop
  // is actively animating, so treat it like 'training' for wording purposes.
  const stopped = !['training', 'stepping'].includes(trainingStatus);
  const hasReLU = hiddenActivationTypes?.some(t => t === 'relu');

  const plateauBody = (
    <div className="space-y-2">
      {hasReLU && (
        <p className="text-orange-200/80 leading-relaxed">
          <span className="font-semibold">Dead ReLU neurons:</span>{' '}
          ReLU outputs 0 whenever its pre-activation is negative. If enough
          weights go negative during training, a neuron fires 0 for every input —
          its gradient is always 0 and it never recovers. This is a common cause
          of XOR plateaus with ReLU networks.
        </p>
      )}
      <p className="text-slate-400">
        Loss has not improved by &gt;{MIN_IMPROVEMENT} in {PLATEAU_PATIENCE} epochs
        (still above {PLATEAU_MIN_LOSS}). Try in this order:
      </p>
      <ol className="space-y-1">
        {[
          ['Reset weights', 'Reinitialize with a new random seed — free, no architecture change. Different init may avoid dead neurons entirely.'],
          ['Switch to Tanh', 'Tanh gradient is nonzero everywhere: σ\'(z) = 1−tanh²(z) > 0 for all z. Neurons can\'t go dead. Best first fix for ReLU plateaus.'],
          ['Adjust learning rate', 'Too large → oscillates and overshoots; too small → never escapes flat regions. Try 0.01–0.05, or use Adam which adapts LR per weight.'],
          ['Add capacity last', 'More neurons or layers won\'t help if gradients are already near zero. Exhaust the above options first.'],
        ].map(([title, desc], i) => (
          <li key={i} className="flex gap-1.5">
            <span className="text-orange-400 font-bold shrink-0">{i + 1}.</span>
            <span className="text-slate-300">
              <span className="font-semibold text-slate-200">{title}</span>
              {' — '}{desc}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );

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
      body: plateauBody,
      pytorch: '# Adam adapts the learning rate per weight — more robust than SGD:\noptimizer = torch.optim.Adam(model.parameters(), lr=0.01)',
    },
    converged: {
      title: 'XOR Solved ✓', color: 'border-emerald-500', icon: '✓',
      body: 'The network correctly classifies all 4 XOR points with high confidence. Training stopped automatically. You can click anywhere on the boundary canvas to test inference.',
      pytorch: `model.eval()\nwith torch.no_grad():\n    pred = model(test_input)`,
    },
    vanishingGradient: {
      title: 'Vanishing Gradient Detected', color: 'border-red-500', icon: '⚠',
      body: 'Input-layer gradients are <1% of the network-wide maximum. Sigmoid derivatives max at 0.25 — stacked sigmoid layers drive gradients exponentially toward zero. Try ReLU or Tanh for hidden layers.',
      pytorch: '# Replace nn.Sigmoid() with nn.ReLU() in hidden layers',
    },
    inferencePoint: {
      title: 'Click-to-Predict', color: 'border-emerald-500', icon: '◎',
      body: 'The network ran a real forward pass on your clicked point using the current weights. The animation shows activations propagating layer-by-layer. No gradients are computed.',
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
      {/* body can be a string or JSX — both render fine inside a div */}
      <div className="text-slate-300 mt-1 text-xs leading-relaxed">{c.body}</div>
      {c.pytorch && (
        <pre className="mt-2 text-xs bg-black/40 rounded p-2 text-emerald-400 overflow-x-auto">{c.pytorch}</pre>
      )}
    </div>
  );
}

// =============================================================================
// COMPONENT: PyTorchPanel
//
// Replaces the old PyTorchSidebar. Three sections:
//   1. Compact mapping summary (architecture, activations, loss, optimizer,
//      params, visual-concept → PyTorch API table)
//   2. Export buttons: copy runnable .py script, download .ipynb notebook
//   3. Collapsible full PyTorch code (collapsed by default)
//
// Weights are NOT exported — the generated code reinitializes randomly.
// A note in the UI and the exported files makes this explicit.
// =============================================================================
function PyTorchPanel({ layerSizes, hiddenActivationTypes, learningRate }) {
  const [codeExpanded, setCodeExpanded] = useState(false);
  const [copyDone,     setCopyDone]     = useState(false);

  const paramCount = layerSizes.slice(0, -1).reduce(
    (s, n, i) => s + n * layerSizes[i + 1] + layerSizes[i + 1], 0
  );
  const actLabels = hiddenActivationTypes.map(t => ACTIVATIONS[t]?.label || t).join(', ');

  const handleCopyScript = () => {
    const script = generateFullScript(layerSizes, hiddenActivationTypes, learningRate);
    const write = () => {
      if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(script).catch(fallback);
      } else { fallback(); }
    };
    const fallback = () => {
      const ta = document.createElement('textarea');
      ta.value = script; ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta); ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    };
    write();
    setCopyDone(true);
    setTimeout(() => setCopyDone(false), 2000);
  };

  const handleExportNotebook = () => {
    const nb   = generateNotebook(layerSizes, hiddenActivationTypes, learningRate);
    const blob = new Blob([JSON.stringify(nb, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'neural-viz-xor.ipynb'; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-shrink-0 border-b border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between px-3 pt-2.5 pb-1">
        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">PyTorch Mapping</h2>
        <span className="text-xs text-slate-600">explanatory only</span>
      </div>

      {/* Compact summary */}
      <div className="px-3 pb-1.5 text-xs font-mono space-y-0.5">
        <div>
          <span className="text-slate-600">arch  </span>
          <span className="text-slate-200">{layerSizes.slice(0,-1).join('→')}→1</span>
          <span className="text-slate-600"> · </span>
          <span className="text-amber-400">{paramCount}p</span>
        </div>
        <div>
          <span className="text-slate-600">act   </span>
          <span className="text-slate-300">{actLabels}</span>
          <span className="text-slate-600">, Sigmoid(out)</span>
        </div>
        <div>
          <span className="text-slate-600">opt   </span>
          <span className="text-slate-300">SGD lr={learningRate.toFixed(3)}</span>
          <span className="text-slate-600"> · </span>
          <span className="text-slate-300">BCELoss</span>
        </div>

        {/* Visual concept → PyTorch API mapping */}
        <div className="border-t border-slate-800 pt-1.5 mt-1">
          <div className="text-slate-600 mb-1 text-xs">Visual → PyTorch</div>
          {[
            ['nodes/edges',  'nn.Linear(in, out)'],
            ['forward pass', 'out = model(X)'],
            ['loss',         'criterion(out, y)'],
            ['backprop',     'loss.backward()'],
            ['W ← W−lr·∂W', 'optimizer.step()'],
            ['inference',    'model.eval() + no_grad()'],
          ].map(([vis, pt]) => (
            <div key={vis} className="flex items-baseline gap-1 leading-tight py-px">
              <span className="text-slate-500 w-24 shrink-0 truncate">{vis}</span>
              <span className="text-slate-700 shrink-0">→</span>
              <span className="text-emerald-400/80 truncate">{pt}</span>
            </div>
          ))}
        </div>

        {/* Weights-not-exported notice */}
        <div className="border-t border-slate-800 pt-1 mt-0.5 text-slate-600 leading-tight">
          Exported code reinitializes weights randomly — architecture matches, values don't.
          To export actual trained values, use the <span className="text-amber-500/80">Weights</span> tab.
        </div>
      </div>

      {/* Export buttons */}
      <div className="flex gap-1.5 px-3 pb-2.5">
        <button onClick={handleCopyScript}
          className={`flex-1 py-1 rounded text-xs border transition-colors ${
            copyDone
              ? 'bg-emerald-900/40 text-emerald-400 border-emerald-700/50'
              : 'bg-slate-700 hover:bg-slate-600 text-slate-300 border-slate-600'
          }`}>
          {copyDone ? '✓ Copied!' : '📋 Copy Script'}
        </button>
        <button onClick={handleExportNotebook}
          className="flex-1 py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 border border-slate-600 transition-colors">
          📓 Notebook
        </button>
      </div>

      {/* Collapsible full code */}
      <button
        onClick={() => setCodeExpanded(v => !v)}
        className="w-full flex items-center justify-between px-3 py-1.5 text-xs text-slate-500 hover:text-slate-300 hover:bg-slate-800/40 border-t border-slate-800 transition-colors">
        <span className="font-mono">Full PyTorch code</span>
        <span className="font-mono text-slate-600">{codeExpanded ? '▲' : '▼'}</span>
      </button>
      {codeExpanded && (
        <div className="bg-gray-950 border-t border-slate-800 overflow-y-auto"
             style={{ maxHeight: '220px' }}>
          <pre className="text-xs font-mono text-slate-300 leading-relaxed whitespace-pre-wrap p-3">
            {generatePyTorchCode(layerSizes, hiddenActivationTypes)}
          </pre>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN APP COMPONENT
// =============================================================================
export default function App() {
  // ── Architecture ──────────────────────────────────────────────────────────
  const [numHiddenLayers, setNumHiddenLayers] = useState(2);
  const [neuronsPerLayer, setNeuronsPerLayer]  = useState([4, 4]);
  const [activationTypes, setActivationTypes]  = useState(['relu', 'relu']);

  // ── Network weights ────────────────────────────────────────────────────────
  const [network, setNetwork] = useState(null);

  // ── Training state ─────────────────────────────────────────────────────────
  const [isTraining,             setIsTraining]             = useState(false);
  const [trainingStatus,         setTrainingStatus]         = useState('idle');
  const [stopReason,             setStopReason]             = useState('');
  const [epoch,                  setEpoch]                  = useState(0);
  const [lossHistory,            setLossHistory]            = useState([]);
  const [learningRate,           setLearningRate]           = useState(0.1);
  const [bestLoss,               setBestLoss]               = useState(Infinity);
  const [epochsSinceImprovement, setEpochsSinceImprovement] = useState(0);
  const [maxEpochs,              setMaxEpochs]              = useState(10000);
  const [lastGradients,          setLastGradients]          = useState(null);
  const [lastForwardData,        setLastForwardData]        = useState(null);
  const [xorResults,             setXorResults]             = useState(null);

  // ── Phase 2: Explained step animation state ────────────────────────────────
  // stepModeStage tracks which stage of the 4-stage explained epoch we're in:
  //   'idle'     — no animation running
  //   'forward'  — Stage 1: forward pass animating left→right
  //   'loss'     — Stage 2: loss value being shown
  //   'backward' — Stage 3: backprop gradients animating right→left
  //   'update'   — Stage 4: weight update being applied
  const [stepModeStage,         setStepModeStage]         = useState('idle');
  const [stepModeLoss,          setStepModeLoss]          = useState(null);  // pre-update loss for Stage 2 display
  const [stepModeResult,        setStepModeResult]        = useState(null);  // pre-computed epoch data
  const [stepModeLayerAnimDone, setStepModeLayerAnimDone] = useState(false); // true once per-layer anim finishes
  const [stepModeAutoPlay,      setStepModeAutoPlay]      = useState(false); // auto-advance stages
  const [stepModeAutoSpeed,     setStepModeAutoSpeed]     = useState(4);     // seconds between auto-advances

  // ── Phase 2: View options ──────────────────────────────────────────────────
  const [showConfidence,     setShowConfidence]     = useState(false); // confidence vs class boundary
  const [showGradientLabels, setShowGradientLabels] = useState(true);  // numeric ∂L/∂W on edges
  const [rightPanelTab,      setRightPanelTab]      = useState('audit'); // 'audit' | 'gradcheck' | 'weights' | 'calc'

  // ── Animation ─────────────────────────────────────────────────────────────
  const [animatingLayer,     setAnimatingLayer]     = useState(-1);
  const [forwardPassDisplay, setForwardPassDisplay] = useState(null);

  // ── Inference ─────────────────────────────────────────────────────────────
  const [inferencePoint, setInferencePoint] = useState(null);

  // ── Concept callouts ───────────────────────────────────────────────────────
  const [callouts,          setCallouts]         = useState(new Set());
  const [dismissedCallouts, setDismissedCallouts] = useState(new Set());

  // ── Refs (avoid stale closures in RAF loop and async animations) ───────────
  const trainingRef           = useRef(false);
  const networkRef            = useRef(null);
  const epochRef              = useRef(0);
  const lossHistoryRef        = useRef([]);
  const bestLossRef           = useRef(Infinity);
  const epochsSinceImprovRef  = useRef(0);
  const consecutiveCorrectRef = useRef(0);
  const maxEpochsRef          = useRef(10000);
  const dismissedCalloutsRef  = useRef(new Set());
  const trainingLoopRef       = useRef(null);
  const isAnimatingRef        = useRef(false); // prevents concurrent animation runs
  const stepModeAutoTimerRef  = useRef(null);  // setTimeout ID for explained-step auto-play
  const stepModeAppliedRef    = useRef(false); // prevents double weight-apply on revisit of update stage

  useEffect(() => { dismissedCalloutsRef.current = dismissedCallouts; }, [dismissedCallouts]);
  useEffect(() => { maxEpochsRef.current = maxEpochs; }, [maxEpochs]);

  const layerSizes = [2, ...neuronsPerLayer.slice(0, numHiddenLayers), 1];

  // ── Network initialization ─────────────────────────────────────────────────
  const initializeNetwork = useCallback(() => {
    if (trainingLoopRef.current) cancelAnimationFrame(trainingLoopRef.current);
    trainingRef.current           = false;
    isAnimatingRef.current        = false;
    const net                     = initNetwork(layerSizes);
    networkRef.current            = net;
    epochRef.current              = 0;
    lossHistoryRef.current        = [];
    bestLossRef.current           = Infinity;
    epochsSinceImprovRef.current  = 0;
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
    setStepModeStage('idle');
    setStepModeLoss(null);
    setStepModeResult(null);
    setStepModeLayerAnimDone(false);
    setStepModeAutoPlay(false);
    stepModeAppliedRef.current = false;
    if (stepModeAutoTimerRef.current) {
      clearTimeout(stepModeAutoTimerRef.current);
      stepModeAutoTimerRef.current = null;
    }
  }, [layerSizes.join(',')]);

  useEffect(() => { initializeNetwork(); }, []);

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
  const runTrainingStep = useCallback(() => {
    if (!networkRef.current) return { shouldStop: false };

    if (epochRef.current >= maxEpochsRef.current) {
      const reason = `Reached maximum of ${maxEpochsRef.current} epochs`;
      setTrainingStatus('maxEpochs');
      setStopReason(reason);
      return { shouldStop: true };
    }

    const { weights, biases } = networkRef.current;
    const result = trainOneEpoch(weights, biases, activationTypes, learningRate);

    networkRef.current = { weights: result.weights, biases: result.biases };
    setNetwork({ weights: result.weights, biases: result.biases });

    epochRef.current += 1;
    setEpoch(epochRef.current);

    lossHistoryRef.current = [...lossHistoryRef.current,
      { epoch: epochRef.current, loss: result.loss }].slice(-200);
    setLossHistory([...lossHistoryRef.current]);

    setLastGradients({ dWeights: result.avgDW, dBiases: result.avgDB });
    setLastForwardData(result.allForwardData[0]);

    if (result.loss < bestLossRef.current - MIN_IMPROVEMENT) {
      bestLossRef.current          = result.loss;
      epochsSinceImprovRef.current = 0;
      setBestLoss(result.loss);
    } else {
      epochsSinceImprovRef.current += 1;
    }
    setEpochsSinceImprovement(epochsSinceImprovRef.current);

    const xorEval = evaluateXOR(result.weights, result.biases, activationTypes);
    setXorResults(xorEval);

    const allHighConf = xorEval.every(r => r.correct && r.confidence > CONVERGENCE_CONFIDENCE);
    if (allHighConf) consecutiveCorrectRef.current += 1;
    else             consecutiveCorrectRef.current  = 0;

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

    if (epochRef.current === 1 && !dismissedCalloutsRef.current.has('firstBackprop')) {
      setCallouts(prev => new Set([...prev, 'firstBackprop']));
    }

    // Vanishing gradient: input-layer max gradient is <1% of network-wide max.
    // Only trigger when loss is still high — if the model has converged and loss is
    // near zero, tiny gradients are expected and correct, not a problem.
    const firstLayerMax = Math.max(...result.avgDW[0].flat().map(Math.abs));
    const networkMax    = Math.max(...result.avgDW.flat(2).map(Math.abs));
    if (
      networkMax > 0 &&
      firstLayerMax / networkMax < 0.01 &&
      result.loss > PLATEAU_MIN_LOSS &&
      !dismissedCalloutsRef.current.has('vanishingGradient')
    ) {
      setCallouts(prev => new Set([...prev, 'vanishingGradient']));
    }

    return { shouldStop: false };
  }, [activationTypes, learningRate]);

  // ── RAF training loop ──────────────────────────────────────────────────────
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
          return;
        }
      }
      trainingLoopRef.current = requestAnimationFrame(loop);
    };
    trainingLoopRef.current = requestAnimationFrame(loop);
    return () => { if (trainingLoopRef.current) cancelAnimationFrame(trainingLoopRef.current); };
  }, [isTraining, runTrainingStep]);

  // ── Control handlers ────────────────────────────────────────────────────────
  const handleToggleTraining = () => {
    if (stepModeStage !== 'idle') return; // don't interrupt explained step
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
    if (isTraining || stepModeStage !== 'idle') return;
    const { shouldStop } = runTrainingStep();
    if (shouldStop) setIsTraining(false);
    else setTrainingStatus('paused');
  };

  const handleReset = () => { initializeNetwork(); };

  // ── Forward-pass animation (standalone) ───────────────────────────────────
  const runForwardPassAnimation = useCallback(async () => {
    if (!networkRef.current || isTraining || isAnimatingRef.current) return;
    isAnimatingRef.current = true;
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
    isAnimatingRef.current = false;
  }, [activationTypes, layerSizes.length, isTraining]);

  // ── Explained epoch — user-paced 4-stage mode ────────────────────────────
  // startExplainedEpoch pre-computes one training epoch and enters step mode.
  // The user navigates Forward → Loss → Backward → Update via Next/Prev/Done
  // buttons in the StepBanner. Auto-play can advance stages automatically.
  //
  //   Stage 1 (forward):  activations animate left→right
  //   Stage 2 (loss):     BCE loss shown; forward activations remain visible
  //   Stage 3 (backward): gradient magnitudes on edges, right→left highlight
  //   Stage 4 (update):   weights applied once (idempotent), epoch increments
  //
  // PyTorch equivalent:
  //   out = model(X)          → Stage 1
  //   loss = criterion(out,y) → Stage 2
  //   loss.backward()         → Stage 3
  //   optimizer.step()        → Stage 4
  const startExplainedEpoch = useCallback(() => {
    if (!networkRef.current || isTraining || isAnimatingRef.current) return;
    if (stepModeStage !== 'idle') return;
    const { weights, biases } = networkRef.current;
    const result = trainOneEpoch(weights, biases, activationTypes, learningRate);
    stepModeAppliedRef.current = false;
    setStepModeResult(result);
    setStepModeLoss(result.loss);
    setStepModeAutoPlay(false);
    setStepModeLayerAnimDone(false);
    setStepModeStage('forward');
    setTrainingStatus('stepping');
  }, [activationTypes, learningRate, isTraining, stepModeStage]);

  // Per-stage layer animation: fires on every stage transition.
  // Uses a cancellation token so navigating away stops the in-flight animation.
  useEffect(() => {
    if (stepModeStage === 'idle' || !stepModeResult) return;
    let cancelled = false;
    const delay = ms => new Promise(r => setTimeout(r, ms));

    setStepModeLayerAnimDone(false);
    setAnimatingLayer(-1);
    isAnimatingRef.current = true;

    const run = async () => {
      if (stepModeStage === 'forward') {
        const fwdData = stepModeResult.allForwardData[0];
        for (let l = 0; l < layerSizes.length; l++) {
          if (cancelled) return;
          setAnimatingLayer(l);
          setForwardPassDisplay({
            activations:    fwdData.activations.slice(0, l + 1),
            preActivations: fwdData.preActivations,
          });
          await delay(300);
        }
        if (!cancelled) {
          setAnimatingLayer(-1);
          setForwardPassDisplay({
            activations:    fwdData.activations,
            preActivations: fwdData.preActivations,
          });
        }
      } else if (stepModeStage === 'loss') {
        // No layer animation — forward activations stay from prior stage
      } else if (stepModeStage === 'backward') {
        setLastGradients({ dWeights: stepModeResult.avgDW, dBiases: stepModeResult.avgDB });
        if (!dismissedCalloutsRef.current.has('firstBackprop')) {
          setCallouts(prev => new Set([...prev, 'firstBackprop']));
        }
        for (let l = layerSizes.length - 1; l >= 0; l--) {
          if (cancelled) return;
          setAnimatingLayer(l);
          await delay(300);
        }
        if (!cancelled) setAnimatingLayer(-1);
      } else if (stepModeStage === 'update') {
        // Apply weights exactly once; re-visiting this stage is idempotent.
        if (!stepModeAppliedRef.current) {
          stepModeAppliedRef.current = true;
          networkRef.current = { weights: stepModeResult.weights, biases: stepModeResult.biases };
          setNetwork({ weights: stepModeResult.weights, biases: stepModeResult.biases });

          epochRef.current += 1;
          setEpoch(epochRef.current);

          lossHistoryRef.current = [...lossHistoryRef.current,
            { epoch: epochRef.current, loss: stepModeResult.loss }].slice(-200);
          setLossHistory([...lossHistoryRef.current]);

          if (stepModeResult.loss < bestLossRef.current - MIN_IMPROVEMENT) {
            bestLossRef.current          = stepModeResult.loss;
            epochsSinceImprovRef.current = 0;
            setBestLoss(stepModeResult.loss);
          } else {
            epochsSinceImprovRef.current += 1;
          }
          setEpochsSinceImprovement(epochsSinceImprovRef.current);

          const xorEval = evaluateXOR(
            stepModeResult.weights, stepModeResult.biases, activationTypes
          );
          setXorResults(xorEval);

          const allHighConf = xorEval.every(
            r => r.correct && r.confidence > CONVERGENCE_CONFIDENCE
          );
          if (allHighConf) consecutiveCorrectRef.current += 1;
          else             consecutiveCorrectRef.current  = 0;

          const { converged, reason } = checkConvergence(
            stepModeResult.loss, xorEval, consecutiveCorrectRef.current
          );
          if (converged) {
            setTrainingStatus('converged');
            setStopReason(reason);
            if (!dismissedCalloutsRef.current.has('converged')) {
              setCallouts(prev => new Set([...prev, 'converged']));
            }
          } else if (
            stepModeResult.loss > PLATEAU_MIN_LOSS &&
            epochsSinceImprovRef.current >= PLATEAU_PATIENCE &&
            !dismissedCalloutsRef.current.has('lossPlateauing')
          ) {
            setTrainingStatus('plateaued');
            setStopReason(
              `Loss has not improved by >${MIN_IMPROVEMENT} in ${PLATEAU_PATIENCE} epochs`
            );
            setCallouts(prev => new Set([...prev, 'lossPlateauing']));
          }

          // Vanishing gradient check — same logic as runTrainingStep
          const firstLayerMax = Math.max(...stepModeResult.avgDW[0].flat().map(Math.abs));
          const networkMax    = Math.max(...stepModeResult.avgDW.flat(2).map(Math.abs));
          if (
            networkMax > 0 &&
            firstLayerMax / networkMax < 0.01 &&
            stepModeResult.loss > PLATEAU_MIN_LOSS &&
            !dismissedCalloutsRef.current.has('vanishingGradient')
          ) {
            setCallouts(prev => new Set([...prev, 'vanishingGradient']));
          }
        }
      }

      if (!cancelled) {
        setStepModeLayerAnimDone(true);
        isAnimatingRef.current = false;
      }
    };

    run();
    return () => {
      cancelled = true;
      setAnimatingLayer(-1);
      isAnimatingRef.current = false;
    };
  // stepModeAppliedRef is a ref — intentionally omitted from deps
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stepModeStage, stepModeResult]);

  // Navigation handlers — defined before auto-play effect so Next can reference Done
  const handleExplainedDone = useCallback(() => {
    if (stepModeAutoTimerRef.current) {
      clearTimeout(stepModeAutoTimerRef.current);
      stepModeAutoTimerRef.current = null;
    }
    setStepModeStage('idle');
    setStepModeResult(null);
    setStepModeAutoPlay(false);
    setAnimatingLayer(-1);
    isAnimatingRef.current = false;
    if (!['converged', 'plateaued', 'maxEpochs'].includes(trainingStatus)) {
      setTrainingStatus('paused');
    }
  }, [trainingStatus]);

  const handleExplainedPrev = useCallback(() => {
    const stages = ['forward', 'loss', 'backward', 'update'];
    const idx = stages.indexOf(stepModeStage);
    if (idx > 0) setStepModeStage(stages[idx - 1]);
  }, [stepModeStage]);

  const handleExplainedNext = useCallback(() => {
    const stages = ['forward', 'loss', 'backward', 'update'];
    const idx = stages.indexOf(stepModeStage);
    if (idx < stages.length - 1) setStepModeStage(stages[idx + 1]);
    else handleExplainedDone();
  }, [stepModeStage, handleExplainedDone]);

  // Auto-play: when enabled and current stage animation is complete, schedule Next.
  useEffect(() => {
    if (stepModeAutoTimerRef.current) {
      clearTimeout(stepModeAutoTimerRef.current);
      stepModeAutoTimerRef.current = null;
    }
    if (!stepModeAutoPlay || !stepModeLayerAnimDone || stepModeStage === 'idle') return;
    stepModeAutoTimerRef.current = setTimeout(
      () => handleExplainedNext(),
      stepModeAutoSpeed * 1000
    );
    return () => {
      if (stepModeAutoTimerRef.current) {
        clearTimeout(stepModeAutoTimerRef.current);
        stepModeAutoTimerRef.current = null;
      }
    };
  }, [stepModeAutoPlay, stepModeLayerAnimDone, stepModeStage, stepModeAutoSpeed, handleExplainedNext]);

  // ── Click-to-predict with forward-pass animation ──────────────────────────
  // Runs a real forward pass on the clicked canvas point, then animates the
  // activations propagating through the network layer-by-layer.
  const handleCanvasClick = async (e) => {
    if (!networkRef.current || isAnimatingRef.current) return;
    const canvas = e.currentTarget;
    const rect   = canvas.getBoundingClientRect();
    const x1     = (e.clientX - rect.left) / rect.width;
    const cy     = (e.clientY - rect.top)  / rect.height;
    const x2     = 1 - cy; // flip: canvas top = x₂=1

    const { weights, biases } = networkRef.current;
    const { activations, preActivations } = forwardPass([x1, x2], weights, biases, activationTypes);
    const prediction = activations[activations.length - 1][0];

    setInferencePoint({ x: x1, y: x2, prediction });
    if (!dismissedCalloutsRef.current.has('inferencePoint')) {
      setCallouts(prev => new Set([...prev, 'inferencePoint']));
    }

    // Animate the forward pass through the network for this custom point
    isAnimatingRef.current = true;
    for (let l = 0; l <= layerSizes.length - 1; l++) {
      setAnimatingLayer(l);
      setForwardPassDisplay({
        activations:    activations.slice(0, l + 1),
        preActivations,
      });
      await new Promise(r => setTimeout(r, 370));
    }
    setAnimatingLayer(-1);
    setForwardPassDisplay({ activations, preActivations });
    isAnimatingRef.current = false;
  };

  const dismissCallout = (type) => {
    setCallouts(prev        => { const s = new Set(prev); s.delete(type); return s; });
    setDismissedCallouts(prev => new Set([...prev, type]));
  };

  const latestLoss     = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].loss : null;
  const activeCallouts = [...callouts].filter(c => !dismissedCallouts.has(c)).slice(0, 2);
  const isBusy         = isTraining || stepModeStage !== 'idle';

  // During the explained epoch, hide the old gradient data from the forward and
  // loss stages so the user sees plain edges (before backprop runs). This makes
  // the before/after contrast of Stage 3 visually clear.
  const networkBackpropData = (stepModeStage === 'forward' || stepModeStage === 'loss')
    ? null
    : lastGradients;

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans flex flex-col">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-700 px-5 py-2.5 flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-base font-bold text-white leading-tight">Neural Net Playground</h1>
          <p className="text-xs text-slate-500">Make the math visible.</p>
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
              <button onClick={handleToggleTraining} disabled={isBusy && !isTraining}
                className={`w-full py-1.5 rounded font-bold text-sm transition-colors ${
                  isTraining
                    ? 'bg-amber-600 hover:bg-amber-500'
                    : isBusy
                    ? 'bg-slate-700 opacity-50 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-500'
                } text-white`}>
                {isTraining ? '⏸ Pause' : trainingStatus === 'converged' ? '▶ Continue' : '▶ Train'}
              </button>

              <button onClick={handleStepEpoch} disabled={isBusy}
                className="w-full py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed">
                Step 1 Epoch
              </button>

              {/* Phase 2: Explained epoch button — user-paced 4-stage mode */}
              <button onClick={startExplainedEpoch} disabled={isBusy}
                className="w-full py-1 rounded text-xs bg-violet-900/70 hover:bg-violet-800 text-violet-300 disabled:opacity-40 disabled:cursor-not-allowed border border-violet-700/50">
                Explained Step ←→
              </button>

              <button onClick={runForwardPassAnimation} disabled={isBusy}
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
              <span className="text-slate-400 font-medium block mb-0.5">Click to Predict</span>
              Click anywhere on the decision boundary to run a real forward pass and animate it.
            </div>
            {inferencePoint && (
              <div className="mt-1.5 bg-slate-800 rounded p-2 space-y-1.5">
                {/* Prominent class + confidence pill */}
                <div className={`text-center py-1.5 rounded font-bold text-sm ${
                  inferencePoint.prediction > 0.5
                    ? 'bg-orange-900/50 text-orange-300 border border-orange-700/40'
                    : 'bg-blue-900/50 text-blue-300 border border-blue-700/40'
                }`}>
                  Class {inferencePoint.prediction > 0.5 ? 1 : 0}
                  <span className="ml-2 font-normal text-xs opacity-80">
                    {(Math.abs(inferencePoint.prediction - 0.5) * 2 * 100).toFixed(1)}% conf
                  </span>
                </div>
                {/* Raw values */}
                <div className="text-xs font-mono text-slate-500 space-y-0.5">
                  <div>x₁ = {inferencePoint.x.toFixed(3)}  x₂ = {inferencePoint.y.toFixed(3)}</div>
                  <div>p(class=1) = <span className="text-slate-300">{inferencePoint.prediction.toFixed(5)}</span></div>
                  <div className="text-slate-600 text-xs">conf = |p−0.5|×2 = {(Math.abs(inferencePoint.prediction - 0.5)*2).toFixed(4)}</div>
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
                <ConceptCallout key={type} type={type} onDismiss={() => dismissCallout(type)}
                  trainingStatus={trainingStatus} hiddenActivationTypes={activationTypes} />
              ))}
            </div>
          )}

          {/* ── Explained Step Banner — visible while step mode is active ───────── */}
          {stepModeStage !== 'idle' && (() => {
            const stages = ['forward', 'loss', 'backward', 'update'];
            const stageIdx = stages.indexOf(stepModeStage);
            const isLast = stageIdx === stages.length - 1;
            const col = {
              forward:  { border: 'border-blue-500',    bg: 'bg-blue-950/60',    title: 'text-blue-300',    dot: 'bg-blue-400' },
              loss:     { border: 'border-amber-500',   bg: 'bg-amber-950/60',   title: 'text-amber-300',   dot: 'bg-amber-400' },
              backward: { border: 'border-violet-500',  bg: 'bg-violet-950/60',  title: 'text-violet-300',  dot: 'bg-violet-400' },
              update:   { border: 'border-emerald-500', bg: 'bg-emerald-950/60', title: 'text-emerald-300', dot: 'bg-emerald-400' },
            }[stepModeStage];
            return (
              <div className={`flex-shrink-0 border-l-4 ${col.border} ${col.bg} rounded-r-lg p-2.5 text-xs`}>
                {/* Progress dots + close button */}
                <div className="flex items-center gap-1.5 mb-1.5">
                  {stages.map((s, i) => (
                    <div key={s} className={`w-2 h-2 rounded-full transition-all ${
                      s === stepModeStage ? `${col.dot} animate-pulse` :
                      i < stageIdx       ? 'bg-slate-500' : 'bg-slate-700'
                    }`} />
                  ))}
                  <span className="text-slate-500 ml-0.5 font-mono">Stage {stageIdx + 1} of 4</span>
                  <button onClick={handleExplainedDone}
                    className="ml-auto text-slate-500 hover:text-white text-xs leading-none px-1 py-0.5 rounded hover:bg-slate-700/60"
                    title="Close explained step">✕</button>
                </div>

                {/* Per-stage educational content */}
                {stepModeStage === 'forward' && (
                  <div>
                    <div className={`font-bold ${col.title} mb-0.5`}>
                      → Forward Pass <span className="font-mono font-normal text-slate-400">input = [0, 0]</span>
                    </div>
                    <div className="text-slate-300">Each neuron computes <span className="font-mono text-blue-200">z = W·x + b</span> then <span className="font-mono text-blue-200">a = activation(z)</span>. Watch values light up left→right.</div>
                    <div className="mt-1 font-mono text-blue-400">out = model([0,0])  →  p(class=1) shown on output neuron</div>
                  </div>
                )}
                {stepModeStage === 'loss' && (
                  <div>
                    <div className={`font-bold ${col.title} mb-0.5`}>📊 Loss Calculation</div>
                    <div className="text-slate-300">Binary Cross-Entropy penalizes wrong-confident predictions harshly. Lower = better fit to all 4 XOR samples.</div>
                    <div className="mt-1 font-mono text-amber-400">
                      L = −(1/4)·Σ[y·log(ŷ)+(1−y)·log(1−ŷ)] = <span className="text-amber-200 font-bold">{stepModeLoss?.toFixed(6)}</span>
                    </div>
                  </div>
                )}
                {stepModeStage === 'backward' && (
                  <div>
                    <div className={`font-bold ${col.title} mb-0.5`}>
                      ← Backpropagation <span className="font-mono font-normal text-slate-400">computing ∂L/∂w</span>
                    </div>
                    <div className="text-slate-300">Chain rule flows right→left. Edge brightness = gradient magnitude. Violet pulse = active layer. Brighter edge = larger weight update coming.</div>
                    <div className="mt-1 font-mono text-violet-400">δ[L]=ŷ−y · δ[l]=(Wᵀδ[l+1])⊙σ'(z) · ∂L/∂W=δ·aᵀ</div>
                  </div>
                )}
                {stepModeStage === 'update' && (
                  <div>
                    <div className={`font-bold ${col.title} mb-0.5`}>↻ Weight Update Applied</div>
                    <div className="text-slate-300">Every weight stepped opposite its gradient. Edges still show the gradients just applied. The boundary canvas redraws with new weights.</div>
                    <div className="mt-1 font-mono text-emerald-400">W ← W − {learningRate.toFixed(3)}·∂L/∂W <span className="text-slate-600">(optimizer.step())</span></div>
                  </div>
                )}

                {/* Navigation controls */}
                <div className="flex items-center gap-2 border-t border-slate-700/60 pt-2 mt-2">
                  <button onClick={handleExplainedPrev} disabled={stageIdx === 0}
                    className="px-2 py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300 transition-colors">
                    ← Prev
                  </button>
                  <button onClick={() => setStepModeAutoPlay(v => !v)}
                    className={`px-2 py-1 rounded text-xs border transition-colors ${
                      stepModeAutoPlay
                        ? 'bg-slate-600 text-white border-slate-500'
                        : 'bg-slate-800 text-slate-400 border-slate-600 hover:text-slate-300'
                    }`}>
                    {stepModeAutoPlay ? '⏸ Auto' : '▶ Auto'}
                  </button>
                  {stepModeAutoPlay ? (
                    <div className="flex items-center gap-1 flex-1 min-w-0">
                      <span className="text-slate-400 shrink-0">{stepModeAutoSpeed}s</span>
                      <input type="range" min={1} max={10} step={0.5} value={stepModeAutoSpeed}
                        onChange={e => setStepModeAutoSpeed(+e.target.value)}
                        className="flex-1 accent-slate-400 min-w-0" />
                    </div>
                  ) : (
                    <div className="flex-1" />
                  )}
                  <button onClick={isLast ? handleExplainedDone : handleExplainedNext}
                    className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
                      isLast
                        ? 'bg-emerald-700 hover:bg-emerald-600 text-white'
                        : 'bg-blue-700 hover:bg-blue-600 text-white'
                    }`}>
                    {isLast ? 'Done ✓' : stepModeLayerAnimDone ? 'Next →' : 'Next…'}
                  </button>
                </div>
              </div>
            );
          })()}

          {/* Network Graph */}
          <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-2 flex-shrink-0">
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-xs font-semibold text-slate-300">Network</h2>
              <div className="flex items-center gap-2 text-xs">
                {/* During standalone animations show layer indicator */}
                {stepModeStage === 'idle' && animatingLayer >= 0 && (
                  <span className="text-blue-400 font-mono animate-pulse">→ layer {animatingLayer}</span>
                )}
                {stepModeStage === 'idle' && lastGradients && animatingLayer < 0 && (
                  <span className="text-violet-400 font-mono">edges = |∂L/∂w|</span>
                )}
                {/* Toggle numeric gradient labels on edges */}
                {lastGradients && (
                  <button
                    onClick={() => setShowGradientLabels(v => !v)}
                    className={`text-xs px-1.5 py-0.5 rounded border transition-colors ${
                      showGradientLabels
                        ? 'bg-violet-900/60 text-violet-300 border-violet-700'
                        : 'bg-slate-800 text-slate-500 border-slate-700 hover:text-slate-300'
                    }`}>
                    ∂w labels
                  </button>
                )}
              </div>
            </div>
            <NetworkGraph
              layerSizes={layerSizes}
              hiddenActivationTypes={activationTypes}
              forwardData={forwardPassDisplay || lastForwardData}
              backpropData={networkBackpropData}
              animatingLayer={animatingLayer}
              animatingBackward={stepModeStage === 'backward'}
              showGradientLabels={showGradientLabels}
            />
          </div>

          {/* Bottom row: Decision Boundary + Loss Curve */}
          <div className="flex gap-3 flex-shrink-0" style={{ height: '295px' }}>
            {/* Decision Boundary */}
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-2 flex-shrink-0">
              <div className="flex items-center justify-between mb-1">
                <h2 className="text-xs font-semibold text-slate-300">Decision Boundary</h2>
                {/* Phase 2: Toggle between class-color and confidence-brightness views */}
                <button
                  onClick={() => setShowConfidence(v => !v)}
                  className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                    showConfidence
                      ? 'bg-amber-800/60 text-amber-300 border-amber-700'
                      : 'bg-slate-700 text-slate-400 border-slate-600 hover:text-slate-200'
                  }`}>
                  {showConfidence ? 'Confidence ▲' : 'Class ▲'}
                </button>
              </div>
              {network ? (
                <DecisionBoundaryCanvas
                  weights={network.weights}
                  biases={network.biases}
                  hiddenActivationTypes={activationTypes}
                  inferencePoint={inferencePoint}
                  onClick={handleCanvasClick}
                  showConfidence={showConfidence}
                />
              ) : (
                <div className="w-[260px] h-[260px] bg-slate-800 rounded flex items-center justify-center text-slate-600 text-xs">
                  Initializing…
                </div>
              )}
              <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                {!showConfidence ? (
                  <>
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-blue-500 inline-block"/>Class 0</span>
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-orange-500 inline-block"/>Class 1</span>
                    <span className="text-slate-700">· click = predict</span>
                  </>
                ) : (
                  <>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: '#1e293b', border: '1px solid #334155' }}/>
                      boundary (uncertain)
                    </span>
                    <span className="text-slate-700">→</span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-amber-500 inline-block"/>
                      confident
                    </span>
                    <span className="text-slate-600 ml-auto">|p−0.5|×2</span>
                  </>
                )}
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
        <div className="w-80 border-l border-slate-700 flex flex-col overflow-hidden flex-shrink-0">

          <PyTorchPanel
            layerSizes={layerSizes}
            hiddenActivationTypes={activationTypes}
            learningRate={learningRate}
          />

          <div className="flex-shrink-0 p-3 border-b border-slate-700">
            <div className="bg-slate-800/40 rounded-lg border border-slate-700 p-2.5">
              <XorVerifyPanel xorResults={xorResults} />
            </div>
          </div>

          {/* ── Math Audit / ∂w Check / Weights / Calculus — tabbed ── */}
          <div className="flex-1 min-h-0 flex flex-col overflow-hidden border-b border-slate-700">
            {/* Tab switcher — 4 tabs */}
            <div className="flex flex-shrink-0 border-b border-slate-700">
              {[
                { id: 'audit',    label: 'Audit',    active: 'text-blue-400 border-blue-500'    },
                { id: 'gradcheck',label: '∂w Check', active: 'text-indigo-400 border-indigo-500'},
                { id: 'weights',  label: 'Weights',  active: 'text-amber-400 border-amber-500'  },
                { id: 'calc',     label: '∫ Calc',   active: 'text-emerald-400 border-emerald-500'},
              ].map(({ id, label, active }) => (
                <button key={id}
                  onClick={() => setRightPanelTab(id)}
                  className={`flex-1 text-xs py-1.5 font-medium transition-colors ${
                    rightPanelTab === id
                      ? `${active} bg-slate-800/50 border-b-2`
                      : 'text-slate-500 hover:text-slate-300'
                  }`}>
                  {label}
                </button>
              ))}
            </div>
            {/* Tab content */}
            <div className="flex-1 min-h-0 overflow-y-auto p-3">
              {rightPanelTab === 'weights' && (
                <WeightsInspector
                  network={network}
                  layerSizes={layerSizes}
                  hiddenActivationTypes={activationTypes}
                  epoch={epoch}
                  trainingStatus={trainingStatus}
                  latestLoss={latestLoss}
                  xorResults={xorResults}
                />
              )}
              {rightPanelTab === 'calc' && (
                <CalcPanel
                  network={network}
                  layerSizes={layerSizes}
                  hiddenActivationTypes={activationTypes}
                  lastGradients={lastGradients}
                  lastForwardData={lastForwardData}
                />
              )}
              {(rightPanelTab === 'audit' || rightPanelTab === 'gradcheck') && (
                <div className="bg-slate-800/40 rounded-lg border border-slate-700 p-2.5">
                  {rightPanelTab === 'audit' ? (
                    <MathAuditPanel
                      xorResults={xorResults}
                      hiddenActivationTypes={activationTypes}
                      layerSizes={layerSizes}
                    />
                  ) : (
                    <GradientCheckPanel
                      network={network}
                      hiddenActivationTypes={activationTypes}
                      layerSizes={layerSizes}
                      lastGradients={lastGradients}
                    />
                  )}
                </div>
              )}
            </div>
          </div>

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
