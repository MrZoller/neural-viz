# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Planned — Phase 2
- Backprop visualization with numeric gradient values displayed on edges
- Vanishing gradient highlight with color-coded input-layer edges
- Confidence heatmap toggle on the decision boundary canvas
- Forward-pass animation for inference (layer-by-layer activation display on click-to-predict)

### Planned — Phase 3
- Chain rule tracer: click any weight to see the full ∂L/∂w derivation written out step by step
- Activation derivative explorer: plot activation function with moveable input point and tangent line
- Loss surface contour plot over two selected weights with gradient descent path trail
- Test batch panel: add multiple test points, view results table with predicted class and confidence

---

## [0.1.0] — 2026-05-31

### Added

**Core neural network math (all implemented in plain JavaScript, no ML libraries)**

- `initNetwork()` — Xavier (Glorot) weight initialization: `scale = sqrt(2 / (fan_in + fan_out))`
- `forwardPass()` — full forward propagation: `z = W·x + b`, `a = activation(z)`, stores all intermediate values for backprop reuse
- `computeLoss()` — Binary Cross-Entropy loss with numerical clipping to avoid `log(0)`
- `backprop()` — full backpropagation via chain rule recurrence: `δ[L] = ŷ−y` (BCE+sigmoid shortcut), `δ[l] = (Wᵀ·δ[l+1]) ⊙ σ'(z[l])`
- `updateWeights()` — vanilla SGD: `W ← W − lr · ∂L/∂W`
- `trainOneEpoch()` — full-batch gradient descent over all 4 XOR examples with gradient accumulation and averaging
- `computeDecisionBoundary()` — 40×40 grid of real forward passes over `[0,1]²`
- `generatePyTorchCode()` — live PyTorch code generation reflecting current architecture

**Activation functions with exact derivatives**
- ReLU: `max(0, x)`, derivative `x > 0 ? 1 : 0`
- Tanh: `tanh(x)`, derivative `1 − tanh²(x)`
- Sigmoid: `1 / (1 + e^−x)`, derivative `σ(x) · (1 − σ(x))`

**UI — Network Builder**
- 1–4 configurable hidden layers
- 2–8 neurons per hidden layer
- Activation function selector per hidden layer (ReLU / Tanh / Sigmoid)
- Architecture resets and reinitializes the network on any change

**UI — Visualization**
- SVG network graph with neurons and weighted edges
- Neurons colored by activation magnitude during forward pass (blue = low, orange = high)
- Edges colored by `∂L/∂W` gradient magnitude after each backward pass (gray = near-zero, red = large) with inline legend
- Layer-by-layer forward-pass animation with actual computed activation values on each neuron

**UI — Training Controls**
- Train (continuous) / Pause / Step one epoch / Reset
- Learning rate slider (0.001 – 1.0)
- Live loss curve via Recharts (BCE, epoch axis)
- Epoch and current loss displayed in header

**UI — Decision Boundary Canvas**
- 40×40 grid drawn to HTML canvas, recomputed from actual weights on each render
- XOR training points overlaid (blue = class 0, orange = class 1)
- Click anywhere to run inference at that point; result shows predicted class and confidence

**UI — PyTorch Sidebar**
- Live `nn.Sequential` code reflecting current architecture
- Updates on every layer/neuron/activation change
- Parameter count summary per layer and total

**UI — Concept Callouts**
- First forward pass: explains `z = W·x + b → activation(z)` and the PyTorch equivalent
- First backprop: explains the chain rule recurrence and `loss.backward()`
- Loss plateauing: fires when `Δloss < 0.005` over 20 epochs while loss is still above 0.1
- Vanishing gradient: fires when first-layer max gradient is less than 1% of the overall max gradient
- Inference mode: explains `model.eval()`, `torch.no_grad()`, and why they matter

**Project setup**
- Vite 5 + React 18 + Tailwind CSS 3 + Recharts 2
- Single-file artifact: all math and components in `src/App.jsx`

[Unreleased]: https://github.com/MrZoller/neural-viz/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MrZoller/neural-viz/releases/tag/v0.1.0
