# Neural Net Playground

**Make the math visible.**

An interactive, browser-based neural-network visualizer. Build and train a multilayer perceptron on XOR, watch activations propagate forward, inspect gradients flowing back, trace the chain rule step-by-step, and explore activation functions — all backed by real mathematics implemented in plain JavaScript.

No ML libraries. No backend. No mocked values. Every number you see is computed from first principles.

---

## Features

### Network builder

- 1–4 configurable hidden layers, 2–8 neurons each
- Activation function per hidden layer: ReLU, Tanh, or Sigmoid
- Architecture changes reinitialize the network immediately

### Datasets

- **Logical gates**: XOR, AND, OR — the classic 4-point truth tables
- **Geometric (generated)**: circles, moons, spirals, linear, blobs — all normalized into `[0,1]²`
- Geometric datasets expose **points**, **noise**, and **seed** controls (seeded so a given configuration is reproducible; ↻ reshuffle bumps the seed)
- Switching dataset reinitializes the network and re-runs training, the decision boundary, and every panel against the new points
- Panels adapt to dataset size: per-point tables for the 4-point gates; small dot markers, an accuracy/confidence summary, and dropdown sample pickers for the larger generated sets

### Guided lessons

A **📚 Lessons** launcher (header) opens a non-modal tour player. Each lesson configures the playground for you (dataset, architecture, optimizer, learning rate, active panel) and walks you through a concept while you drive the real controls:

- **Solving XOR** — why a hidden layer is needed; watch the boundary bend
- **Capacity & dead ReLU** — how too few ReLU neurons stall, and why width helps
- **Optimizers race** — on Spirals, see Adam/Momentum outpace plain SGD
- **The loss landscape** — read a 2-D loss-surface slice and trace a descent path

Lesson content lives in `src/lessons.js` and is validated by tests.

### Training

| Control | Description |
|---|---|
| **Optimizer** | SGD, Momentum, RMSProp, or Adam — implemented from scratch, matching PyTorch's defaults |
| **Train / Pause** | Continuous full-batch gradient descent |
| **Step** | Advance one epoch at a time |
| **Explained Step** | 4-stage interactive walkthrough: Forward → Loss → Backward → Update, with Next / Prev / Auto-play controls |
| **Reset** | Reinitialize with new random weights |

Convergence auto-stops when loss drops below 0.001, or when every point in the active dataset is correctly classified with >95% confidence for 50 consecutive epochs. A plateau detector fires when improvement stalls.

### Visualization

| Panel | Description |
|---|---|
| **Network graph** | SVG with neurons colored by activation magnitude; edges colored by `\|∂L/∂w\|` after each backward pass |
| **Decision boundary** | 40×40 grid of real forward passes over `[0,1]²`, updated every render |
| **Confidence heatmap** | Toggle between class colors and `\|p − 0.5\| × 2` brightness |
| **Click-to-predict** | Click anywhere on the boundary canvas to run inference and animate the forward pass |
| **Loss curve** | Live BCE loss plotted with Recharts |

### Right panel tabs

| Tab | Contents |
|---|---|
| **Audit** | Per-sample forward-pass trace with symbolic BCE formula and numeric values |
| **∂w Check** | Backprop gradient vs symmetric finite-difference estimate `[L(w+ε)−L(w−ε)] / 2ε`; auto-pick selects the weight with the largest `\|∂L/∂w\|` |
| **Weights** | Color-coded weight matrices and bias vectors per layer (amber = positive, blue = negative); parameter JSON export |
| **∫ Calc** | Chain Rule Tracer and Activation Function Explorer (see below) |
| **Surface** | 2-D loss-landscape slice over two selectable weights (real-loss heatmap), with a live position marker, zoom, and an optimizer descent-path overlay |

### ∫ Calc — Calculus panel

**∂w Trace (Chain Rule Tracer)**

Select any weight W[layer][j][k] and any XOR input sample. The panel re-runs a live forward + backprop to show:
- Symbolic formula: `∂L/∂w = δⱼ · aₖ`
- Per-term numeric breakdown: aₖ (incoming activation), zⱼ + f′(zⱼ), δⱼ
- Dead ReLU and saturation warnings where applicable
- Per-sample gradient vs batch-averaged gradient with update-direction explanation

**f(z) Plot (Activation Function Explorer)**

Select any neuron and XOR input. Plots:
- f(z) curve and f′(z) derivative overlay (toggleable)
- Tangent line at the current z value
- Reference markers for z, f(z), f′(z)
- Saturation and dead-ReLU callouts from actual computed values

### PyTorch export panel

- Architecture/optimizer/activation mapping summary
- **Copy Script**: complete runnable `.py` file
- **Export Notebook**: `.ipynb` (nbformat v4, 18 cells) covering imports, training, loss curve, XOR verification, decision boundary, and inference
- Collapsible full code block

### Weights / Parameters Inspector

- Per-layer weight matrices: color-coded by sign and magnitude
- `W[out_feature][in_feature]` — same shape as `nn.Linear.weight`; no transposition needed when loading into PyTorch
- **Copy JSON / Download JSON** → `neural-viz-params.json`: weights, biases, architecture, training state, per-point XOR verification, and convergence reason
- Collapsible LLM analogy and PyTorch weight-loading snippet

---

## Getting Started

### Prerequisites

- Node.js 18 or later
- npm 9 or later

### Install and run

```bash
git clone https://github.com/MrZoller/neural-viz.git
cd neural-viz
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### Build for production

```bash
npm run build
npm run preview
```

---

## How It Works

All neural-network math lives in `src/nn/` with detailed inline comments, separated from the React UI in `src/App.jsx`. The data flow:

```
initNetwork()               Xavier-initialized weights, zero biases
      ↓
forwardPass()               z = W·x + b → activation(z), layer by layer; stores all
                            intermediate values for backprop reuse
      ↓
computeLoss()               Binary cross-entropy: −mean(y·log(p) + (1−y)·log(1−p))
      ↓
backprop()                  δ[L] = ŷ−y  (BCE+sigmoid shortcut)
                            δ[l] = (Wᵀ·δ[l+1]) ⊙ f′(z[l])
                            dW[l] = δ[l+1] · a[l]ᵀ
      ↓
updateWeights()             W ← W − lr · ∂L/∂W
      ↓
computeDecisionBoundary()   40×40 forward passes over [0,1]² input space
```

### Key design decisions

**Why XOR?** XOR is not linearly separable, so a single-layer network cannot solve it. This makes it the minimal demonstration of why hidden layers and nonlinear activations exist.

**Why full-batch gradient descent?** Simpler to understand for educational purposes — every epoch is one clean pass over the whole dataset, which keeps the loss curve and the Explained Step walkthrough easy to follow. (For the 4-point logical gates, mini-batching would add noise without benefit; the generated datasets are still small enough that full batch is fine.)

**Why BCE + sigmoid on the output?** Binary Cross-Entropy paired with a sigmoid output has a numerically convenient combined gradient: `∂L/∂z_output = ŷ − y`. The sigmoid derivative cancels algebraically, avoiding the saturation problem at the output layer.

**Why Xavier initialization?** Setting weight scale to `sqrt(2 / (fan_in + fan_out))` keeps activation variance roughly constant across layers at the start of training, reducing the chance of vanishing or exploding gradients before learning begins.

---

## Project Structure

```
neural-viz/
├── public/
│   └── favicon.svg        # SVG favicon (2-2-1 network icon)
├── src/
│   ├── nn/                # Neural-network math core (no React, fully unit-tested)
│   │   ├── activations.js # Activation functions + derivatives, activation curve
│   │   ├── datasets.js    # XOR dataset
│   │   ├── network.js     # initNetwork, forwardPass, loss, backprop, update, boundary
│   │   ├── training.js    # trainOneEpoch, evaluateXOR, convergence, gradient check
│   │   ├── index.js       # Public surface (barrel) imported by App.jsx
│   │   └── __tests__/     # Vitest suites (activations, network, training, optimizers, surface)
│   ├── lessons.js         # Guided-lesson content (data) + its test in src/__tests__
│   ├── App.jsx            # All UI components — imports the math from src/nn
│   ├── main.jsx           # React root mount
│   └── index.css          # Tailwind directives + minimal animation helpers
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

### Math core (`src/nn/`)

The from-scratch neural-network math is isolated from the UI so it can be read
and tested on its own. Nothing in `src/nn/` imports React.

| Module | Contents |
|---|---|
| `activations.js` | Activation functions (ReLU, Tanh, Sigmoid) with exact derivatives; activation-curve generator for the calculus panel |
| `datasets.js` | Dataset registry — logical gates (XOR/AND/OR) and seeded geometric generators (circles, moons, spirals, linear, blobs) |
| `network.js` | Network initialization (Xavier / Glorot), forward pass, BCE loss, backpropagation, gradient-descent update, decision-boundary computation |
| `optimizers.js` | SGD, Momentum, RMSProp, Adam — buffers + update rules matching PyTorch |
| `training.js` | One full-batch training epoch (optimizer-aware), dataset evaluation, optimizer comparison, convergence / stop conditions, finite-difference gradient check |
| `surface.js` | 2-D loss-surface slice over two weights, and the descent-path tracer |

`App.jsx` holds the React components (in order of declaration: `NetworkGraph`,
`DecisionBoundaryCanvas`, `ConceptCallout`, `XorVerifyPanel`, `MathAuditPanel`,
`GradientCheckPanel`, `ChainRuleTracer`, `ActivationExplorer`, `CalcPanel`,
`WeightsInspector`, `TrainingStatusBar`, `App`) plus the PyTorch code/notebook
generators and the SVG colour/layout helpers.

### Tests

```bash
npm test          # run the Vitest suite once
npm run test:watch
```

The suite (37 tests) covers the math core directly. Its centrepiece verifies
backpropagation against a symmetric finite-difference estimate — the same check
exposed in the UI's **∂w Check** tab — fuzzing across 40 randomly generated
architectures and activation combinations to assert that every analytical
gradient agrees with the numerical one to within ~1e-4 relative error. Because
the whole project's pitch is "every number is real math," these tests are the
proof.

Components (in order of declaration): `NetworkGraph`, `DecisionBoundaryCanvas`, `ConceptCallout`, `XorVerifyPanel`, `MathAuditPanel`, `GradientCheckPanel`, `ChainRuleTracer`, `ActivationExplorer`, `CalcPanel`, `WeightsInspector`, `TrainingStatusBar`, and the main `App`.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| React | 18 | UI and state management |
| Vite | 5 | Dev server and production build |
| Tailwind CSS | 3 | Utility-first styling |
| Recharts | 2 | Loss curve and activation function plots |
| Vitest | 2 | Unit tests for the math core (dev only) |

No ML libraries are used at runtime. All neural-network math — initialization, forward pass, backpropagation, gradient descent — is implemented from scratch in plain JavaScript in `src/nn/` and verified by the Vitest suite.

---

## Honesty Constraints

This tool makes deliberate simplifications for educational clarity. Each simplification is labeled in both the UI and the code comments.

| Simplification | Detail |
|---|---|
| Full-batch gradient descent | Not stochastic or mini-batch |
| No weight decay / schedules | Optimizers (SGD, Momentum, RMSProp, Adam) are supported, but there is no weight decay or learning-rate schedule |
| Fixed output activation | Output is always sigmoid regardless of hidden activations chosen |
| Binary classification only | Two classes; output is a single sigmoid probability |
| PyTorch code is explanatory | Generated scripts embed the active dataset's points but reinitialize weights randomly; use the Weights tab to export trained values |
| Loss surface is a 2-D slice | The Surface tab plots real loss over two weights with all others fixed; the true landscape has as many dimensions as there are parameters, and the descent-path overlay is the trajectory projected onto those two axes |

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
