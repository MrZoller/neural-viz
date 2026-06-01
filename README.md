# Neural Network Learning Tool

An interactive, browser-based tool for understanding how neural networks work. Build and train a multilayer perceptron (MLP) in real time, watch activations propagate forward, inspect gradients flowing back, and see the decision boundary update live as training progresses — all backed by real mathematics implemented in JavaScript.

No ML libraries. No backend. No mocked values. Every number you see is computed from first principles.

---

## Features

### Phase 1 (current)

| Feature | Description |
|---|---|
| **Network builder** | Configure 1–4 hidden layers, 2–8 neurons each, with ReLU / Tanh / Sigmoid per layer |
| **Forward-pass animation** | Watch activations propagate layer by layer with actual computed values on every neuron |
| **Training loop** | Full-batch gradient descent on XOR with pause / resume / step / reset controls |
| **Live loss curve** | Real-time BCE loss plotted with Recharts |
| **Decision boundary** | 40×40 grid recomputed from actual weights every render — not interpolated |
| **Gradient edge coloring** | Edges colored by ∂L/∂W magnitude after each backward pass |
| **Click-to-predict** | Click anywhere on the boundary canvas to run a forward pass at that point |
| **PyTorch sidebar** | Live code snippet that reflects the current architecture and updates as you change it |
| **Concept callouts** | Explanatory panels triggered at first forward pass, first backprop, plateau, vanishing gradient, and inference |

### Phase 2 (planned)

- Backprop visualization with numeric gradient values on edges
- Vanishing gradient detection with visual callout
- Confidence heatmap toggle
- Improved inference mode with animation

### Phase 3 (planned)

- Chain rule tracer: click any weight and see the full ∂L/∂w derivation written out
- Activation derivative explorer with moveable tangent line
- Loss surface contour plot over two selected weights
- Gradient descent path trail
- Test batch panel

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

All neural network math lives in `src/App.jsx` with detailed inline comments. Here is the data flow:

```
initNetwork()               Xavier-initialized weights and zero biases
      ↓
forwardPass()               z = W·x + b → activation(z), layer by layer
      ↓
computeLoss()               Binary cross-entropy: −mean(y·log(p) + (1−y)·log(1−p))
      ↓
backprop()                  Chain rule: δ[L] = ŷ−y, δ[l] = (Wᵀ·δ[l+1]) ⊙ σ'(z[l])
      ↓
updateWeights()             W ← W − lr · ∂L/∂W
      ↓
computeDecisionBoundary()   40×40 forward passes over [0,1]² input space
```

### Key design decisions

**Why XOR?** XOR is not linearly separable, so a network with no hidden layers cannot solve it. This makes it the canonical demonstration of why hidden layers exist.

**Why full-batch gradient descent?** Simpler to understand for educational purposes. With only 4 training examples, mini-batching would add noise without benefit.

**Why BCE + sigmoid on the output?** Binary Cross-Entropy loss paired with a sigmoid output has a numerically convenient gradient: `∂L/∂z_output = ŷ − y`. The sigmoid derivative cancels out, avoiding the saturation problem at the output layer.

**Xavier initialization** sets weight scale to `sqrt(2 / (fan_in + fan_out))`, keeping activation variance roughly constant across layers at the start of training.

---

## Project Structure

```
neural-viz/
├── src/
│   ├── App.jsx        # Everything: math, components, state — single-file artifact
│   ├── main.jsx       # React root mount
│   └── index.css      # Tailwind directives + minimal animation helpers
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

`App.jsx` is organized into numbered sections:

| Section | Contents |
|---|---|
| 1 | Activation functions (ReLU, Tanh, Sigmoid) and their derivatives |
| 2 | XOR dataset |
| 3 | Network initialization (Xavier) |
| 4 | Forward pass |
| 5 | BCE loss |
| 6 | Backpropagation |
| 7 | Gradient descent weight update |
| 8 | One training epoch (full batch) |
| 9 | Decision boundary computation |
| 10 | PyTorch code generator |
| 11 | Color utilities (activation, gradient, boundary) |
| 12 | SVG layout computation |
| — | Components: NetworkGraph, DecisionBoundaryCanvas, ConceptCallout, PyTorchSidebar, App |

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| React | 18 | UI and state |
| Vite | 5 | Dev server and build |
| Tailwind CSS | 3 | Styling |
| Recharts | 2 | Loss curve chart |

No ML libraries are used. All neural network math — forward pass, backpropagation, gradient descent — is implemented from scratch in plain JavaScript.

---

## Accuracy Constraints

This tool makes deliberate simplifications for educational clarity. Each simplification is labeled in both the UI and the code comments.

- **Full-batch gradient descent** — not stochastic or mini-batch
- **No momentum, no Adam** — vanilla SGD only in Phase 1
- **XOR dataset only** — 4 training examples in Phase 1
- **Output always sigmoid** — the output activation is fixed regardless of the hidden activation chosen
- **PyTorch code is explanatory** — the sidebar shows equivalent PyTorch but does not execute

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
