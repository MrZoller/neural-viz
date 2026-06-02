# Neural Net Playground — User Guide

**Make the math visible.**

This guide walks you through every feature of the Neural Net Playground: configuring networks, training them, interpreting every panel, and exporting your work to PyTorch. No prior deep-learning experience is needed — the app is designed to make the mathematics observable at every step.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Left Panel — Configuration](#left-panel--configuration)
   - [Dataset](#dataset)
   - [Architecture](#architecture)
   - [Training Controls](#training-controls)
4. [Center Panel — Visualizations](#center-panel--visualizations)
   - [Network Graph](#network-graph)
   - [Decision Boundary](#decision-boundary)
   - [Loss Curve](#loss-curve)
5. [Right Panel — Math Tools](#right-panel--math-tools)
   - [Audit Tab](#audit-tab)
   - [∂w Check Tab](#w-check-tab)
   - [Weights Tab](#weights-tab)
   - [∫ Calc Tab](#-calc-tab)
   - [Surface Tab](#surface-tab)
6. [PyTorch Export](#pytorch-export)
7. [Guided Lessons](#guided-lessons)
8. [Tips & Common Patterns](#tips--common-patterns)

---

## Getting Started

**Live demo:** [https://mrzoller.github.io/neural-viz/](https://mrzoller.github.io/neural-viz/) — runs entirely in your browser, nothing to install.

### Prerequisites

To run locally:

- Node.js 18 or later
- npm 9 or later

### Running locally

```bash
git clone https://github.com/MrZoller/neural-viz.git
cd neural-viz
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. The full app loads in the browser — there is no server-side compute. Every number you see is calculated in JavaScript from the raw weight matrices.

---

## Interface Overview

![Full app overview](screenshots/01-overview.png)

The interface is divided into three columns:

| Column | Width | Purpose |
|---|---|---|
| **Left** | ~260 px | Dataset, architecture, and training controls |
| **Center** | Flexible | Network graph, decision boundary canvas, and loss curve |
| **Right** | ~320 px | PyTorch mapping, XOR verification, and math-detail tabs |

A status bar at the top shows training state (Not Started / Training / Paused / Converged), current epoch, and current loss.

---

## Left Panel — Configuration

![Left config panel](screenshots/08-config-panel.png)

### Dataset

**Logical gates** (4 data points each):
- **XOR** — the classic non-linearly-separable gate; the default starting point
- **AND** — both inputs must be 1
- **OR** — at least one input must be 1

**Geometric datasets** (generated, normalized to `[0,1]²`):
- **Circles** — concentric rings; tests radial separation
- **Moons** — two interleaved crescents; moderate difficulty
- **Spirals** — two interleaved spirals; the hardest dataset here
- **Linear** — linearly separable; a sanity check
- **Blobs** — two Gaussian clusters; easy with any network

For geometric datasets three extra sliders appear: **Points** (number of training samples), **Noise** (label noise fraction), and **Seed** (use ↻ to reshuffle with a new seed while keeping all other settings).

> Switching dataset resets the network, restarts training state, and re-runs the decision boundary.

### Architecture

**Hidden Layers** — use **+** / **−** to add or remove layers (1–4 supported).

For each layer:
- **Neurons** — number of hidden units in that layer (2–8)
- **Activation** — ReLU, Tanh, or Sigmoid applied to the pre-activation `z`

The output layer always uses a Sigmoid activation (binary classification). Changing any architecture setting immediately reinitializes the network with new Xavier weights.

**Xavier initialization** sets each weight to a random value scaled by `√(2 / (fan_in + fan_out))`, which keeps activation variance roughly constant at initialization and reduces early vanishing/exploding gradients.

### Training Controls

![Training controls](screenshots/04-training-controls.png)

| Button | What it does |
|---|---|
| **▶ Train** | Starts continuous full-batch gradient descent. The button becomes **⏸ Pause** while running. |
| **Step 1 Epoch** | Runs exactly one epoch (one full forward + backward + update pass) and pauses. |
| **Explained Step ←→** | Opens the 4-stage interactive walkthrough (see below). |
| **Forward Pass ▶** | Animates one forward pass without updating weights. |
| **⚖ Compare Optimizers** | Opens a side-by-side chart of SGD, Momentum, RMSProp, and Adam training from the same starting point. |
| **Reset** | Reinitializes with a new random seed (same architecture, same dataset). |

**Optimizer and learning-rate controls** sit above the buttons:

- **Optimizer** — SGD, Momentum, RMSProp, or Adam (implemented from scratch, matching PyTorch's default hyperparameters)
- **LR** — learning rate slider; current value shown numerically
- **Max Epochs** — presets (1k / 5k / 10k / 50k) cap how long **Train** runs before auto-stopping

**Convergence auto-stop** fires when either:
- BCE loss drops below 0.001, or
- Every point in the active dataset is classified with > 95% confidence for 50 consecutive epochs

A plateau detector warns when loss improvement stalls but convergence hasn't been reached.

#### Explained Step mode

Click **Explained Step ←→** to enter the 4-stage walkthrough for one training epoch:

| Stage | What you see |
|---|---|
| **Forward** | Activations animate left-to-right through the network; neurons light up in sequence |
| **Loss** | The BCE loss formula appears with real numeric values |
| **Backward** | Gradients animate right-to-left; edges pulse by `\|∂L/∂w\|` |
| **Update** | Weights shift; the decision boundary re-renders with the new values |

Use **‹ Prev** / **Next ›** to step manually, or toggle **Auto-play** for timed advancement. Press **Done** to exit back to normal mode.

---

## Center Panel — Visualizations

### Network Graph

![Network graph after training](screenshots/02-network-graph.png)

The SVG network diagram shows the live state of the model:

- **Neuron color** — intensity encodes the output activation magnitude after the last forward pass. Brighter = larger `|a|`.
- **Edge color** — after a backward pass, edges are colored by `|∂L/∂w|` (gradient magnitude). The top-8 edges by gradient magnitude are labeled with their numeric values. Blue edges carry larger gradients; pale/grey edges carry near-zero gradients.
- **Labels on neurons** — show the layer name (Input, Hidden n, Output) and neuron index.

> During the Explained Step walkthrough, neurons and edges animate in sequence to illustrate the direction of computation.

### Decision Boundary

![Decision boundary canvas](screenshots/03-decision-boundary.png)

The canvas runs a real forward pass for every point in a 40×40 grid over `[0,1]²` input space and colors each cell by the network's output probability:

- **Blue region** — network predicts Class 0 (probability < 0.5)
- **Orange region** — network predicts Class 1 (probability > 0.5)
- **Color intensity** — encodes confidence; the boundary line is where probability = 0.5

Training data points are plotted as dots on the canvas. Click **Class ▲** in the top-right of the canvas to toggle between class-color mode and **confidence heatmap** mode, where brightness encodes `|p − 0.5| × 2` (bright = more confident).

**Click anywhere** on the canvas to run inference at that point. The network animates a forward pass and shows the predicted class and probability.

### Loss Curve

The Recharts loss plot sits to the right of the decision boundary and updates every epoch. The y-axis is BCE loss; the x-axis is epoch count. Watch for:

- A smooth, rapid descent → learning is working
- A plateau → the optimizer may need a higher learning rate, more capacity, or a reset
- Loss oscillating → learning rate may be too high

---

## Right Panel — Math Tools

The right panel is split into three sections stacked vertically: the **PyTorch Mapping** block, the **XOR Verify / dataset summary**, and a **tabbed math panel**.

### Audit Tab

The default tab. Shows a forward-pass trace for the active dataset — per-sample predictions and the symbolic BCE formula with real numeric values filled in. Run at least one training step to populate it.

### ∂w Check Tab

Verifies that the backpropagation code is correct by comparing the analytical gradient `∂L/∂w` to a symmetric finite-difference estimate:

```
[L(w + ε) − L(w − ε)] / 2ε
```

- **Auto-pick max |∂w|** jumps to the weight with the largest current gradient — a good place to check because large-gradient weights have the most visible numeric disagreement when broken.
- Click **Run** to execute the check on the selected weight.

A relative error below ~1 × 10⁻⁴ means backprop is correct. The app's 69-unit test suite runs this same check across 40 randomly generated architectures and activation combinations.

### Weights Tab

![Weights tab](screenshots/05-weights-tab.png)

Shows every weight matrix and bias vector in the network, color-coded by sign and magnitude:

- **Amber** — positive weights
- **Blue** — negative weights
- **Intensity** — encodes magnitude; white/grey ≈ near zero

The weight layout matches PyTorch's `nn.Linear.weight` convention: `W[out_feature][in_feature]`. No transposition is needed when loading these into PyTorch.

**Export options:**

| Button | Output |
|---|---|
| **Copy JSON** | Copies `neural-viz-params.json` to clipboard — weights, biases, architecture, training state, per-point results, and convergence reason |
| **Download JSON** | Saves the same payload as a file |
| **PyTorch: load weights** | Expands a collapsible code snippet that loads the JSON into a `torch.nn.Sequential` model |

### ∫ Calc Tab

![Calculus tab](screenshots/06-calculus-tab.png)

Two sub-tools toggled by the **∂w Trace** / **f(z) Plot** selector:

#### ∂w Trace — Chain Rule Tracer

Select any weight `W[layer][j][k]` and any input sample. The panel re-runs a live forward + backprop and shows:

- **Symbolic formula**: `∂L/∂w = δⱼ · aₖ`
- **Per-term breakdown**: `aₖ` (incoming activation), `zⱼ`, `f′(zⱼ)`, `δⱼ`
- **Per-sample vs batch-averaged gradient**: clarifies that training uses the mean gradient over all samples
- **Dead ReLU / saturation warnings** when applicable

Use the **Auto-pick max |∂w|** button to jump to the weight currently receiving the strongest update signal.

#### f(z) Plot — Activation Function Explorer

Select any hidden neuron and input sample. The Recharts plot shows:

- `f(z)` curve over a range of pre-activation values
- `f′(z)` derivative overlay (toggle with the **Show f′(z)** button)
- A tangent line at the current `z` value
- Reference markers for `z`, `f(z)`, and `f′(z)`
- **Saturation callout** (Sigmoid / Tanh) when `|z|` is large and the derivative is near zero
- **Dead ReLU callout** when `z ≤ 0` and the gradient cannot flow

### Surface Tab

![Loss surface tab](screenshots/07-surface-tab.png)

Plots a 2-D slice of the loss landscape over two selectable weights, all other weights held at their current values.

- **Color** — emerald = low loss, red = high loss (real BCE values)
- **Pink dot** — marks the current position of those two weights in the landscape
- **↘ Trace descent path** — runs the current optimizer from the starting weights and projects its trajectory onto this 2-D slice as an amber line

Controls:
- **W axis A / W axis B** dropdowns — select which two weights to sweep
- **Span** — the range swept around each weight's current value
- **Zoom** toggle — enlarges the surface view

> Because all other weights move during real training, the descent path can wander off the static valley shown. This illustrates how high-dimensional the true loss surface is.

---

## PyTorch Export

![PyTorch export panel](screenshots/09-pytorch-export.png)

The top of the right panel always shows a **PyTorch Mapping** summary matching the current architecture:

- Architecture string (e.g. `2·4·4·1`)
- `nn.Linear` layer sizes and activations
- Matching PyTorch loss function and optimizer constructor with your current settings

Two export buttons are available below:

| Button | Output |
|---|---|
| **Copy Script** | Copies a complete, runnable `.py` file to the clipboard — includes imports, model definition, the active dataset's training points, training loop, loss curve plot, and inference |
| **Notebook** | Downloads a `.ipynb` notebook (nbformat v4, 18 cells) covering imports, model definition, training, loss curve, XOR verification, decision boundary, and inference |

A **Full PyTorch code** collapsible shows the same script inline.

> **Note:** The exported code reinitializes weights randomly. To export the trained weight values, use the **Weights** tab's **Download JSON** button and the provided PyTorch loading snippet.

---

## Guided Lessons

![Guided lessons panel](screenshots/10-lessons.png)

Click **📚 Lessons** in the top-right of the header to open the guided tour player. The panel is non-modal — you can interact with all controls while a lesson is open.

Four lessons are available:

### Solving XOR
*Why a hidden layer is needed, and watching the boundary bend.*

Walks through the XOR problem, a single Forward Pass animation, training until convergence, and interpreting the Audit and ∂w Check tabs. Recommended as a first lesson.

**Setup applied:** XOR dataset, 2→2→1 network with Tanh, SGD, LR 0.1.

### Capacity & Dead ReLU
*How too few ReLU neurons can stall, and why width helps.*

Demonstrates how a 2-neuron ReLU network can get stuck (dead units) and how widening to 4 neurons reduces that risk. Ends with the f(z) Plot to visualize a dead neuron directly.

**Setup applied:** XOR, 2 then 4 ReLU neurons, SGD, LR 0.1.

### Optimizers Race
*On a hard dataset, see Adam and Momentum outpace plain SGD.*

Switches to Spirals with a 6-6 Tanh architecture, opens the Compare Optimizers overlay, and trains with Adam to show the difference in convergence speed on a difficult dataset.

**Setup applied:** Spirals, 2 hidden layers of 6 Tanh neurons, Adam, LR 0.05.

### The Loss Landscape
*Read a 2-D slice of the loss surface and trace a descent path.*

Opens the Surface tab on XOR and uses "Trace descent path" to show the optimizer trajectory projected onto two weights. Illustrates the difference between a 2-D slice and the true high-dimensional landscape.

**Setup applied:** XOR, 3-neuron Tanh hidden layer, Surface tab active.

#### Navigating a lesson

- **Next ›** advances to the next step (and applies its setup automatically)
- **‹ Back** returns to the previous step
- **← Lesson menu** returns to the lesson list without closing the panel
- **✕** closes the lessons panel entirely

---

## Tips & Common Patterns

**Network won't converge on XOR?**
Try clicking **Reset** a few times. Xavier initialization is random, and occasionally a bad seed lands in a poor basin. With ReLU, dead neurons are a common culprit — switch to Tanh or increase neuron count to reduce the chance.

**Loss plateaus early on Spirals or Moons?**
Use Adam with LR 0.05 and at least 2 hidden layers of 6+ neurons. Click **⚖ Compare Optimizers** to see all four optimizers training in parallel from the same starting point.

**Want to understand a specific gradient?**
Click **Step 1 Epoch** once (so there are computed gradients), then open the **∂w Check** tab and click **Auto-pick max |∂w|** to jump to the weight with the largest gradient. Then open **∫ Calc → ∂w Trace** for a per-term symbolic breakdown of that gradient.

**Want to see the geometry of the loss surface?**
Train the network partially (a few hundred epochs), then open **Surface → ↘ Trace descent path**. You'll see both the current landscape and the optimizer's actual trajectory through it.

**Exporting to PyTorch?**
Use **Copy Script** or **Notebook** for a fresh runnable experiment. If you want to continue fine-tuning the exact weights you trained here, use **Weights → Download JSON** and paste in the load snippet from **PyTorch: load weights**.

**Dead ReLU neurons?**
Open **∫ Calc → f(z) Plot** and step through each hidden neuron. If `z ≤ 0` for all four XOR inputs, that neuron is dead and contributes nothing to the forward pass. Solutions: Reset (new random seed), switch to Tanh or Sigmoid for that layer, or add more neurons.

**Checking if the math is correct?**
Open **∂w Check**, pick any weight, and click **Run**. A relative error below ~1×10⁻⁴ means the backprop gradient matches the finite-difference estimate. This is the same numerical test the 69-unit Vitest suite runs on every commit.

---

*For architecture documentation, tech stack details, and the math behind each module, see the [README](../README.md).*
