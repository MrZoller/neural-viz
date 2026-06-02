# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Changed
- **Math core extracted from `App.jsx` into `src/nn/`** — activation functions, network (init/forward/loss/backprop/update/decision boundary), training/evaluation/convergence, and the finite-difference gradient check now live in dedicated, React-free modules (`activations.js`, `datasets.js`, `network.js`, `training.js`) re-exported through `src/nn/index.js`. Behaviour is unchanged; `App.jsx` imports the same functions it previously defined inline.

### Added
- **Optimizers** — choose SGD, Momentum, RMSProp, or Adam from the Training panel. Implemented from scratch in `src/nn/optimizers.js` (matching PyTorch's hyper-parameters) and unit-tested, including a test that Adam beats SGD on XOR at equal lr/epochs. Switching optimizer rebuilds its momentum/moment buffers; changing the learning rate updates it in place. The explained-step preview runs on a throwaway optimizer clone so buffers only commit when an epoch is actually applied. The PyTorch export emits the matching optimizer (`SGD`/`SGD+momentum`/`RMSprop`/`Adam`).
- **Dataset-aware PyTorch export** — the generated `.py` script, `.ipynb` notebook, and the inline code snippet now embed the **active dataset's** actual points (logical gates inline, geometric datasets one point per line) and train on them, with dataset-specific titles, descriptions, and verification. The notebook filename follows the dataset (`neural-viz-<id>.ipynb`). Weights are still randomly re-initialized (stated in the files); the previous "still trains on XOR" disclaimer is removed.
- **Dataset picker** — choose between logical gates (XOR/AND/OR) and generated geometric datasets (circles, moons, spirals, linear, blobs) from the left panel. Geometric sets expose points / noise / seed controls (with a ↻ reshuffle button). Selecting a dataset reinitializes the network and re-runs all training, the decision boundary, and the calculus/audit panels against it. The boundary canvas, verify, audit, ∂w-trace and f(z) panels adapt to large datasets (small dot markers, accuracy/confidence summary, dropdown sample pickers) while keeping the exact per-point tables for the 4-point logical gates.
- **Unit test suite (Vitest)** — 47 tests covering the math core. `npm test` runs them. The centrepiece checks backpropagation against a symmetric finite-difference estimate, fuzzing across 40 random architectures and activation combinations to assert <1e-4 relative gradient error — the same correctness guarantee the UI's ∂w Check tab demonstrates. The fuzz loop is now fully seeded (deterministic, never flaky) and uses smooth activations; ReLU is verified separately with explicit filtering of the `f′(0)=0` kink cases the centered finite-difference cannot model.
- **Dataset generators** (`src/nn/datasets.js`) — logical gates (XOR/AND/OR) plus seeded geometric datasets (circles, moons, spirals, linear, blobs), all normalized into `[0,1]²` so the existing decision-boundary and forward-pass visualizations work unchanged. A `mulberry32` PRNG makes generated datasets deterministic per (points, noise, seed). The math core (`trainOneEpoch`, `evaluateDataset`, `runGradientCheck`) is now parameterized by dataset, defaulting to XOR for backward compatibility. _(UI dataset picker still to come.)_
- **Continuous integration** — GitHub Actions workflow (`.github/workflows/ci.yml`) runs `npm ci`, `npm test`, and `npm run build` on every pull request and on pushes to `main`.

### Planned
- Click-to-draw datasets: place class-0/class-1 points directly on the boundary canvas and train on them
- Optimizers: momentum / RMSProp / Adam with a side-by-side loss-curve comparison
- Gradient flow summary: per-layer average/max/min gradient magnitudes, dead-ReLU count, vanishing-gradient flag
- Loss surface viewer: 2D slice over two selected weights with current-position marker
- Test batch panel: add multiple test points, view predicted class and confidence table

---

## [0.2.0] — 2026-06-01

### Added

**App identity**
- Renamed from "Neural Network Learning Tool" to **Neural Net Playground**; tagline "Make the math visible."
- Browser `<title>`, app header, exported script/notebook headers, and JSON `source` field all updated

**Phase 2 — Backpropagation visualization and interactive training**

- Backprop edge coloring: edges colored by `|∂L/∂w|` magnitude (gray = near-zero → red = large) after every backward pass
- Numeric `∂w` labels toggle: overlay exact gradient values on each edge
- Gradient legend moved from SVG to HTML so it no longer overlaps input-layer neurons
- Convergence auto-stop: two criteria — loss below 0.001, or all 4 XOR points correct with >95% confidence for 50 consecutive epochs; fires a "Converged ✓" callout
- Plateau detection: stops training when loss improvement is less than 0.0005 over 100 epochs; callout suggests reset, Tanh, LR adjustment, or capacity change, with dead-ReLU explanation
- Vanishing-gradient detection: fires when first-layer max gradient is below 1% of global max
- Explained Step mode: 4-stage interactive walkthrough (Forward → Loss → Backward → Update) with Next / Prev / Auto-play controls and a speed slider; highlights the active stage in the network graph
- Confidence heatmap toggle on decision boundary canvas (|p − 0.5| × 2 mapped to amber brightness)
- Click-to-predict: click anywhere on the decision boundary canvas to run a real forward pass and animate activations layer-by-layer
- Math Audit panel: per-sample forward-pass trace with symbolic BCE formula and numeric values
- Finite-difference gradient check (∂w Check tab): verifies backprop gradient against symmetric numerical estimate `[L(w+ε) − L(w−ε)] / 2ε`; auto-pick button selects the weight with the largest `|∂L/∂w|`

**PyTorch export panel** (replaces old sidebar)
- Compact architecture/optimizer/activation summary
- Visual-concept → PyTorch API mapping table
- Copy Script button: copies a complete, runnable `.py` file to clipboard (2 s "✓ Copied!" feedback)
- Export Notebook button: downloads a `.ipynb` file (nbformat v4, 18 cells) covering imports, dataset, model, training loop, loss curve, XOR verification, decision boundary, and custom inference
- Collapsible full PyTorch code block

**Parameters / Weights Inspector** (Weights tab)
- Per-layer weight matrices rendered as color-coded tables: amber = positive, blue = negative, opacity = relative magnitude
- Bias vector chips below each weight matrix
- Storage/PyTorch orientation note per layer: `W[out_feature][in_feature]` matches `nn.Linear.weight`; no transposition needed
- Educational callout explaining weights as learned multipliers and biases as threshold shifters
- LLM analogy callout (collapsible): contextualises parameter count against frontier models without unverified claims; includes architectural disclaimer
- Copy JSON / Download JSON (`neural-viz-params.json`): exports weights, biases, architecture, and training state
- Parameter JSON includes `xor_verification` array (per-point input/expected/predicted/confidence/correct), `convergence_reason` field (`loss_threshold`, `xor_verified`, or both), and an explanatory `note` when converged via confidence criterion with loss still above threshold
- PyTorch weight-loading snippet (collapsible)

**Phase 3 — Calculus panel** (∫ Calc tab)

*∂w Trace — Chain Rule Tracer*
- Select any weight W[layer][j][k] and any of the 4 XOR input samples
- Re-runs a live forward + backprop (not stored averages) to produce exact per-sample deltas
- Symbolic formula box: `∂L/∂w = δⱼ · aₖ` with output-layer BCE+σ shortcut or hidden-layer chain expansion written out
- Three numeric term cards: aₖ (incoming activation), zⱼ + f′(zⱼ) (with dead/saturated inline warning for hidden layers), δⱼ (with derivation note)
- Output-layer f′: labelled as informational only; explains that BCE+sigmoid simplifies δ to ŷ−y so the derivative is already accounted for
- Result box: per-sample gradient vs batch-averaged gradient with update-direction explanation

*f(z) Plot — Activation Function Explorer*
- Layer + neuron dropdowns; XOR sample picker; f′(z) overlay toggle
- Recharts `LineChart` showing f(z) curve (colored by activation type), f′(z) dashed overlay, tangent line segment at current z (±1.5 window), vertical `ReferenceLine`, and filled `ReferenceDot` markers
- Numeric readout: z, f(z), f′(z) from the actual forward pass for the selected sample
- Dead-ReLU callout: "z stays below 0 for all training samples" (per-dataset framing, not per-input)
- Saturation callout for Tanh/Sigmoid when |f′(z)| < 0.05
- Educational notes per activation type covering saturation, dead neurons, vanishing gradients, and the f′(0)=0 convention for ReLU

### Fixed
- Decision boundary y-axis was inverted; corrected so canvas top = x₂ = 1
- Right panel overflow: PyTorch code no longer bled into XOR verify / audit panels
- Network graph auto-sizes to content; removed empty vertical space
- `firstBackprop` callout title was shown after training stopped; now correctly suppressed

### Changed
- Right panel tab bar expanded from 2 tabs (Math Audit, ∂w Check) to 4 (Audit, ∂w Check, Weights, ∫ Calc)
- `xor_solved` in JSON export uses `r.correct` (consistent with XOR Verify panel); was reading undefined field `r.prediction`

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
