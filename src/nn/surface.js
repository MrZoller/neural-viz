// =============================================================================
// LOSS SURFACE
// =============================================================================
// The true loss landscape has as many dimensions as there are parameters, so it
// can't be drawn directly. What we CAN draw honestly is a 2-D *slice*: pick two
// weights, sweep them over a grid while holding every other parameter fixed at
// its current value, and evaluate the real full-batch BCE loss at each grid
// point. That is exactly what computeLossSurface does — no approximation.
//
// computeDescentPath then records where those same two weights travel during a
// real optimization run. Because the optimizer moves ALL weights at once, the
// path is the true trajectory *projected* onto the two chosen axes — so it may
// leave the static slice's apparent valley (the other dimensions moved too).
// =============================================================================
import { forwardPass, computeLoss } from './network.js';
import { trainOneEpoch } from './training.js';
import { createOptimizer } from './optimizers.js';

// Evaluate the full-batch loss over a gridSize×gridSize sweep of two weights,
// centered on their current values and spanning ±span in each direction.
// coordA / coordB are [layer, out, in] indices into net.weights.
export function computeLossSurface(net, hiddenActivationTypes, dataset, coordA, coordB, opts = {}) {
  const { span = 3, gridSize = 31 } = opts;
  const [la, ja, ka] = coordA;
  const [lb, jb, kb] = coordB;

  const centerA = net.weights[la][ja][ka];
  const centerB = net.weights[lb][jb][kb];

  // Work on a private copy so the caller's network is never mutated.
  const W = net.weights.map(M => M.map(row => row.slice()));
  const targets = dataset.map(d => d.label);

  const axis = (center) =>
    Array.from({ length: gridSize }, (_, i) => center - span + (2 * span) * (i / (gridSize - 1)));
  const aVals = axis(centerA);
  const bVals = axis(centerB);

  const grid = [];
  let min = Infinity, max = -Infinity;
  for (let bi = 0; bi < gridSize; bi++) {
    W[lb][jb][kb] = bVals[bi];
    const rowArr = [];
    for (let ai = 0; ai < gridSize; ai++) {
      W[la][ja][ka] = aVals[ai];
      const preds = dataset.map(d => forwardPass(d.input, W, net.biases, hiddenActivationTypes).activations.at(-1));
      const loss = computeLoss(preds, targets);
      rowArr.push(loss);
      if (loss < min) min = loss;
      if (loss > max) max = loss;
    }
    grid.push(rowArr);
  }

  return { grid, aVals, bVals, center: { a: centerA, b: centerB }, min, max, coordA, coordB, span };
}

// Record the (weightA, weightB) position at every epoch of a real optimization
// run starting from `net`, so it can be overlaid on a surface as a trajectory.
export function computeDescentPath(net, hiddenActivationTypes, dataset, coordA, coordB, lr, steps, optimizerType = 'sgd') {
  const [la, ja, ka] = coordA;
  const [lb, jb, kb] = coordB;
  let weights = structuredClone(net.weights);
  let biases  = structuredClone(net.biases);
  const opt = createOptimizer(optimizerType, lr, { weights, biases });

  const path = [{ a: weights[la][ja][ka], b: weights[lb][jb][kb], loss: null }];
  for (let s = 0; s < steps; s++) {
    const r = trainOneEpoch(weights, biases, hiddenActivationTypes, lr, dataset, opt);
    weights = r.weights;
    biases  = r.biases;
    path.push({ a: weights[la][ja][ka], b: weights[lb][jb][kb], loss: r.loss });
  }
  return path;
}
