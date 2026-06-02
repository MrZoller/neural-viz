// =============================================================================
// TRAINING, EVALUATION, CONVERGENCE & GRADIENT CHECK
// =============================================================================
import { XOR_DATA } from './datasets.js';
import { forwardPass, computeLoss, backprop, updateWeights } from './network.js';
import { optimizerStep, createOptimizer } from './optimizers.js';

// -----------------------------------------------------------------------------
// ONE TRAINING EPOCH (FULL BATCH)
// Forward → accumulate gradients → average → update weights.
// `dataset` defaults to XOR for backward compatibility, but any list of
// { input, label } points works.
// -----------------------------------------------------------------------------
export function trainOneEpoch(weights, biases, hiddenActivationTypes, lr, dataset = XOR_DATA, optimizer = null) {
  const L = weights.length;
  const N = dataset.length;

  const totalDW = weights.map(W => W.map(row => row.map(() => 0)));
  const totalDB = weights.map(W => new Array(W.length).fill(0));

  const allPredictions = [];
  const allTargets     = [];
  const allForwardData = [];

  for (const { input, label } of dataset) {
    const { activations, preActivations } = forwardPass(
      input, weights, biases, hiddenActivationTypes
    );
    allPredictions.push(activations[L]);
    allTargets.push(label);
    allForwardData.push({ activations, preActivations });

    const { dWeights, dBiases } = backprop(
      [label], activations, preActivations, weights, hiddenActivationTypes
    );
    for (let l = 0; l < L; l++) {
      for (let j = 0; j < dWeights[l].length; j++) {
        for (let k = 0; k < dWeights[l][j].length; k++) totalDW[l][j][k] += dWeights[l][j][k];
        totalDB[l][j] += dBiases[l][j];
      }
    }
  }

  const avgDW = totalDW.map(W => W.map(row => row.map(v => v / N)));
  const avgDB = totalDB.map(b => b.map(v => v / N));

  // With an optimizer, delegate the update rule (momentum/RMSProp/Adam); the
  // optimizer carries lr in its config. Without one, fall back to plain SGD so
  // existing callers (and the default lr argument) behave exactly as before.
  const { weights: newWeights, biases: newBiases } = optimizer
    ? optimizerStep(optimizer, weights, biases, avgDW, avgDB)
    : updateWeights(weights, biases, avgDW, avgDB, lr);

  return {
    weights:     newWeights,
    biases:      newBiases,
    loss:        computeLoss(allPredictions, allTargets),
    avgDW,
    avgDB,
    allForwardData,
  };
}

// -----------------------------------------------------------------------------
// OPTIMIZER COMPARISON
// Train one fresh copy of the SAME starting network with each optimizer for a
// fixed number of epochs, recording the loss curve of each. Because every run
// shares the identical initial weights, dataset, architecture and learning
// rate, the only variable is the update rule — so the overlaid curves are a
// fair head-to-head ("watch Adam escape the plateau SGD gets stuck on").
// -----------------------------------------------------------------------------
export function runOptimizerComparison(net, hiddenActivationTypes, dataset, optimizerTypes, lr, epochs) {
  return optimizerTypes.map(type => {
    let weights = structuredClone(net.weights);
    let biases  = structuredClone(net.biases);
    const opt = createOptimizer(type, lr, { weights, biases });
    const losses = [];
    for (let e = 0; e < epochs; e++) {
      const r = trainOneEpoch(weights, biases, hiddenActivationTypes, lr, dataset, opt);
      weights = r.weights;
      biases  = r.biases;
      losses.push(r.loss);
    }
    return { type, losses, finalLoss: losses.length ? losses[losses.length - 1] : null };
  });
}

// -----------------------------------------------------------------------------
// DATASET EVALUATION
// Per-point forward pass with predicted class, confidence and sample loss.
// -----------------------------------------------------------------------------
export function evaluateDataset(weights, biases, hiddenActivationTypes, dataset = XOR_DATA) {
  const L = weights.length;
  return dataset.map(({ input, label }) => {
    const { activations, preActivations } = forwardPass(
      input, weights, biases, hiddenActivationTypes
    );
    const rawOutput      = activations[L][0];
    const predictedClass = rawOutput > 0.5 ? 1 : 0;
    const confidence     = Math.abs(rawOutput - 0.5) * 2;
    const correct        = predictedClass === label;
    const eps            = 1e-12;
    const p              = Math.max(eps, Math.min(1 - eps, rawOutput));
    const sampleLoss     = -(label * Math.log(p) + (1 - label) * Math.log(1 - p));
    return { input, label, rawOutput, predictedClass, confidence, correct, sampleLoss,
             activations, preActivations };
  });
}

// Backward-compatible alias — XOR is just the default dataset.
export const evaluateXOR = evaluateDataset;

// -----------------------------------------------------------------------------
// CONVERGENCE / STOP CONDITIONS
// -----------------------------------------------------------------------------
export const CONVERGENCE_LOSS_THRESHOLD      = 0.001;
export const CONVERGENCE_CONSECUTIVE_EPOCHS  = 50;
export const CONVERGENCE_CONFIDENCE          = 0.95;
export const PLATEAU_PATIENCE                = 100;
export const MIN_IMPROVEMENT                 = 0.0005;
export const PLATEAU_MIN_LOSS                = 0.05;

export function checkConvergence(loss, xorResults, consecutiveCorrect) {
  if (loss < CONVERGENCE_LOSS_THRESHOLD) {
    return { converged: true, reason: `Loss dropped below ${CONVERGENCE_LOSS_THRESHOLD} (current: ${loss.toFixed(6)})` };
  }
  const allHighConf = xorResults.every(r => r.correct && r.confidence > CONVERGENCE_CONFIDENCE);
  if (allHighConf && consecutiveCorrect >= CONVERGENCE_CONSECUTIVE_EPOCHS) {
    return { converged: true, reason: `All ${xorResults.length} points correctly classified with >${(CONVERGENCE_CONFIDENCE*100).toFixed(0)}% confidence for ${CONVERGENCE_CONSECUTIVE_EPOCHS} consecutive epochs` };
  }
  return { converged: false, reason: '' };
}

// -----------------------------------------------------------------------------
// GRADIENT CHECK  (finite-difference verification)
//
// Verifies that the backprop gradient ∂L/∂w for a single weight matches the
// numerical estimate computed via the symmetric (centered) finite-difference:
//
//   ∂L/∂w ≈ [L(w + ε) − L(w − ε)] / (2ε)      O(ε²) truncation error
//
// This is the standard numerical gradient check used to validate autograd
// implementations. If the analytical and numerical gradients agree to within
// ~1e-4 relative error, the backprop computation is almost certainly correct.
//
// ε = 1e-4 balances two opposing concerns:
//   • Too large → the linear approximation breaks down (quadratic error term)
//   • Too small → catastrophic cancellation in float64 (significant bits lost
//     when subtracting nearly-equal L+ and L−)
//
// Gradients are averaged over all 4 XOR samples (full-batch), matching the
// convention used by trainOneEpoch.
//
// PyTorch equivalent: torch.autograd.gradcheck(model, inputs, eps=1e-4)
// -----------------------------------------------------------------------------
export function runGradientCheck(weights, biases, hiddenActivationTypes, l, j, k, epsilon = 1e-4, dataset = XOR_DATA) {
  const N = dataset.length;
  const L = weights.length;

  // ① Analytical gradient — backprop averaged over all samples in the dataset.
  //    This is the exact same computation used during training; we're just
  //    isolating one scalar entry dWeights[l][j][k].
  let totalDW = 0;
  for (const { input, label } of dataset) {
    const { activations, preActivations } = forwardPass(
      input, weights, biases, hiddenActivationTypes
    );
    const { dWeights } = backprop(
      [label], activations, preActivations, weights, hiddenActivationTypes
    );
    totalDW += dWeights[l][j][k];
  }
  const backpropGrad = totalDW / N;

  // ② Numerical gradient — perturb only W[l][j][k] by ±ε, recompute full loss.
  const perturbed = (sign) =>
    weights.map((W, li) =>
      W.map((row, ji) =>
        row.map((w, ki) => (li === l && ji === j && ki === k) ? w + sign * epsilon : w)
      )
    );

  const evalLoss = (perturbedWeights) => {
    const preds = dataset.map(({ input }) => {
      const { activations } = forwardPass(input, perturbedWeights, biases, hiddenActivationTypes);
      return activations[L];
    });
    return computeLoss(preds, dataset.map(d => d.label));
  };

  const lossPlus  = evalLoss(perturbed(+1));
  const lossMinus = evalLoss(perturbed(-1));
  const fdGrad    = (lossPlus - lossMinus) / (2 * epsilon);

  const absError = Math.abs(backpropGrad - fdGrad);
  // Relative error normalised by the sum of magnitudes (avoids div-by-zero
  // when both gradients are essentially zero — which is correct agreement).
  const relError = absError / Math.max(Math.abs(backpropGrad) + Math.abs(fdGrad), 1e-10);

  return {
    backpropGrad, fdGrad, absError, relError,
    lossPlus, lossMinus, epsilon,
    currentWeight: weights[l][j][k],
  };
}
