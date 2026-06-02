// =============================================================================
// NEURAL-NETWORK MATH CORE — public surface
// Every number the UI shows is computed by these from-scratch functions.
// No ML libraries; see individual modules for the math conventions.
// =============================================================================
export { ACTIVATIONS, computeActivationCurve } from './activations.js';
export { XOR_DATA } from './datasets.js';
export {
  initNetwork,
  forwardPass,
  computeLoss,
  backprop,
  updateWeights,
  computeDecisionBoundary,
} from './network.js';
export {
  trainOneEpoch,
  evaluateXOR,
  checkConvergence,
  runGradientCheck,
  CONVERGENCE_LOSS_THRESHOLD,
  CONVERGENCE_CONSECUTIVE_EPOCHS,
  CONVERGENCE_CONFIDENCE,
  PLATEAU_PATIENCE,
  MIN_IMPROVEMENT,
  PLATEAU_MIN_LOSS,
} from './training.js';
