// =============================================================================
// DATASETS
// =============================================================================
// XOR is not linearly separable, so a single-layer network cannot solve it.
// This makes it the minimal demonstration of why hidden layers and nonlinear
// activations exist.
export const XOR_DATA = [
  { input: [0, 0], label: 0 },
  { input: [0, 1], label: 1 },
  { input: [1, 0], label: 1 },
  { input: [1, 1], label: 0 },
];
