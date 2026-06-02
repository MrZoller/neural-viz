// =============================================================================
// ACTIVATION FUNCTIONS
// Each entry has: fn (the activation), derivative (its derivative w.r.t. input).
// These are the exact functions PyTorch uses under the hood.
// =============================================================================
export const ACTIVATIONS = {
  relu: {
    label: 'ReLU',
    fn: x => Math.max(0, x),
    derivative: x => (x > 0 ? 1 : 0),
    color: '#60a5fa',
  },
  tanh: {
    label: 'Tanh',
    fn: x => Math.tanh(x),
    derivative: x => 1 - Math.tanh(x) ** 2,
    color: '#a78bfa',
  },
  sigmoid: {
    label: 'Sigmoid',
    // σ(x) = 1/(1+e^−x). Derivative σ(x)·(1−σ(x)) peaks at 0.25 — causes
    // vanishing gradients in deep nets when used in hidden layers.
    fn: x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
    derivative: x => {
      const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
      return s * (1 - s);
    },
    color: '#f472b6',
  },
};

// Activation curve + derivative + tangent for Recharts.
// Returns numPoints objects: { z, a, dAdZ, tangent }
// tangent is only defined within ±1.5 of currentZ (null elsewhere → gap in chart).
export function computeActivationCurve(activType, zMin = -4, zMax = 4, numPoints = 80, currentZ = null) {
  const fn    = ACTIVATIONS[activType].fn;
  const deriv = ACTIVATIONS[activType].derivative;
  return Array.from({ length: numPoints }, (_, i) => {
    const z       = zMin + (i / (numPoints - 1)) * (zMax - zMin);
    const tangent = (currentZ !== null && Math.abs(z - currentZ) <= 1.5)
      ? fn(currentZ) + deriv(currentZ) * (z - currentZ)
      : null;
    return { z, a: fn(z), dAdZ: deriv(z), tangent };
  });
}
