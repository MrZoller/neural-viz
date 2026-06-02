import { describe, it, expect } from 'vitest';
import { ACTIVATIONS, computeActivationCurve } from '../activations.js';

// Numerically estimate f'(x) with a centered finite difference so we can check
// each activation's analytical derivative independently of the network.
function numericDeriv(fn, x, eps = 1e-6) {
  return (fn(x + eps) - fn(x - eps)) / (2 * eps);
}

describe('ACTIVATIONS', () => {
  it('exposes relu, tanh and sigmoid with the expected shape', () => {
    for (const key of ['relu', 'tanh', 'sigmoid']) {
      expect(ACTIVATIONS[key]).toBeDefined();
      expect(typeof ACTIVATIONS[key].fn).toBe('function');
      expect(typeof ACTIVATIONS[key].derivative).toBe('function');
    }
  });

  describe('relu', () => {
    const { fn, derivative } = ACTIVATIONS.relu;
    it('clamps negatives to zero and passes positives through', () => {
      expect(fn(-3)).toBe(0);
      expect(fn(0)).toBe(0);
      expect(fn(2.5)).toBe(2.5);
    });
    it('has derivative 0 for z<=0 and 1 for z>0 (f′(0)=0 convention)', () => {
      expect(derivative(-1)).toBe(0);
      expect(derivative(0)).toBe(0);
      expect(derivative(1)).toBe(1);
    });
  });

  describe('tanh', () => {
    const { fn, derivative } = ACTIVATIONS.tanh;
    it('matches Math.tanh', () => {
      expect(fn(0)).toBe(0);
      expect(fn(1)).toBeCloseTo(Math.tanh(1), 12);
    });
    it('derivative agrees with finite differences', () => {
      for (const z of [-2, -0.5, 0, 0.5, 2]) {
        expect(derivative(z)).toBeCloseTo(numericDeriv(fn, z), 5);
      }
    });
  });

  describe('sigmoid', () => {
    const { fn, derivative } = ACTIVATIONS.sigmoid;
    it('maps to (0,1) and equals 0.5 at the origin', () => {
      expect(fn(0)).toBeCloseTo(0.5, 12);
      expect(fn(100)).toBeGreaterThan(0.999);
      expect(fn(-100)).toBeLessThan(0.001);
    });
    it('does not overflow for extreme inputs (clamped exponent)', () => {
      expect(Number.isFinite(fn(10000))).toBe(true);
      expect(Number.isFinite(fn(-10000))).toBe(true);
    });
    it('derivative peaks at 0.25 and agrees with finite differences', () => {
      expect(derivative(0)).toBeCloseTo(0.25, 12);
      for (const z of [-3, -1, 0, 1, 3]) {
        expect(derivative(z)).toBeCloseTo(numericDeriv(fn, z), 5);
      }
    });
  });
});

describe('computeActivationCurve', () => {
  it('returns the requested number of evenly spaced points across the range', () => {
    const pts = computeActivationCurve('tanh', -4, 4, 80);
    expect(pts).toHaveLength(80);
    expect(pts[0].z).toBeCloseTo(-4, 12);
    expect(pts[79].z).toBeCloseTo(4, 12);
    // a and dAdZ are the activation and its derivative at each z.
    expect(pts[40].a).toBeCloseTo(Math.tanh(pts[40].z), 12);
  });

  it('only defines the tangent within ±1.5 of currentZ', () => {
    const pts = computeActivationCurve('sigmoid', -4, 4, 80, 0);
    const near = pts.find(p => Math.abs(p.z) < 0.1);
    const far  = pts.find(p => p.z > 3);
    expect(near.tangent).not.toBeNull();
    expect(far.tangent).toBeNull();
  });
});
