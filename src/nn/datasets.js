// =============================================================================
// DATASETS
// =============================================================================
// Every dataset is a list of { input: [x1, x2], label: 0 | 1 } where the inputs
// live in [0,1]². Keeping all datasets inside the unit square means the existing
// decision-boundary canvas (which samples [0,1]²), the forward-pass animation,
// and the network-graph input neurons all keep working unchanged regardless of
// which dataset is selected.
// =============================================================================

// XOR is not linearly separable, so a single-layer network cannot solve it.
// This makes it the minimal demonstration of why hidden layers and nonlinear
// activations exist. Kept as a named export for backward compatibility.
export const XOR_DATA = [
  { input: [0, 0], label: 0 },
  { input: [0, 1], label: 1 },
  { input: [1, 0], label: 1 },
  { input: [1, 1], label: 0 },
];

const AND_DATA = [
  { input: [0, 0], label: 0 },
  { input: [0, 1], label: 0 },
  { input: [1, 0], label: 0 },
  { input: [1, 1], label: 1 },
];

const OR_DATA = [
  { input: [0, 0], label: 0 },
  { input: [0, 1], label: 1 },
  { input: [1, 0], label: 1 },
  { input: [1, 1], label: 1 },
];

// -----------------------------------------------------------------------------
// Seeded PRNG (mulberry32) — deterministic so a given (dataset, points, noise,
// seed) always produces the same points. Makes generated datasets reproducible
// and shareable, and keeps the test suite stable.
// -----------------------------------------------------------------------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp01 = v => Math.max(0, Math.min(1, v));

// Box–Muller standard normal from a uniform RNG.
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// -----------------------------------------------------------------------------
// Generators — each returns points normalized into [0,1]².
// `points` is the total across both classes (split evenly); `noise` is the
// std-dev of jitter as a fraction of the unit square; `seed` drives the PRNG.
// -----------------------------------------------------------------------------

// Concentric rings: inner disc = class 0, outer ring = class 1.
function genCircles(points, noise, seed) {
  const rng = mulberry32(seed);
  const out = [];
  const per = Math.floor(points / 2);
  for (let cls = 0; cls < 2; cls++) {
    const baseR = cls === 0 ? 0.0 : 0.38;
    const spanR = cls === 0 ? 0.16 : 0.10;
    for (let i = 0; i < per; i++) {
      const ang = rng() * 2 * Math.PI;
      const r = baseR + rng() * spanR;
      const x = 0.5 + r * Math.cos(ang) + gaussian(rng) * noise;
      const y = 0.5 + r * Math.sin(ang) + gaussian(rng) * noise;
      out.push({ input: [clamp01(x), clamp01(y)], label: cls });
    }
  }
  return out;
}

// Two interleaving half-moons.
function genMoons(points, noise, seed) {
  const rng = mulberry32(seed);
  const out = [];
  const per = Math.floor(points / 2);
  for (let i = 0; i < per; i++) {
    const t = (i / (per - 1 || 1)) * Math.PI;
    // Upper moon (class 0)
    let x = 0.5 + 0.4 * Math.cos(t);
    let y = 0.6 + 0.4 * Math.sin(t);
    out.push({
      input: [clamp01(x + gaussian(rng) * noise), clamp01(y + gaussian(rng) * noise)],
      label: 0,
    });
    // Lower moon (class 1), shifted and flipped
    x = 0.5 + 0.4 * Math.cos(t) + 0.2 * 0.4;
    y = 0.4 - 0.4 * Math.sin(t);
    out.push({
      input: [clamp01(x + gaussian(rng) * noise), clamp01(y + gaussian(rng) * noise)],
      label: 1,
    });
  }
  return out;
}

// Two-arm Archimedean spiral.
function genSpiral(points, noise, seed) {
  const rng = mulberry32(seed);
  const out = [];
  const per = Math.floor(points / 2);
  for (let cls = 0; cls < 2; cls++) {
    for (let i = 0; i < per; i++) {
      const frac = i / (per - 1 || 1);
      const r = 0.05 + frac * 0.42;
      const ang = frac * 2.5 * Math.PI + cls * Math.PI;
      const x = 0.5 + r * Math.cos(ang) + gaussian(rng) * noise;
      const y = 0.5 + r * Math.sin(ang) + gaussian(rng) * noise;
      out.push({ input: [clamp01(x), clamp01(y)], label: cls });
    }
  }
  return out;
}

// Linearly separable: split by a diagonal with a margin gap.
function genLinear(points, noise, seed) {
  const rng = mulberry32(seed);
  const out = [];
  for (let i = 0; i < points; i++) {
    const x = rng();
    const y = rng();
    const s = x + y; // diagonal score in [0,2]
    if (Math.abs(s - 1) < 0.18) { i--; continue; } // carve a margin
    const label = s > 1 ? 1 : 0;
    out.push({
      input: [clamp01(x + gaussian(rng) * noise), clamp01(y + gaussian(rng) * noise)],
      label,
    });
  }
  return out;
}

// Two Gaussian blobs on a diagonal.
function genBlobs(points, noise, seed) {
  const rng = mulberry32(seed);
  const out = [];
  const per = Math.floor(points / 2);
  const centers = [[0.3, 0.3], [0.7, 0.7]];
  for (let cls = 0; cls < 2; cls++) {
    for (let i = 0; i < per; i++) {
      const [cx, cy] = centers[cls];
      const x = cx + gaussian(rng) * (0.07 + noise);
      const y = cy + gaussian(rng) * (0.07 + noise);
      out.push({ input: [clamp01(x), clamp01(y)], label: cls });
    }
  }
  return out;
}

// -----------------------------------------------------------------------------
// Registry. Logical gates are fixed 4-point truth tables (no options); the
// geometric datasets are generated from points/noise/seed.
// -----------------------------------------------------------------------------
export const DATASETS = {
  xor:     { id: 'xor',     label: 'XOR',     kind: 'logical',  description: 'Exclusive OR — not linearly separable; the classic reason hidden layers exist.', generate: () => XOR_DATA },
  and:     { id: 'and',     label: 'AND',     kind: 'logical',  description: 'Logical AND — linearly separable; one layer suffices.',                          generate: () => AND_DATA },
  or:      { id: 'or',      label: 'OR',      kind: 'logical',  description: 'Logical OR — linearly separable; one layer suffices.',                           generate: () => OR_DATA },
  circles: { id: 'circles', label: 'Circles', kind: 'geometric', description: 'Concentric rings — inner disc vs outer ring; needs a curved boundary.',         generate: genCircles },
  moons:   { id: 'moons',   label: 'Moons',   kind: 'geometric', description: 'Two interleaving half-moons — the canonical nonlinear toy problem.',            generate: genMoons },
  spiral:  { id: 'spiral',  label: 'Spirals', kind: 'geometric', description: 'Two intertwined spiral arms — hard; rewards more capacity and epochs.',         generate: genSpiral },
  linear:  { id: 'linear',  label: 'Linear',  kind: 'geometric', description: 'Linearly separable blobs split by a diagonal margin.',                          generate: genLinear },
  blobs:   { id: 'blobs',   label: 'Blobs',   kind: 'geometric', description: 'Two Gaussian clusters — easy; good for sanity-checking training.',              generate: genBlobs },
};

export const DEFAULT_DATASET_OPTS = { points: 120, noise: 0.03, seed: 42 };

// Build the point list for a dataset id. Logical gates ignore the options.
export function makeDataset(id, opts = {}) {
  const spec = DATASETS[id] || DATASETS.xor;
  const { points, noise, seed } = { ...DEFAULT_DATASET_OPTS, ...opts };
  return spec.generate(points, noise, seed);
}
