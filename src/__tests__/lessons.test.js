import { describe, it, expect } from 'vitest';
import { LESSONS } from '../lessons.js';
import { DATASETS, OPTIMIZERS } from '../nn/index.js';

const TABS = ['audit', 'gradcheck', 'weights', 'calc', 'surface'];

describe('LESSONS content integrity', () => {
  it('has lessons, each with a unique id, title, summary and ≥1 step', () => {
    expect(LESSONS.length).toBeGreaterThan(0);
    const ids = new Set();
    for (const lesson of LESSONS) {
      expect(typeof lesson.id).toBe('string');
      expect(ids.has(lesson.id)).toBe(false);
      ids.add(lesson.id);
      expect(lesson.title.length).toBeGreaterThan(0);
      expect(lesson.summary.length).toBeGreaterThan(0);
      expect(lesson.steps.length).toBeGreaterThan(0);
    }
  });

  it('every step has a title and body', () => {
    for (const lesson of LESSONS) {
      for (const step of lesson.steps) {
        expect(step.title.length).toBeGreaterThan(0);
        expect(step.body.length).toBeGreaterThan(0);
      }
    }
  });

  it('every step.setup references only valid, in-range configuration', () => {
    for (const lesson of LESSONS) {
      for (const step of lesson.steps) {
        const s = step.setup;
        if (!s) continue;
        if (s.datasetId)     expect(DATASETS[s.datasetId]).toBeDefined();
        if (s.optimizer)     expect(OPTIMIZERS[s.optimizer]).toBeDefined();
        if (s.rightPanelTab) expect(TABS).toContain(s.rightPanelTab);
        if (s.lr != null)    { expect(s.lr).toBeGreaterThan(0); expect(s.lr).toBeLessThanOrEqual(1); }
        if (s.neuronsPerLayer) {
          expect(s.neuronsPerLayer.length).toBeGreaterThanOrEqual(1);
          expect(s.neuronsPerLayer.length).toBeLessThanOrEqual(4);   // UI cap: 1–4 hidden layers
          for (const n of s.neuronsPerLayer) {
            expect(n).toBeGreaterThanOrEqual(2);                     // UI cap: 2–8 neurons
            expect(n).toBeLessThanOrEqual(8);
          }
        }
        if (s.activations) {
          // activations must line up with the hidden layers and be known types
          if (s.neuronsPerLayer) expect(s.activations.length).toBe(s.neuronsPerLayer.length);
          for (const a of s.activations) expect(['relu', 'tanh', 'sigmoid']).toContain(a);
        }
      }
    }
  });
});
