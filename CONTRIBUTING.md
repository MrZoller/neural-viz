# Contributing

Thank you for your interest in contributing to the Neural Network Learning Tool. This project is an educational resource, so the most important quality standard is **mathematical honesty**: every visualization must reflect real computation, not a demo approximation.

---

## Ground Rules

1. **No fake math.** If a value is displayed to the user, it must be computed. If something is simplified or approximated, label it clearly in both the UI and the code comments.
2. **Prefer readable over clever.** This codebase is read by people learning neural networks, not just experienced engineers. Verbose and clear beats compact and opaque.
3. **Comment the why, not the what.** Well-named functions explain themselves. Comments should explain the mathematical reason a step exists, not restate what the code does.
4. **Stay single-file.** All React components and math functions live in `src/App.jsx`. Do not split into multiple component files without discussion — the single-file structure is intentional so learners can read the full implementation in one place.

---

## Development Setup

```bash
git clone https://github.com/MrZoller/neural-viz.git
cd neural-viz
npm install
npm run dev
```

The dev server runs at `http://localhost:5173` with hot module reload.

To verify a production build before submitting a PR:

```bash
npm run build
npm run preview
```

---

## What to Work On

The project follows a phased roadmap. Check the [open issues](https://github.com/MrZoller/neural-viz/issues) for work tagged by phase.

**Good first contributions:**

- Fixing a mathematical error in the existing implementation
- Improving an explanation in a concept callout
- Improving the PyTorch code generator output
- Adding inline code comments where the math is unclear
- Accessibility improvements (keyboard navigation, color-blind-friendly palette)

**Larger contributions — discuss first:**

- Phase 2 or Phase 3 features
- Changing the training dataset beyond XOR
- Adding a second optimizer (Adam, RMSProp)
- Adding a second loss function

Open an issue before starting significant work so we can align on the approach.

---

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`.
2. Make your changes in `src/App.jsx` (and supporting files if needed).
3. Verify the build passes: `npm run build`.
4. Test manually: train to convergence, animate a forward pass, click to infer, change the architecture.
5. Fill out the pull request template.

### Commit message style

Use a short imperative summary line, optionally followed by a blank line and a body:

```
Fix vanishing-gradient detection threshold

Previous threshold (1%) fired too aggressively for sigmoid networks
that were still training. Raised to 2% and added a minimum-epoch
guard so it doesn't fire before the first meaningful gradient update.
```

---

## Code Style

- **No TypeScript** — plain JavaScript for maximum accessibility to learners
- **React hooks only** — no class components
- **Tailwind for all styling** — no inline style objects except where dynamic values require them
- **No extra dependencies** — the only runtime dependencies are React, Recharts, and Tailwind

---

## Mathematical Correctness Checklist

When adding or modifying any math, verify:

- [ ] Forward-pass activation values match manual calculation for a known input
- [ ] Loss decreases over time when training on XOR
- [ ] Decision boundary visually matches the expected XOR pattern after sufficient training
- [ ] Gradient values on edges are non-zero and vary across the network
- [ ] Inference results are consistent with the forward pass shown during training

---

## Questions

Open a [GitHub Discussion](https://github.com/MrZoller/neural-viz/discussions) for anything that doesn't fit an issue — conceptual questions, design ideas, or requests for clarification on the math.
