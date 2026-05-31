## Summary

<!-- One or two sentences describing what this PR does. -->

## Type of change

- [ ] Bug fix — corrects a mathematical or visual error
- [ ] Feature — implements part of the Phase 2 or Phase 3 roadmap
- [ ] Documentation — improves comments, callouts, or the PyTorch sidebar
- [ ] Refactor — restructures code without changing behavior
- [ ] Other:

## Mathematical correctness

<!-- If this PR touches any computed values, confirm: -->

- [ ] Forward-pass activation values match manual calculation for at least one known input
- [ ] Loss decreases over training on XOR
- [ ] Decision boundary updates from actual model weights (not approximated)
- [ ] If gradient values are displayed, they match the backprop computation
- [ ] If inference is shown, it uses the same weights as the training visualization

## Testing

<!-- Describe how you verified this works: -->

- [ ] Trained to convergence on XOR (loss < 0.05)
- [ ] Animated a forward pass and verified neuron values
- [ ] Clicked to predict on the boundary canvas
- [ ] Changed architecture (layers, neurons, activation) and confirmed reinitialize works
- [ ] Verified PyTorch sidebar updates on architecture change
- [ ] Checked build passes: `npm run build`

## Related issues

Closes #

## Notes for reviewers

<!-- Anything the reviewer should pay particular attention to, especially regarding the math. -->
