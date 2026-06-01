## Summary

<!-- One or two sentences describing what this PR does. -->

## Type of change

- [ ] Bug fix — corrects a mathematical or visual error
- [ ] Feature — adds a new panel, visualization, or export capability
- [ ] Documentation — improves comments, callouts, or educational text
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

- [ ] Trained to convergence on XOR
- [ ] Ran an Explained Step and confirmed all 4 stages display correctly
- [ ] Animated a forward pass (click-to-predict) and verified neuron values
- [ ] Clicked to predict on the boundary canvas
- [ ] Changed architecture (layers, neurons, activation) and confirmed reinitialize works
- [ ] Verified PyTorch export panel updates on architecture change
- [ ] Opened the Weights tab and confirmed matrix display and JSON export
- [ ] Opened the ∫ Calc tab and confirmed chain rule trace and activation plot
- [ ] Checked build passes: `npm run build`

## Related issues

Closes #

## Notes for reviewers

<!-- Anything the reviewer should pay particular attention to, especially regarding the math. -->
