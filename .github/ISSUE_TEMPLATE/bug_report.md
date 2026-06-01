---
name: Bug report
about: Something is broken or mathematically wrong
title: ''
labels: bug
assignees: ''
---

## What happened

<!-- Describe the bug clearly. If a computed value looks wrong, say what you expected and what you saw. -->

## Is this a math bug or a visual bug?

- [ ] Math bug — a computed value (activation, loss, gradient) is wrong
- [ ] Visual bug — a value is correct but displayed incorrectly
- [ ] UI bug — a control, animation, or layout doesn't work as expected

## Steps to reproduce

1. Network architecture used (hidden layers, neurons, activation):
2. What you did:
3. What happened:
4. What you expected:

## Verification

For math bugs, if you can, show the manual calculation:

```
# Example: what the forward pass should compute for input [0, 1]
z = W[0] @ [0, 1] + b[0] = ...
a = relu(z) = ...
```

## Environment

- Browser and version:
- OS:
- Node version (if building locally):

## Additional context

<!-- Anything else: screenshots, console errors, etc. -->
