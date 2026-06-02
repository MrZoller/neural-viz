// =============================================================================
// GUIDED LESSONS
// =============================================================================
// A lesson is an ordered list of steps. Each step has explanatory text and an
// optional `setup` that configures the playground (dataset, architecture,
// optimizer, learning rate, which right-panel tab to show, whether to open the
// optimizer-comparison overlay). The lesson player applies a step's setup when
// the user navigates to it, then asks them to drive the real controls — so
// every lesson is backed by the same from-scratch math as the rest of the app.
//
// `setup` fields (all optional):
//   datasetId        — one of DATASETS
//   neuronsPerLayer  — array; its length sets the number of hidden layers
//   activations      — array (same length as neuronsPerLayer)
//   optimizer        — one of OPTIMIZERS
//   lr               — learning rate
//   rightPanelTab    — 'audit' | 'gradcheck' | 'weights' | 'calc' | 'surface'
//   openCompare      — true to pop the optimizer-comparison overlay
// =============================================================================

export const LESSONS = [
  {
    id: 'xor-bend',
    title: 'Solving XOR',
    summary: 'Why a hidden layer is needed, and watching the boundary bend.',
    steps: [
      {
        title: 'The problem',
        body: 'XOR outputs 1 when **exactly one** input is 1. Its two classes can’t be split by a single straight line, so the network needs a hidden layer to bend the boundary. We’ve set up a 2→2→1 network with Tanh on XOR.',
        setup: { datasetId: 'xor', neuronsPerLayer: [2], activations: ['tanh'], optimizer: 'sgd', lr: 0.1, rightPanelTab: 'audit' },
      },
      {
        title: 'One forward pass',
        body: 'Click **Forward Pass ▶** in the left panel. Watch activations flow left→right; the output neuron is the network’s predicted probability for the first input.',
      },
      {
        title: 'Train it',
        body: 'Click **▶ Train**. Watch the loss curve fall and the decision boundary bend until the four points land in the right regions. Training auto-stops once all four are classified with high confidence.',
      },
      {
        title: 'What you saw',
        body: 'The hidden layer combined two lines into a non-linear boundary. Open the **Audit** tab to see the exact numbers behind each prediction, or **∂w Check** to confirm the gradients match a finite-difference estimate.',
      },
    ],
  },
  {
    id: 'capacity',
    title: 'Capacity & dead ReLU',
    summary: 'How too few ReLU neurons can stall, and why width helps.',
    steps: [
      {
        title: 'Too little capacity',
        body: 'We’ve set **2 ReLU** neurons on XOR. Train it a few times, hitting **Reset** between runs — sometimes a ReLU neuron “dies” (outputs 0 for every input) and the network gets stuck on a plateau.',
        setup: { datasetId: 'xor', neuronsPerLayer: [2], activations: ['relu'], optimizer: 'sgd', lr: 0.1 },
      },
      {
        title: 'More neurons help',
        body: 'Now we’ve widened to **4 ReLU** neurons. Train again — the extra capacity makes a single dead unit far less likely to stall learning.',
        setup: { neuronsPerLayer: [4], activations: ['relu'] },
      },
      {
        title: 'Inspect a neuron',
        body: 'Open **∫ Calc → f(z) Plot** and pick a hidden neuron. If its z stays ≤ 0 for every input, ReLU’s derivative there is 0 and it can’t learn — that’s a dead ReLU.',
        setup: { rightPanelTab: 'calc' },
      },
    ],
  },
  {
    id: 'optimizers',
    title: 'Optimizers race',
    summary: 'On a hard dataset, see Adam and Momentum outpace plain SGD.',
    steps: [
      {
        title: 'A harder dataset',
        body: 'We’ve switched to **Spirals** — two intertwined arms that need real capacity and a good optimizer — with 2 hidden layers of 6 Tanh neurons and Adam selected.',
        setup: { datasetId: 'spiral', neuronsPerLayer: [6, 6], activations: ['tanh', 'tanh'], optimizer: 'adam', lr: 0.05 },
      },
      {
        title: 'Compare them',
        body: 'Click **⚖ Compare Optimizers** (we’ll open it for you). All four train from the *same* start — watch Adam and Momentum drop faster than plain SGD. Close the chart when you’re done.',
        setup: { openCompare: true },
      },
      {
        title: 'Train with Adam',
        body: 'Adam is selected. Click **▶ Train** and watch the spiral boundary slowly form — much harder than XOR, and a plateau detector will warn you if progress stalls.',
      },
    ],
  },
  {
    id: 'landscape',
    title: 'The loss landscape',
    summary: 'Read a 2-D slice of the loss surface and trace a descent path.',
    steps: [
      {
        title: 'A slice of the surface',
        body: 'We’ve opened the **Surface** tab on XOR. It plots the real loss over two weights (all others held fixed) — emerald is lower loss. The pink dot marks where those two weights currently sit.',
        setup: { datasetId: 'xor', neuronsPerLayer: [3], activations: ['tanh'], rightPanelTab: 'surface' },
      },
      {
        title: 'Roll downhill',
        body: 'Click **↘ Trace descent path**. The amber line is the real optimizer trajectory projected onto these two weights. Because every other weight moves too, the path can leave this static slice’s valley — a glimpse of how high-dimensional the true surface is.',
      },
    ],
  },
];
