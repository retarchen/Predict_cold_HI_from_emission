---
name: hi-model-experiment
description: Compare an emission-only fCNM and RHI model with the baselines on the shared spatial benchmark.
---

# Model experiment

The [benchmark protocol](../../../docs/BENCHMARK.md) defines the inputs, targets, and spatial splits. The [experiment log](../../../EXPERIMENTS.md) records the run. A candidate comparison requires the shared evaluator and development splits.

Test one change at a time against the same baselines. Fit any learned preprocessing on training data within each fold. Evaluate both targets on identical held-out rows and record where the candidate performs poorly, including zero-CNM and higher-opacity regimes when available.

Save the exact run command, seed, input and split versions, and output paths. Record what changed, the measured result, and its limits. Keep final holdout results separate from model development.
