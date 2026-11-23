---
name: hi-benchmark-review
description: Check the target definitions, spatial splits, reproducibility, and map inference in an H I emission experiment.
---

# Benchmark review

Use the [benchmark protocol](../../../docs/BENCHMARK.md), candidate code, and saved results. Review the candidate and evaluator without editing them.

Trace the targets to the source catalog or simulation truth and the predictor inputs to the emission data. Check how the analysis handles missing values, zero CNM fractions, and opacity corrections near unity. Verify that footprints and noise realizations do not overlap across splits and that model selection and learned preprocessing use the appropriate training data.

Reproduce a small development run when the inputs and command are available. Check that the candidate and baselines use the same held-out rows and that undefined metrics are identified. For mapping, check the WCS, masks, tile boundaries, and coverage of the labeled sample.

Record findings with file or run evidence and distinguish checks that were run from checks based on code inspection. State what the results support and what remains uncertain.
