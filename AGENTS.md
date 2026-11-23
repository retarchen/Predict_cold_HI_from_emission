# H I emission analysis

Estimate sightline-integrated $f_{\rm CNM}$ and $\mathcal{R}_{\rm HI}$ from H I emission. Fit the LMC model using its own labeled sightlines. Use TIGRESS to test the same sparse-training procedure against known values.

Use the existing paired LMC spectra and their absorption-derived labels. Before preparing the TIGRESS inputs, read Sections II.1 and II.2 of [Chen et al., arXiv:2608.21115v2](https://arxiv.org/html/2608.21115v2). Use the paper's observing geometry as the reference for the [TIGRESS mock sightlines](docs/WORKFLOW.md#2-prepare-the-tigress-mock-sightlines).

The [benchmark protocol](docs/BENCHMARK.md) sets the target definitions, spatial splits, and evaluation rules. The [analysis workflow](docs/WORKFLOW.md) describes the experiments and expected outputs. The [experiment log](EXPERIMENTS.md) records changes that affect the scientific interpretation or the comparison between runs.

The [input files](config/data_sources.json) and earlier experiments are read-only references. Save reusable code in `scripts/`, processed data in `data/derived/`, and run results in `runs/<dataset>/<run-id>/`. See the [project folders](docs/WORKFLOW.md#project-folders) for details. All candidate models use the shared evaluator and the same development splits. Keep absorption labels, positions, and source IDs out of the predictor. Positions may be used to extract spectra, define spatial groups, and write the final maps. Fit any learned preprocessing inside the training folds.

Keep the final holdout labels and scores separate from candidate development. A sealed test requires separate access to those inputs. Check a small fixture and a sky cutout before running the full cube. Retain the evidence for unsuccessful experiments as well as successful ones.

Include held-out TIGRESS truth-versus-prediction plots and Pearson $r$ and Lin’s CCC for both targets. Test training sizes of 10, 15, 25, 50, 100, 155, 200, 300, and 500 sightlines, with 100 training-sample draws at each size. Follow the [TIGRESS reporting workflow](docs/WORKFLOW.md#5-report-the-tigress-results-and-cnn-comparison) for the figures, metrics, and comparison with the previous CNN work.
