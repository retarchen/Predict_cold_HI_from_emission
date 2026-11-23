# Analysis workflow

The LMC and TIGRESS experiments use the same model-fitting and evaluation code, with separate data and fitted parameters. The [benchmark protocol](BENCHMARK.md) defines the comparisons. The [experiment log](../EXPERIMENTS.md) records the data versions, commands, and results.

## Project folders

All paths below are relative to `hi_emission_agent_lab/`. Create subfolders as their outputs are produced.

| Location | What goes there |
|---|---|
| `scripts/` | Reusable code for extraction, fitting, evaluation, CNN comparisons, and plotting. |
| `config/` | Input paths and saved experiment settings, including the aperture, training sample size, and random seed where applicable. |
| `data/derived/<dataset>/<dataset-id>/` | Reusable processed data, such as extracted TIGRESS spectra, velocity grids, and valid-data masks. Record the source files and extraction settings used to create each version. |
| `runs/<dataset>/<run-id>/` | Outputs for one run, including the split and sampled sightline lists, fitted models, predictions, metrics, figures, FITS maps, and `REPORT.md` when produced. Use a new run ID when changing the split, extraction, or model settings so earlier results remain available. |

For example, a TIGRESS run can save its figure at `runs/tigress/annulus-n155-v01/tigress_predict_vs_true.png`. Use `lmc`, `tigress`, or `kok14` for the dataset folder. Keep the original inputs at the locations in [the data manifest](../config/data_sources.json). Record the derived-data directory, run directory, and command in [EXPERIMENTS.md](../EXPERIMENTS.md) so another session can find and reproduce the result.

## 1. Check the existing LMC spectra

Use the existing paired emission and absorption spectra. Match the emission spectra to the absorption-derived labels by source ID and verify the target definitions, units, and velocity convention.

## 2. Prepare the TIGRESS mock sightlines

Read Sections II.1 and II.2 of the [LMC paper](https://arxiv.org/html/2608.21115v2) before choosing the mock extraction. Verify the FITS extensions, spatial orientation, velocity axis, and alignment of the TIGRESS emission and truth arrays.

- **Mock emission.** Average the TIGRESS brightness-temperature spectra in an annulus around each selected position, following the paper's 28–56 arcsec LMC aperture. Convert those radii to physical distances at the adopted LMC distance, then to pixels using the TIGRESS grid. Check the effective beam and spectral response before applying any smoothing or rebinning.
- **On-source reference.** Pair the annular spectrum with the supplied TIGRESS $f_{\rm CNM}$ and $\mathcal{R}_{\rm HI}$ at the corresponding central sightline, after verifying their footprint. If a simulated optical-depth cube is available, extract its on-source profile at the same position. With the listed brightness-temperature and truth files, the reference for this experiment is the supplied truth maps.
- **Validation.** Check a few positions for aperture coverage, spatial alignment, and velocity sampling before processing the cube. Record the aperture, any smoothing or noise treatment, and departures from the observed setup in [EXPERIMENTS.md](../EXPERIMENTS.md).

## 3. Establish the baseline

Use a small fixture to check the evaluator before fitting candidates. The evaluator needs reproducible spatial splits, the constant baselines, and a table of held-out predictions. Check that training and test regions stay separate, missing labels are handled consistently, and repeated runs with the same inputs and seed give the same result. Record the rules and the command used to reproduce the check.

## 4. Compare candidate models

Begin with a few emission descriptors, such as integrated brightness, peak temperature, and line width. Test whether a simple regularized model improves on the baseline. Evaluate both targets on identical held-out sightlines and examine where the model performs poorly. For TIGRESS, use the same sparse training draws and reserved regions for every candidate.

When using Codex and Gemini, both can develop candidates against the same data definitions, splits, and evaluator. The [model experiment instructions](../.agents/skills/hi-model-experiment/SKILL.md) specify what each run records. The [benchmark review instructions](../.agents/skills/hi-benchmark-review/SKILL.md) describe an independent check of the code and results. Candidate selection uses development results. Final scoring follows once the method is fixed.

## 5. Report the TIGRESS results and CNN comparison

Use the [earlier Gemini simulation report](../../diff_model_development/gemini/LMC_model/workflow.md) and the [Milky Way model report](../../diff_model_development/gemini/Milky_Way/model.md) for the figure and table layouts. Follow the [benchmark protocol](BENCHMARK.md#tigress-prediction-and-cnn-comparison) for the scoring masks and model comparisons. Save the following products together in `runs/tigress/<run-id>/`.

| Product | Contents |
|---|---|
| `tigress_training_size.png` | Mean held-out Pearson $r$ and Lin’s CCC versus training sample size for both targets, with ±1 standard deviation across 100 training-sample draws at each size. Include 10, 15, 25, 50, 100, 155, 200, 300, and 500 sightlines. Use the [earlier training-size figure](../../diff_model_development/gemini/Milky_Way/figures/fig1_tigress_datapoints_convergence.png) as a format reference. |
| `tigress_predict_vs_true.png` | Two held-out density panels for $f_{\rm CNM}$ and $\mathcal{R}_{\rm HI}$, with a 1:1 line, Pearson $r$, CCC, and pixel count. |
| `tigress_map_reconstruction.png` | Truth, prediction, and signed residual maps for both targets, with training and held-out regions identified. |
| `tigress_cnn_comparison.png` | Side-by-side truth-versus-prediction panels for the new method, Murray CNN, and TPCNet, using the same held-out mask for each target. Use the [earlier TIGRESS comparison figure](../../diff_model_development/gemini/Milky_Way/figures/tigress_simulation_comparison.png) as a reference. |
| `tigress_predictions.csv` and `tigress_metrics.json` | Pixel coordinates, split and draw IDs, truth, predictions, and scoring mask, plus per-model, per-target, and per-training-size metrics for reproducing the figures and tables. |
| `REPORT.md` | A short methods and results report with the extraction setup, training counts and splits, metrics tables, figures, and a discussion of where the methods agree or differ. Identify any comparisons based on archived products with different input conditions. |

Include the training-size plot and a metrics table for this experiment in `REPORT.md`. Keep the detailed truth-versus-prediction and map figures for the 155-sightline case. Include a separate section and figures for any reproduced KOK14 evaluation. Link the report and the run command in [EXPERIMENTS.md](../EXPERIMENTS.md).

## 6. Apply the selected method to the LMC

After evaluation, refit the selected procedure on the eligible LMC sightlines. Compare predictions from a small cube cutout with predictions from the original source spectra at the same positions. Then run the full cube in tiles and check that tile boundaries and missing data are handled consistently.

The map products are two-dimensional celestial FITS images for the line-of-sight $f_{\rm CNM}$ and $\mathcal{R}_{\rm HI}$. Include the valid footprint and flag spectra outside the range represented by the labeled sample. Keep the model and input versions with the maps so the results can be reproduced.
