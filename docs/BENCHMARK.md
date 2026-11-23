# Benchmark protocol

Can a small set of labeled H I sightlines support useful estimates of $f_{\rm CNM}$ and $\mathcal{R}_{\rm HI}$ across a galaxy? Fix the target definitions, input convention, spatial splits, and metrics before comparing candidate models. Record any later changes that affect interpretation or comparability.

| Experiment | Training sample | Evaluation |
|---|---|---|
| LMC | Labeled LMC sightlines in the training sky groups | Absorption-derived targets at sightlines in held-out groups |
| TIGRESS | 10, 15, 25, 50, 100, 155, 200, 300, and 500 simulated sightlines per draw | Known targets in reserved regions and, when available, independent realizations |
| LMC map | All eligible labeled LMC sightlines, after model selection | Agreement between source and map predictions, plus coverage across the cube |

TIGRESS and the LMC have separate fitted parameters. A model trained on TIGRESS and tested on the LMC would address a separate question about transfer between domains. Because the TIGRESS cube has been used in earlier work, a new split within that cube tests recovery of reserved regions. A fresh realization would provide a stronger test of generalization.

## Inputs and targets

The predictor uses brightness temperature $T_B$ in K, the velocity grid in $\mathrm{km\,s^{-1}}$, and a valid-channel mask. Integrated brightness, peak temperature, line width, and narrow-line power are examples of features derived from these inputs. Additional emission-derived features, such as noise or aperture dispersion, need the same definition and availability in the source spectra, simulations, and maps. Positions and source IDs are used for extraction, grouping, and mapping.

The LMC catalog reports $f_{\rm CNM}$ as `f_c`. Check its agreement with the ratio of CNM column density to total atomic column density under the catalog’s conventions for missing components. The working definition of $\mathcal{R}_{\rm HI}$ is the absorption-corrected H I column divided by the matched optically thin H I column. Verify the units, velocity integration range, and meaning of `NHI_uncorr_mean`. Report nonpositive denominators, ratios below unity, missing components, and uncertain labels before deciding how to handle them. Retain the original values and record exclusions. Any physical bounds on predictions are applied separately from these label checks.

Earlier code refers to the `TB` extension in `test_z.fits` and HDUs 1 and 2 in `fcnm_RHI_z.fits`. Confirm which arrays contain emission, $f_{\rm CNM}$, and $\mathcal{R}_{\rm HI}$ using the FITS metadata and data provenance. Check the axes, orientation, channel spacing, and masks before matching simulated spectra to truth.

## Match the simulated and observed spectra

Use the existing LMC spectra and labels. In TIGRESS, use annular emission paired with truth at the central sightline, following the geometry of the observed pairs and verifying the simulation truth footprint. The [workflow](WORKFLOW.md#2-prepare-the-tigress-mock-sightlines) describes how to adapt the aperture to the simulation grid. Reference values must be independent of added noise and model predictions.

Check whether the saved LMC spectra and the spectra extracted for mapping use the same aperture, velocity convention, channel mask, and preprocessing. State how the simulation accounts for the beam, aperture, spectral response, and noise. Clean and observation-like spectra can be assessed separately.

## Spatial splits and scoring

**LMC.** Use fixed spatial folds or a separate held-out region, depending on the source geometry. Account for spatial dependence and overlapping extraction apertures when grouping the sources. Keep related footprints and repeated source IDs together. Select features and tune parameters using inner spatial folds within the outer training data. Label any holdout used repeatedly during development accordingly.

**TIGRESS.** Reserve blocks before drawing the training sightlines. Choose the buffer between training and test regions with the effective beam, aperture, and spatial correlation in mind. Repeat the experiment over several sparse draws, using the same positions and seeds for all candidates. Keep noise realizations of the same simulated sightline in the same split. Include zero-CNM and near-unity opacity cases in the map-recovery experiment and report their errors separately. A draw matched to the observed absorption selection can be evaluated separately. Report the number of pixels and the number of independent blocks or realizations.

**Baselines and metrics.** Use a constant fitted within each training fold for each target and include $\mathcal{R}_{\rm HI}=1$ as a physical reference. A small regularized model provides the first comparison with these baselines. Consider a two-part model for $f_{\rm CNM}$ only after checking the meaning and prevalence of zeros. Compare mean absolute error on the same held-out rows and also report root mean square error, signed bias, Pearson correlation, and Lin’s concordance correlation coefficient. Assess uncertainty across spatial groups or blocks when there are enough to support it. Report errors at higher opacity separately. If using ROC AUC, specify thresholds before scoring and report it only where both classes are present.

## TIGRESS prediction and CNN comparison

Test the eight training sizes used in the [earlier TIGRESS sample-size experiment](../../diff_model_development/gemini/Milky_Way/model.md) and add 155 to match the LMC sample. At each size, draw 100 training samples and refit the selected procedure for each draw. Sample without replacement within a draw, allow sightlines to recur across draws, and record the seed and selected positions for each draw. Use the same selection rule, mock extraction, model procedure, and held-out blocks across sample sizes. Save the held-out metrics for each draw. Plot mean Pearson $r$ and mean Lin’s CCC against training size for both targets, with error bars of ±1 sample standard deviation across draws. These bars describe sensitivity to the choice of training sightlines on the fixed test region. Summarize RMSE, MAE, and signed bias at each size as well. Report the number of valid draws for each metric. If comparing different selection rules, label those experiments separately so the effect of training size is clear.

For the 155-sightline case, plot simulated truth on the horizontal axis and predictions on the vertical axis for both $f_{\rm CNM}$ and $\mathcal{R}_{\rm HI}$. Use hexbin density plots, a 1:1 line, and annotations for Pearson $r$, Lin’s CCC, and the number of scored pixels. Compute the metrics from the same reserved pixels shown in each panel. If plotting predictions averaged across draws, report their scores separately from the average of the individual draw scores. A full-field map can illustrate spatial recovery, with the holdout footprint identified and scores computed on the held-out pixels. Use [the earlier Gemini truth-versus-prediction figure](../../diff_model_development/gemini/LMC_model/simulation/simulation_predict_vs_true.png) and [TIGRESS reconstruction figure](../../diff_model_development/gemini/Milky_Way/figures/fig3_tigress_map_reconstruction.png) as layout references.

Compare the new model with the **Murray et al. (2020) 1D CNN** and **TPCNet** from the [earlier Milky Way application report](../../diff_model_development/gemini/Milky_Way/application.md). For the TIGRESS comparison, put the models in columns and the two targets in rows, with common axis limits and the same held-out pixels, truth definition, and valid mask for each target. Use the same underlying mock observation for the direct comparison, then apply each checkpoint’s documented input formatting. Record the checkpoint, training domain, and spectral preprocessing. Use existing CNN prediction maps only after verifying their input spectra, pixel alignment, and masks. If a legacy prediction uses a different extraction or test set, report it separately with that condition stated. Use undefined entries where a checkpoint, prediction, or metric cannot be verified.

The KOK14 comparison in Section 3 of the [earlier application report](../../diff_model_development/gemini/Milky_Way/application.md) provides a separate test. If its original test spectra and reference models are available, evaluate the selected procedure and both CNN references on that common test set. Document how the selected procedure was fitted and keep the KOK14 test labels out of tuning. Use the [KOK14 hexbin figure](../../diff_model_development/gemini/Milky_Way/figures/simulation_predict_vs_true_comparison.png) as the layout reference. Keep its results separate from the TIGRESS scores.

## Map checks

Compare predictions at the labeled positions with values from the WCS-aligned maps. Check the valid footprint, physical bounds, and agreement between a cutout and tiled inference. For aperture-based features, tiles need enough overlap to give consistent results at their boundaries.

Report which areas of the map have emission profiles unlike those in the labeled sample. Coverage in feature space and predictive error are separate quantities. If the held-out evidence is weak, describe the maps as exploratory and state which regions and regimes are poorly constrained.
