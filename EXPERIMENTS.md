# Experiment log

## Input inventory

Command from the project folder:

```sh
python3 scripts/check_inputs.py
```

The LMC absorption catalog contains 155 unique source IDs. Each matches an emission spectrum and an optically thin column entry. The spectra directory and the optically thin column catalog each contain 205 entries. The LMC emission cube and both TIGRESS FITS files are present.

The inventory checked file paths and source IDs. The target values, spectrum extraction, and alignment of the FITS arrays require the checks described in the [benchmark protocol](docs/BENCHMARK.md#inputs-and-targets).

## Run record

```text
Run ID and date
Scientific question
Input, code, and split versions
Candidate and baseline
Run command, environment, and random seed
Derived-data directory and run directory
Output files and held-out results
Uncertainties and limitations
Interpretation
```

## TIGRESS annulus smoke test (2026-11-23)

Run `annulus-smoke-v01` checked whether a few complete off-source apertures could be extracted and paired with central simulation truth. It used the two TIGRESS files in `config/data_sources.json` and three fixed interior positions selected for geometry checks.

```sh
python3 scripts/check_tigress_mock.py
```

Ran in `jupyter_env` and saved the spectra, channel mask, central optical depth, and truth in [sightlines.npz](data/derived/tigress/annulus-smoke-v01/sightlines.npz). The fixture confirmed central exclusion and rejection of an incomplete edge aperture. An independent pixel-by-pixel calculation reproduced all saved mean spectra and central references.

The 28–56 arcsec aperture corresponds to 6.787–13.575 pc at the adopted 50 kpc distance. The 2 pc grid gives radii of 3.394–6.787 pixels and 108 equally weighted pixel centers per aperture. The observed emission has a 30 arcsec beam, 1 km/s spectral resolution, and 1.1 K channel sensitivity. This trial used the native simulation grid with no beam convolution, spectral smoothing, rebinning, or added noise. The cube contains no instrumental beam metadata. See [Chen et al., Sections II.1–II.2](https://arxiv.org/html/2608.21115v2#S2).

The `TB` and `TAU` extensions share the same `[velocity, y, x]` grid. The generator's velocity convention and the primary header give 201 centers from −50 to +50 km/s in 0.5 km/s steps. A standard one-based FITS interpretation of the stored zero-valued `CRPIX3` would offset the axis by one channel. The unnamed 512 × 512 truth HDUs lack spatial WCS. The [generation notebook](../../TIGRESS/diff_Z/example_synthetic-HI.ipynb) establishes the common `[y, x]` orientation by projecting along z in both products and computing HDU 2 from the same `TB` array. HDU 1 stores the fraction for gas at or below 300 K under the generator's density selection, and HDU 2 stores the true-to-thin column ratio.

| Center (x, y), zero-based | Valid channels | Peak annular TB (K) | Peak central tau | Central f_CNM | Central R_HI |
|---|---:|---:|---:|---:|---:|
| (128, 128) | 201 | 8.0768 | 0.0009 | 0.00000 | 1.00038 |
| (256, 256) | 201 | 23.6757 | 0.0013 | 0.00000 | 1.00063 |
| (384, 320) | 201 | 1.2225 | 0.0001 | 0.00000 | 1.02877 |

All three apertures had full coverage and finite central truth. The sample covers only zero-CNM sightlines and tests extraction mechanics.

## LMC input validation (2026-11-23)

Run `input-validation-v01` followed up on the input inventory. It checked the 155 matched sources in the data manifest and compared three saved spectra with cutouts from the LMC FITS cube. The three sources were the first three catalog rows. Their full-precision coordinates came from the local GASKAP v1.1 spectra catalog, matched by source ID.

```sh
python3 scripts/check_lmc_inputs.py
```

Ran in `jupyter_env`. The run directory contains the [source-by-source label audit](runs/lmc/input-validation-v01/label_audit.csv) and [cutout comparisons](runs/lmc/input-validation-v01/cutouts.json).

- **CNM fraction.** The stored `f_c` agrees with `NHI_c / (NHI_c + NHI_w + NHI_u)` within 0.00535 for every source, consistent with the catalog precision. The three component columns are complete. One WNM entry and 153 UNM entries are explicitly zero.
- **Opacity correction.** The historical LMC analysis uses `NHI_uncorr_mean` in cm⁻². The numerator here includes all three phases, whose catalog units are 10²⁰ cm⁻². All 155 denominators are finite and positive. The resulting ratio spans 0.903–1.974, with 12 values below unity. The audit retains these values and the catalog uncertainty fields. See the [historical fitting analysis](../../LMC_all/fitting_analysis.ipynb).
- **Integration check.** Integrating the full saved emission grid with the optically thin conversion gives a median value 0.90% higher than `NHI_uncorr_mean`. The largest difference is +140.3% for J044047-695217. The exact integration window and procedure for the tabulated mean still need provenance confirmation. The full-grid integral and the catalog denominator should be treated as distinct measurements.
- **Velocity and aperture.** All 155 saved spectra have finite velocity, brightness, and dispersion columns sampled at 1 km/s. The three cutouts match all 173 channel centers on the cube's header LSRK grid within 0.000001 km/s. Unweighted 28–56 arcsec annuli give 149–150 pixel centers and RMS brightness differences of 0.094–0.168 K from the saved profiles. These residuals leave the exact extraction and channel-mask convention unresolved. Use the saved spectra for the labeled sample and resolve the extraction differences before treating cube-derived spectra as interchangeable.

This was an input audit. The TIGRESS smoke-test products and the original LMC files were used as read-only references.

## Spatial development comparison (2026-11-23)

Run `spatial-development-v01` compared a fixed ridge model with fold-specific training means and the physical R_HI = 1 reference. LMC and TIGRESS were fit separately. The run used `jupyter_env` with scikit-learn 1.8.0 and random seed 20261123.

```sh
python3 scripts/evaluate_development.py fixture
python3 scripts/evaluate_development.py lmc
python3 scripts/evaluate_development.py tigress
```

### Input and extraction checks

- Rechecked all 155 saved LMC spectra and labels against the input audit. The optically thin integration window behind the catalog denominator remains unverified. The run used the catalog convention, retained all below-unity labels in the eligible development set, and restricted LMC emission features to the paper's 170–325 km/s LSR window.
- Read Sections II.1–II.2 of Chen et al. The observed on-source absorption is continuum-weighted and the off-source emission is averaged over 28–56 arcsec. The TIGRESS mock reproduces the off-source aperture at 50 kpc, with native 2 pc pixels and 0.5 km/s channels. The run applies no beam convolution, spectral smoothing, rebinning or noise. An independent pixel sum reproduced the previous three annular spectra and a fourth at (128, 256), which has central f_CNM = 0.03816, R_HI = 1.00527 and peak annular brightness of 44.285 K. All four had 108 aperture pixels and 201 valid channels. [Chen et al., Sections II.1–II.2](https://arxiv.org/html/2608.21115v2#S2)

### Shared splits and evaluator

- **LMC** used RA quartile boundaries of 75.680°, 81.240° and 84.555°. The three western bands contributed 39, 38 and 39 development validation sources. Each fold trained on the other two bands with a 3 arcmin projected separation rule. The east was reserved. The closest training and validation pair was 0.212° in the adopted projection.
- **TIGRESS** used 64 × 64 pixel blocks with three development bands spanning two block columns each. Samples were on an 8-pixel grid, with block boundaries skipped and the development edge capped at x = 368. The 14-pixel separation rule exceeds twice the 6.787-pixel aperture radius. The closest training and validation centers were 16 pixels apart. Each fold scored 784, 784 or 728 sightlines and used three seeded samples of 155 training sightlines from the other bands. The 2,296 development sightlines cover 48 blocks in one realization. The two eastern block columns were outside development scoring. The historical smoke test had already inspected one position in that region, and the existing LMC inventory had already inspected all labels. These geographic reservations therefore do not qualify as sealed tests.
- Ridge used integrated brightness, peak brightness and a brightness-weighted velocity dispersion. Standardization was fit in each training draw, with alpha fixed at 10. Ridge predictions were bounded to [0, 1] for f_CNM and floored at 1 for R_HI. Positions, IDs, absorption and label uncertainties were excluded from predictors. Each comparison used a common eligible mask. The fixture exercised missing-label exclusion, constant-prediction metrics, separation and seed reproducibility. A repeated development fit reproduced the saved splits, predictions and metrics. The first TIGRESS attempt stopped before scoring because pandas rejected big-endian FITS labels. Conversion to native floating-point values resolved it.

### Development results

TIGRESS entries are the means of three draws on the same validation rows. Metrics include more digits and per-draw, per-fold, per-block and regime results in the saved files.

| Dataset | Target | Method | MAE | RMSE | Bias | Pearson r | Lin's CCC |
|---|---|---|---:|---:|---:|---:|---:|
| LMC | f_CNM | Train mean | 0.1790 | 0.2156 | −0.0000 | −0.125 | −0.016 |
| LMC | f_CNM | Ridge | 0.1719 | 0.2192 | −0.0043 | 0.153 | 0.111 |
| LMC | R_HI | Train mean | 0.1655 | 0.2243 | −0.0001 | −0.289 | −0.083 |
| LMC | R_HI | Unity | 0.2013 | 0.2893 | −0.1955 | Undefined | 0.000 |
| LMC | R_HI | Ridge | 0.1467 | 0.2074 | −0.0082 | 0.261 | 0.171 |
| TIGRESS | f_CNM | Train mean | 0.1362 | 0.1757 | +0.0082 | 0.009 | −0.002 |
| TIGRESS | f_CNM | Ridge | 0.0791 | 0.1244 | +0.0152 | 0.710 | 0.674 |
| TIGRESS | R_HI | Train mean | 0.0554 | 0.1112 | +0.0033 | −0.051 | −0.002 |
| TIGRESS | R_HI | Unity | 0.0370 | 0.1169 | −0.0370 | Undefined | 0.000 |
| TIGRESS | R_HI | Ridge | 0.0428 | 0.0918 | +0.0092 | 0.572 | 0.471 |

Pooled training-mean predictions can have nonzero correlation because the fitted constant varies by fold. Across the three TIGRESS draws, ridge CCC sample standard deviations were 0.011 for f_CNM and 0.013 for R_HI. These describe draw variation on one realization. Ridge overpredicted the 1,517 zero-CNM cases by 0.050 on average. On the 118 TIGRESS cases with R_HI ≥ 1.2 it had MAE 0.270 and bias −0.268. On the 1,791 near-unity cases (|R_HI − 1| ≤ 0.02) ridge MAE was 0.0252 versus 0.0038 for unity. The LMC ridge lowered MAE for both targets and raised f_CNM RMSE slightly. On the 44 LMC high-opacity cases, ridge R_HI bias was −0.179.

Each run directory contains the exact split assignments, prediction table, metrics, settings and fitted-input fingerprints. [LMC run](runs/lmc/spatial-development-v01/) · [TIGRESS run](runs/tigress/spatial-development-v01/) · [TIGRESS development figure](runs/tigress/spatial-development-v01/tigress_development_predict_vs_true.png)

### Archived CNN readiness

| Archive | Verification | Use |
|---|---|---|
| Murray-style 1D CNN `murray_retrained_kok14.pt` | Loaded strictly and produced finite predictions on two development annular spectra using the archived peak-centered, zero-padded 414-channel, 0.42 km/s input recipe. The local training record gives 31,024 KOK14 training spectra and identifies these as locally retrained weights. SHA-256 `1a14ecbba9925105677ab3920dc179ccd40025d0aca8fe747eb77945bd878f09`. | Runnable as a local Murray-style transfer baseline. Published Murray weights were not verified. |
| TPCNet `ctrans_sin_add_10_256_final.pth` | Loaded strictly through the archived checkpoint-compatible architecture after omitting the unused `pos_embedding` entry. Produced finite predictions on the same two spectra with the archived peak-centered 256-channel, 0.3125 km/s and sinusoidal input recipe. The archive documents Saury/Seta training domains, with exact training membership unresolved. SHA-256 `c89a186df2f7dd28e736f4c005dad3a10d7b157d587bf416273049e3f3f0b9f5`. | Runnable as a transfer baseline with the local inference reconstruction identified. |
| Legacy TIGRESS CNN FITS maps | Both targets exist for both CNNs. The 512 × 512 headers lack spatial WCS and processing history. The archive does not establish the current annular input, pixel registration and common evaluation mask. | Historical reference only. Generate fresh predictions for a matched comparison. |
| KOK14 comparison archive | The local `training_data_kok14_galfa.pickle` and Murray training metadata are present. | Available for a separate comparison once the split and fitting procedure are fixed. No KOK14 scores were generated here. |

Archive evidence comes from the local [CNN provenance](../../diff_model_development/gpt/CNN_comparison/provenance.json), [Murray training record](../../diff_model_development/gpt/CNN_comparison/models/murray_training.json), [inference code](../../diff_model_development/gpt/CNN_comparison/src/compare_three_methods.py), and [legacy FITS directory](../../diff_model_development/gemini/Milky_Way/fits/). The CNN checks exercised checkpoint loading and input formatting without scoring either network.

**Next step.** Run both CNN checkpoints on exactly the existing TIGRESS development annuli, using the same saved pixel IDs, truth and scoring mask. Compare opacity errors near unity and at high opacity before fixing the model procedure. Use separate access to an independent realization if a sealed test is required. Final scoring and the 900-draw training-size experiment remain pending.
