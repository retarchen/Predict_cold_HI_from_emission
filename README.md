# Cold H I from 21 cm emission

Can H I emission alone recover the cold neutral medium fraction, $f_{\rm CNM}$, and the opacity correction, $\mathcal{R}_{\rm HI}$, across the Large Magellanic Cloud (LMC)?

The LMC analysis uses 155 sightlines with both emission spectra and absorption-derived labels. A model learns the relation between the emission profile and the two sightline-integrated quantities. At application time, each prediction uses the emission spectrum alone.

The analysis has two parts. In the LMC, spatially held-out absorption sightlines test how well the model predicts unseen regions of the observed galaxy. In TIGRESS, the same fitting procedure is trained on a similarly small set of simulated sightlines and tested against known values in reserved regions. The two fits use their own labels. The TIGRESS experiment measures map recovery under controlled conditions, while the LMC experiment measures predictive accuracy at the observed sightlines.

## Data

The [data manifest](config/data_sources.json) lists the original files. It points to the LMC absorption catalog, emission spectra, optically thin column catalog, and emission cube, and to the TIGRESS emission cube and truth maps. Relative paths are resolved from the manifest’s folder.

The [input check](scripts/check_inputs.py) verifies the paths and cross-matches the source IDs:

```sh
python3 scripts/check_inputs.py
```

All 155 LMC catalog entries matched a spectrum and an optically thin column entry. The [experiment log](EXPERIMENTS.md#input-inventory) gives the counts and scope of that check. Target definitions, extraction apertures, and FITS alignment are part of the data checks described in the protocol.

## Analysis notes

- The [benchmark protocol](docs/BENCHMARK.md) defines the targets, spatial splits, baselines, and evaluation metrics.
- The [analysis workflow](docs/WORKFLOW.md) describes how the input checks, model comparison, independent review, and map tests fit together. Its [project folders](docs/WORKFLOW.md#project-folders) section shows where scripts, processed data, and figures are saved.
- The [experiment log](EXPERIMENTS.md) records data checks, model comparisons, run commands, and decisions that affect the analysis.

The project instructions are in [AGENTS.md](AGENTS.md). The [model experiment](.agents/skills/hi-model-experiment/SKILL.md) and [benchmark review](.agents/skills/hi-benchmark-review/SKILL.md) files describe the two repeatable checks used during development.
