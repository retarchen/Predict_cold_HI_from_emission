"""Render the TIGRESS sample-size, holdout and matched CNN results."""

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np

from evaluate_development import TARGETS, scores


LABELS = (r"$f_{\rm CNM}$", r"$\mathcal{R}_{\rm HI}$")
RIDGE = "Ridge, 155-draw ensemble"
MEASURES = ("mae", "rmse", "bias", "pearson_r", "ccc")


def number(value):
    return "Undefined" if value is None else f"{value:.3f}"


def training_size_figure(output, summary, sizes):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharex=True)
    for i, target in enumerate(TARGETS):
        for j, (metric, title) in enumerate((("pearson_r", "Pearson r"), ("ccc", "Lin’s CCC"))):
            ax = axs[i, j]
            for model, color in (("Ridge", "#146B82"), ("Train mean", "#A55828")):
                rows = [r for r in summary if r["target"] == target and r["model"] == model
                        and r[metric]["valid_draws"]]
                if rows:
                    ax.errorbar([r["training_size"] for r in rows],
                                [r[metric]["mean"] for r in rows],
                                yerr=[r[metric]["sd"] or 0 for r in rows],
                                color=color, marker="o", ms=4, capsize=3, label=model)
            ax.axvline(155, color=".6", lw=1, ls=":")
            ax.set(xscale="log", ylabel=title, title=LABELS[i], xticks=sizes, xticklabels=sizes)
            if i == 1:
                ax.set_xlabel("Training sightlines")
            ax.grid(axis="y", alpha=.2)
    axs[0, 1].legend(frameon=False, loc="best")
    fig.suptitle("TIGRESS geographic holdout · Mean and ±1 SD across 100 draws")
    fig.savefig(output / "tigress_training_size.png", dpi=180)
    plt.close(fig)


def density_figure(output, filename, truth, models, names, single=False):
    rows, cols = (1, 2) if single else (2, len(names))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols + .7, 3.65 * rows),
                             squeeze=False, layout="constrained")
    for j, label in enumerate(LABELS):
        y = truth[:, j]
        values = np.concatenate([y, *(p[:, j] for p in models.values())])
        lower, upper = float(values.min()), float(values.max())
        margin = .025 * (upper - lower)
        lower, upper = lower - margin, upper + margin
        row_axes = []
        for k, name in enumerate(names):
            ax = axes[0, j] if single else axes[j, k]
            row_axes.append(ax)
            if name not in models:
                ax.set_axis_off()
                ax.set_title(name)
                ax.text(.5, .5, "Unavailable", transform=ax.transAxes, ha="center")
                continue
            p = models[name][:, j]
            metric = scores(y, p)
            density = ax.hexbin(y, p, gridsize=36, mincnt=1, cmap="viridis",
                                norm=LogNorm(vmin=1, vmax=len(y)),
                                extent=(lower, upper, lower, upper))
            ax.plot([lower, upper], [lower, upper], "--", color=".5", lw=1)
            ax.set(xlabel=f"Simulation truth {label}", ylabel=f"Prediction {label}",
                   xlim=(lower, upper), ylim=(lower, upper), aspect="equal")
            if not single:
                ax.set_title(name)
            ax.text(.04, .96, f"r = {number(metric['pearson_r'])}\n"
                    f"CCC = {number(metric['ccc'])}\nN = {metric['n']}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
        fig.colorbar(density, ax=row_axes, label="Sightlines per bin", shrink=.75)
    fig.suptitle("TIGRESS geographic holdout · 155-sightline ridge ensemble")
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def map_figure(output, grid):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for i, (target, label) in enumerate(zip(TARGETS, LABELS)):
        y = grid[f"{target}_truth"].to_numpy()
        p = grid[f"{target}_prediction"].to_numpy()
        error = p - y
        norm = Normalize(min(y.min(), p.min()), max(y.max(), p.max()))
        lim = np.max(np.abs(error))
        signed = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        for j, (values, title, cmap, scale) in enumerate((
            (y, "Truth", "viridis", norm), (p, "Prediction", "viridis", norm),
            (error, "Prediction − truth", "RdBu_r", signed),
        )):
            ax = axes[i, j]
            field = ax.scatter(grid.x, grid.y, c=values, s=4.5, marker="s", linewidths=0,
                               cmap=cmap, norm=scale, rasterized=True)
            ax.axvline(384, color=".35", ls="--", lw=1)
            ax.set(xlim=(0, 512), ylim=(0, 512), aspect="equal", title=f"{label}  {title}",
                   xlabel="x (pixel)", ylabel="y (pixel)")
            fig.colorbar(field, ax=ax, shrink=.82)
    fig.suptitle("155-sightline ridge ensemble · Western pool and eastern test region")
    fig.savefig(output / "tigress_map_reconstruction.png", dpi=180)
    plt.close(fig)


def write_report(output, grid, truth, models, summary, comparison, meta):
    training_size_figure(output, summary, meta["sizes"])
    density_figure(output, "tigress_predict_vs_true.png", truth, {RIDGE: models[RIDGE]}, [RIDGE], True)
    names = [RIDGE, "Murray-style CNN", "TPCNet"]
    density_figure(output, "tigress_cnn_comparison.png", truth,
                   {name: models[name] for name in names if name in models}, names)
    map_figure(output, grid)
    skill = {(r["training_size"], r["target"]): r for r in summary if r["model"] == "Ridge"}
    lines = [
        "# TIGRESS sparse-training benchmark",
        "",
        "## Methods",
        "",
        "The fixed procedure uses three emission features: integrated brightness, peak brightness, "
        "and brightness-weighted velocity dispersion. Each draw fits its own standardization and "
        "ridge regression with α = 10. Predictions are bounded to 0–1 for CNM fraction and "
        "floored at one for the opacity correction. The references are the training-target mean "
        "and an opacity correction of one.",
        "",
        f"The training pool is the {meta['train_centers']:,} frozen western development centers in "
        f"48 blocks. The held-out test contains {meta['test_centers']:,} centers in 16 eastern "
        "blocks, sampled on the same 8-pixel grid. The nearest cross-split centers are "
        f"{meta['min_separation_pix']:.0f} pixels apart, above the 14-pixel buffer. At each size, "
        "100 independently seeded sets were drawn without replacement within each draw. "
        f"The base seed is {meta['seed']}. All 900 fits use the same scored pixels.",
        "",
        "The off-source annulus uses 108 equal-weight pixel centers between 28 and 56 arcsec at "
        "50 kpc. Native simulation spectra have 2 pc pixels and 201 channels at 0.5 km/s, "
        "without added noise or smoothing. Truth is sampled at the central pixel. "
        "[LMC observing geometry](https://arxiv.org/html/2608.21115v2#S2)",
        "",
        "The west and east share one simulation realization. An earlier extraction check inspected "
        "one pixel inside the eastern region, at (384, 320), outside the scored 8-pixel grid. "
        "This is a geographic holdout with prior access to the underlying simulation.",
        "",
        "## Training-size sensitivity",
        "",
        "![Mean and sample standard deviation across draws](tigress_training_size.png)",
        "",
        "The table gives mean scores across individual ridge draws. Pearson r and CCC "
        "include ±1 sample standard deviation, reflecting training-sample sensitivity on "
        "the fixed test region. The metrics file gives valid-draw counts for every metric and model.",
        "",
        "| Training size | Target | Pearson r | Lin’s CCC | MAE | RMSE | Bias |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for size in meta["sizes"]:
        for target in TARGETS:
            r = skill[size, target]
            row = [str(size), target]
            row += [f"{number(r[m]['mean'])} ± {number(r[m]['sd'])}" for m in ("pearson_r", "ccc")]
            row += [number(r[m]["mean"]) for m in ("mae", "rmse", "bias")]
            lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## 155-sightline reconstruction and CNN comparison",
        "",
        "The ridge panels use mean predictions from the 100 fits with 155 sightlines. "
        "The scores below are computed after averaging predictions. The preceding table "
        "averages scores computed for each draw.",
        "",
        "![Ridge ensemble by target](tigress_predict_vs_true.png)",
        "",
        "![Truth, predictions and signed residuals on the sparse grid](tigress_map_reconstruction.png)",
        "",
        "The map shows the sampled grid across the two regions. The vertical line marks "
        "the eastern holdout boundary. Western residuals are for spatial context.",
        "",
        "![Comparison on the common test mask](tigress_cnn_comparison.png)",
        "",
        "Both CNNs used the same underlying annular brightness spectra as ridge. "
        "The locally retrained Murray-style CNN uses 414 peak-centered channels at 0.42 km/s. "
        "TPCNet uses 256 channels at 0.3125 km/s and the archive’s sinusoidal input encoding. "
        "Both are zero-padded outside the input range and use the same output bounds as ridge. "
        "The checkpoint digests and verification statuses are stored in the metrics file. "
        "Murray-style weights are from local KOK14 training. TPCNet uses the archived Saury/Seta "
        "checkpoint through the local compatible inference code, with exact training membership "
        "unresolved. These comparisons test transfer with different training domains and budgets. "
        "The old FITS maps lack confirmed registration and extraction provenance for this split.",
        "",
        "| Model | Target | Test N | Pearson r | Lin’s CCC | MAE | RMSE | Bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparison:
        if r["regime"] == "all":
            row = [r["model"], r["target"], str(r["n"])] + [
                number(r[k]) for k in ("pearson_r", "ccc", "mae", "rmse", "bias")
            ]
            lines.append("| " + " | ".join(row) + " |")
    for name, status in meta["cnn"].items():
        if status["status"] != "verified":
            lines += ["", f"{name} is unavailable for comparison. {status['reason']}"]
    lines += [
        "",
        "### Error by truth regime",
        "",
        "| Model | Truth regime | Test N | MAE | Bias |",
        "|---|---|---:|---:|---:|",
    ]
    for r in comparison:
        if r["regime"] != "all":
            lines.append(f"| {r['model']} | {r['regime'].replace('_', ' ')} | {r['n']} | "
                         f"{number(r['mae'])} | {number(r['bias'])} |")
    overall = {(r["model"], r["target"]): r for r in comparison if r["regime"] == "all"}
    lines += ["", "For each target, the lowest common-mask MAE among the available models was:"]
    for target in TARGETS:
        best = min((r for (_, t), r in overall.items() if t == target), key=lambda r: r["mae"])
        lines.append(f"- {target}: {best['model']} ({number(best['mae'])}).")
    high = [r for r in comparison if r["model"] == RIDGE and r["regime"] == "high_opacity"][0]
    lines += [
        "",
        f"On the {high['n']} high-opacity sightlines (R_HI ≥ 1.2), ridge signed bias was "
        f"{number(high['bias'])}. Near-unity means |R_HI − 1| ≤ 0.02. "
        "The regime results help distinguish errors on the dominant low-opacity population "
        "from errors on the high-opacity tail.",
        "",
        "## Reproduction files",
        "",
        "`python3 scripts/run_tigress_benchmark.py` from the lab directory reproduces the run "
        "in a fresh output directory. See the [scoring table](tigress_predictions.csv), "
        "[per-draw metrics](tigress_draw_metrics.csv), [metrics and provenance](tigress_metrics.json), "
        "[draw seeds](draws.csv), and [draw memberships](training_draws.csv). "
        "The [split and mask](frozen_split.csv), [test spectra](matched_test_inputs.npz), "
        "and [all individual draw predictions](draw_predictions.npz) preserve pixel order and "
        "allow exact re-scoring. The NPZ prediction axes are draw, model, pixel, and target.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
