"""Fixed emission-only ridge and baselines on spatial development folds."""

import argparse
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from check_inputs import DEFAULT_MANIFEST
from check_tigress_mock import annulus_mask, extract_annulus


SEED = 20261123
TARGETS = ("f_cnm", "r_hi")
ROOT = DEFAULT_MANIFEST.parents[1]
LMC_BUFFER_DEG = 0.05
TIGRESS_BUFFER_PIX = 14
REGIMES = {"f_cnm": {"zero": lambda y: y == 0},
           "r_hi": {"near_unity": lambda y: abs(y - 1) <= .02,
                    "high_opacity": lambda y: y >= 1.2}}


def features(velocity, tb, valid):
    """Use a common set of descriptors with native channel spacing."""
    good = valid & np.isfinite(velocity) & np.isfinite(tb)
    if np.count_nonzero(good) < 2:
        return [np.nan] * 3
    dv = np.median(np.diff(velocity))
    v, t = velocity[good], tb[good]
    weights = np.maximum(t, 0)
    if weights.sum() == 0:
        return [t.sum() * dv, t.max(), 0.0]
    center = np.average(v, weights=weights)
    return [t.sum() * dv, t.max(), np.sqrt(np.average((v - center)**2, weights=weights))]


def lmc_inputs(paths):
    catalog = pd.read_csv(paths["absorption_catalog"])
    # Fixed coordinate-only quartiles. The eastern quartile is reserved.
    catalog["fold"] = np.searchsorted(catalog.RA_deg.quantile([.25, .5, .75]),
                                      catalog.RA_deg, side="right")
    projected = catalog.RA_deg * np.cos(np.deg2rad(catalog.Dec_deg.median()))
    catalog["x"], catalog["y"] = projected, catalog.Dec_deg
    reserved = catalog.loc[catalog.fold == 3, ["x", "y"]].to_numpy()
    dev = catalog[catalog.fold < 3].copy()
    distance, _ = cKDTree(reserved).query(dev[["x", "y"]])
    dev = dev[distance > LMC_BUFFER_DEG].merge(
        pd.read_csv(paths["thin_column_catalog"]), left_on="Name", right_on="Source",
        validate="one_to_one", how="left",
    )
    dev["r_hi"] = (dev[["NHI_c", "NHI_w", "NHI_u"]].sum(axis=1, min_count=3) * 1e20
                   / dev.NHI_uncorr_mean.where(dev.NHI_uncorr_mean > 0))
    dev["f_cnm"], dev["id"], dev["block"] = dev.f_c, dev.Name, dev.fold
    vectors = []
    for name in dev.Name:
        velocity, tb, _ = np.loadtxt(paths["emission_spectra_dir"] / f"{name}.txt").T
        vectors.append(features(velocity, tb, (velocity >= 170) & (velocity <= 325)))
    return dev, np.array(vectors)


def tigress_inputs(paths):
    # Use the western six block columns. The eastern two are reserved.
    y, x = np.mgrid[8:512:8, 8:384:8]
    interior = (x % 64 != 0) & (y % 64 != 0) & (x <= 368)
    x, y = x[interior], y[interior]
    with fits.open(paths["emission_cube"]) as emission, fits.open(paths["truth_file"]) as truth:
        tb = emission["TB"]
        mask, _, _ = annulus_mask(tb.header["CDELT1"], tb.header["CDELT2"])
        velocity = emission[0].header["VMIN"] + np.arange(tb.shape[0]) * emission[0].header["DV"]
        if any(h.shape != tb.shape[1:] for h in truth[1:3]):
            raise ValueError("Inconsistent TIGRESS footprint")
        labels = [np.asarray(h.data[y, x], dtype=float) for h in truth[1:3]]
        vectors = []
        for px, py in zip(x, y):
            spectrum, valid = extract_annulus(tb.data, px, py, mask)
            vectors.append(features(velocity, spectrum, valid) if valid.all() else [np.nan] * 3)
    return pd.DataFrame({
        "id": [f"{px},{py}" for px, py in zip(x, y)], "x": x, "y": y,
        "block": y // 64 * 8 + x // 64, "fold": x // 128,
        "f_cnm": labels[0], "r_hi": labels[1],
    }), np.array(vectors)


def scores(y, p):
    n = len(y)
    if not n:
        return dict(n=0, mae=None, rmse=None, bias=None, pearson_r=None, ccc=None)
    residual = p - y
    yc, pc = y - y.mean(), p - p.mean()
    vy = np.mean(yc**2) if np.ptp(y) else 0.0
    vp = np.mean(pc**2) if np.ptp(p) else 0.0
    covariance = np.mean(yc * pc) if vy and vp else 0.0
    denominator = vy + vp + residual.mean()**2
    return dict(n=n, mae=float(np.mean(abs(residual))), rmse=float(np.sqrt(np.mean(residual**2))),
                bias=float(residual.mean()), pearson_r=float(covariance / np.sqrt(vy * vp))
                if vy and vp else None, ccc=float(2 * covariance / denominator)
                if denominator else None)


def spatial_fold(frame, fold, buffer):
    test = np.flatnonzero(frame.fold == fold)
    pool = np.flatnonzero(frame.fold != fold)
    distance, _ = cKDTree(frame.iloc[test][["x", "y"]]).query(frame.iloc[pool][["x", "y"]])
    return pool[distance > buffer], test


def ridge_predictions(train_features, train_labels, test_features):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    model.fit(train_features, train_labels)
    fitted = model.predict(test_features)
    fitted[:, 0] = np.clip(fitted[:, 0], 0, 1)
    fitted[:, 1] = np.maximum(fitted[:, 1], 1)
    return fitted


def evaluate(frame, vectors, buffer, draws, training_size=None):
    labels = frame[list(TARGETS)].to_numpy()
    valid = (np.isfinite(labels).all(axis=1) & np.isfinite(vectors).all(axis=1)
             & (labels[:, 0] >= 0) & (labels[:, 0] <= 1) & (labels[:, 1] > 0))
    frame = frame.loc[valid].reset_index(drop=True)
    labels, vectors = labels[valid], vectors[valid]
    rng = np.random.default_rng(SEED)
    columns = ["id", "x", "y", "block", "fold"]
    predictions, splits = [], []
    for fold in sorted(frame.fold.unique()):
        pool, test = spatial_fold(frame, fold, buffer)
        for draw in range(draws):
            train = pool if training_size is None else np.sort(
                rng.choice(pool, size=training_size, replace=False))
            fitted = ridge_predictions(vectors[train], labels[train], vectors[test])
            for indices, role in ((train, "train"), (test, "test")):
                splits.append(frame.iloc[indices][columns].assign(eval_fold=fold, draw=draw, role=role))
            for j, target in enumerate(TARGETS):
                estimates = {"train_mean": labels[train, j].mean(), "ridge": fitted[:, j]}
                if target == "r_hi":
                    estimates["unity"] = 1.0
                for name, prediction in estimates.items():
                    predictions.append(frame.iloc[test][columns].assign(
                        draw=draw, target=target, model=name, truth=labels[test, j], prediction=prediction))
    return pd.concat(predictions, ignore_index=True), pd.concat(splits, ignore_index=True), int((~valid).sum())


def metrics_table(predictions):
    metrics = []
    for (draw, target, model), rows in predictions.groupby(["draw", "target", "model"]):
        groups = [("pooled", rows)]
        groups += [(f"fold_{fold}", r) for fold, r in rows.groupby("fold")]
        groups += [(f"block_{block}", r) for block, r in rows.groupby("block")]
        for scope, subset in groups:
            y, p = subset.truth.to_numpy(), subset.prediction.to_numpy()
            metrics.append(dict(draw=int(draw), target=target, model=model, scope=scope,
                                regime="all", **scores(y, p)))
            if scope == "pooled":
                for regime, criterion in REGIMES[target].items():
                    mask = criterion(y)
                    metrics.append(dict(draw=int(draw), target=target, model=model, scope=scope,
                                        regime=regime, **scores(y[mask], p[mask])))
    return metrics


def plot_tigress(predictions, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    panels = predictions[(predictions.model == "ridge") & (predictions.draw == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, (target, rows), label in zip(axes, panels.groupby("target"),
                                         (r"$f_{\rm CNM}$", r"$\mathcal{R}_{\rm HI}$")):
        y, p = rows.truth.to_numpy(), rows.prediction.to_numpy()
        metric = scores(y, p)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        density = ax.hexbin(y, p, gridsize=30, mincnt=1, bins="log",
                            extent=(lo, hi, lo, hi), cmap="viridis")
        ax.plot([lo, hi], [lo, hi], color="0.55", ls="--", lw=1)
        ax.set(xlabel=f"Simulation truth {label}", ylabel=f"Ridge prediction {label}",
               xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        ax.text(.05, .95, f"r = {metric['pearson_r']:.3f}\nCCC = {metric['ccc']:.3f}\nN = {metric['n']}",
                transform=ax.transAxes, va="top")
        fig.colorbar(density, ax=ax, label="Sightlines per bin")
    fig.suptitle("TIGRESS development folds, first 155-sightline draw")
    fig.savefig(output / "tigress_development_predict_vs_true.png", dpi=180)
    plt.close(fig)


def fixture():
    y = np.array([0., 1., 2.])
    assert scores(y, y)["ccc"] == 1 and scores(y, np.ones(3))["pearson_r"] is None
    assert scores(np.ones(3), np.ones(3))["ccc"] is None
    assert scores(y, y + 1)["bias"] == 1
    frame = pd.DataFrame({"id": list("abcde"), "x": [0, 1, 5, 10, 11], "y": 0,
                          "block": [0, 0, 1, 1, 1], "fold": [0, 0, 1, 1, 1],
                          "f_cnm": [.1, .2, .3, .4, np.nan], "r_hi": [1, 1.1, 1.2, 1.3, 1.4]})
    x = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5], [10, 10, 10], [11, 11, 11]])
    pred, split, omitted = evaluate(frame, x, buffer=4, draws=2, training_size=1)
    repeat, repeated_split, _ = evaluate(frame, x, buffer=4, draws=2, training_size=1)
    pd.testing.assert_frame_equal(pred, repeat)
    pd.testing.assert_frame_equal(split, repeated_split)
    assert omitted == 1 and "e" not in set(pred.id)
    assert pred.groupby(["target", "draw", "model"]).size().eq(4).all()
    for _, subset in split.groupby(["eval_fold", "draw"]):
        train, test = subset[subset.role == "train"], subset[subset.role == "test"]
        assert set(train.block).isdisjoint(test.block)
        assert (abs(train.x.iloc[0] - test.x) > 4).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("lmc", "tigress", "fixture"))
    parser.add_argument("--run-id", default="spatial-development-v01")
    args = parser.parse_args()
    fixture()
    if args.dataset == "fixture":
        print("Evaluator fixture passed")
        return

    paths = {key: DEFAULT_MANIFEST.parent / value for key, value in
             json.loads(DEFAULT_MANIFEST.read_text())[args.dataset].items()}
    if args.dataset == "lmc":
        frame, vectors = lmc_inputs(paths)
        buffer, draws, size = LMC_BUFFER_DEG, 1, None
    else:
        frame, vectors = tigress_inputs(paths)
        buffer, draws, size = TIGRESS_BUFFER_PIX, 3, 155
    predictions, splits, omitted = evaluate(frame, vectors, buffer, draws, size)
    metrics = metrics_table(predictions)
    output = ROOT / "runs" / args.dataset / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    splits.to_csv(output / "splits.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (output / "settings.json").write_text(json.dumps(
        dict(seed=SEED, draws=draws, buffer=buffer, training_size=size, omitted=omitted,
             ridge_alpha=10, development_only=True, inputs={k: str(v.resolve()) for k, v in paths.items()}),
        indent=2) + "\n")
    if args.dataset == "tigress":
        plot_tigress(predictions, output)
    summary = pd.DataFrame(metrics)
    print(summary[(summary.scope == "pooled") & (summary.regime == "all")].to_string(index=False))


if __name__ == "__main__":
    main()
