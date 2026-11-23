"""Score the frozen TIGRESS geographic holdout with a fixed sparse ridge procedure."""

import hashlib
import json
from pathlib import Path
import sys

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import sklearn

from check_inputs import DEFAULT_MANIFEST
from check_tigress_mock import annulus_mask, extract_annulus
from evaluate_development import (
    REGIMES, ROOT, SEED, TARGETS, TIGRESS_BUFFER_PIX, features, fixture,
    ridge_predictions, scores,
)
from report_tigress import write_report


SIZES = (10, 15, 25, 50, 100, 155, 200, 300, 500)
DRAWS = 100
RUN_ID = "frozen-east-900-v01"
FROZEN = ROOT / "runs/tigress/spatial-development-v01"
ARCHIVE = ROOT.parent / "diff_model_development"
CHECKPOINTS = {
    "Murray-style CNN": (
        "gpt/CNN_comparison/models/murray_retrained_kok14.pt",
        "1a14ecbba9925105677ab3920dc179ccd40025d0aca8fe747eb77945bd878f09",
        414, .42,
    ),
    "TPCNet": (
        "data/zenodo_results/ctrans/checkpoint/c256_1split/ctrans_sin_add_10_256_final.pth",
        "c89a186df2f7dd28e736f4c005dad3a10d7b157d587bf416273049e3f3f0b9f5",
        256, .3125,
    ),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inputs():
    manifest = json.loads(DEFAULT_MANIFEST.read_text())["tigress"]
    paths = {name: DEFAULT_MANIFEST.parent / path for name, path in manifest.items()}
    hashes = {name: digest(path) for name, path in paths.items()}
    if hashes != json.loads((FROZEN / "input_sha256.json").read_text()):
        raise ValueError("TIGRESS FITS inputs differ from the frozen development run")
    split_path = FROZEN / "splits.csv"
    splits = pd.read_csv(split_path)
    west = (splits.loc[splits.role == "test", ["x", "y", "block"]]
            .drop_duplicates().sort_values(["y", "x"]).assign(split="train_pool"))
    y, x = np.mgrid[8:512:8, 384:512:8]
    good = (x % 64 != 0) & (y % 64 != 0)
    x, y = x[good], y[good]
    east = pd.DataFrame(dict(x=x, y=y, block=y // 64 * 8 + x // 64, split="test"))
    train, test = west[["x", "y"]].to_numpy(), east[["x", "y"]].to_numpy()
    nearest = cKDTree(test).query(train)[0].min()
    if (len(west) != 2296 or len(east) != 784 or west.x.max() != 368
            or nearest <= TIGRESS_BUFFER_PIX or west.block.nunique() != 48):
        raise ValueError("Frozen spatial split or separation check failed")

    metadata = dict(input_sha256=hashes, development_split_sha256=digest(split_path),
                    train_centers=len(west), test_centers=len(east), min_separation_pix=float(nearest))
    frame = pd.concat([west, east], ignore_index=True)
    x, y = frame.x.to_numpy(), frame.y.to_numpy()
    with fits.open(paths["emission_cube"]) as emission, fits.open(paths["truth_file"]) as truth:
        cube = emission["TB"]
        header = emission[0].header
        velocity = header["VMIN"] + header["DV"] * np.arange(cube.shape[0])
        mask, inner, outer = annulus_mask(cube.header["CDELT1"], cube.header["CDELT2"])
        assert tuple(cube.shape) == (201, 512, 512) and mask.sum() == 108
        for h in truth[1:3]:
            assert h.shape == cube.shape[1:]
        spectra, valid = map(np.stack, zip(*(
            extract_annulus(cube.data, px, py, mask) for px, py in zip(x, y)
        )))
        labels = np.column_stack([np.asarray(h.data[y, x], dtype=float) for h in truth[1:3]])

    descriptors = np.array([features(velocity, tb, good) for tb, good in zip(spectra, valid)])
    eligible = (valid.all(axis=1) & np.isfinite(descriptors).all(axis=1)
                & np.isfinite(labels).all(axis=1)
                & (labels[:, 0] >= 0) & (labels[:, 0] <= 1) & (labels[:, 1] > 0))
    if not eligible.all():
        raise ValueError(f"{(~eligible).sum()} frozen centers failed the common eligibility check")
    frame["score"] = frame.split == "test"
    metadata.update(annulus_pixels=int(mask.sum()), radii_pc=[inner, outer],
                    velocity_kms=[float(velocity[0]), float(velocity[-1]), float(header["DV"])])
    return frame, descriptors, labels, spectra, valid, velocity, metadata


def fit_draws(frame, descriptors, labels):
    train_pool = np.flatnonzero(~frame.score)
    test = np.flatnonzero(frame.score)
    draws, selections, predictions, metrics = [], [], [], []
    all_155 = []
    for size in SIZES:
        for draw, seed in enumerate(np.random.SeedSequence([SEED, size]).generate_state(DRAWS)):
            rng = np.random.default_rng(int(seed))
            train = np.sort(rng.choice(train_pool, size=size, replace=False))
            # Predict across the sparse grid for the 155-sightline map.
            predict_at = np.arange(len(frame)) if size == 155 else test
            fitted = ridge_predictions(descriptors[train], labels[train], descriptors[predict_at])
            if size == 155:
                all_155.append(fitted)
                fitted = fitted[test]
            mean = np.broadcast_to(labels[train].mean(axis=0), fitted.shape)
            record = dict(training_size=size, draw=draw, seed=int(seed))
            draws.append(record)
            selections.append(frame.iloc[train][["x", "y", "block"]].assign(**record))
            predictions.append(np.stack([fitted, mean]))
            for name, pred in (("Ridge", fitted), ("Train mean", mean)):
                for j, target in enumerate(TARGETS):
                    metrics.append(dict(**record, model=name, target=target,
                                        **scores(labels[test, j], pred[:, j])))
        print(f"Completed {DRAWS} draws at N={size}", flush=True)
    return (pd.DataFrame(draws), pd.concat(selections, ignore_index=True),
            np.stack(predictions), pd.DataFrame(metrics), np.mean(all_155, axis=0))


def cnn_predictions(test_spectra, dv):
    import torch

    torch.set_num_threads(2)
    sys.path.insert(0, str(ARCHIVE / "gpt/CNN_comparison/src"))
    from networks import MurrayCNN, TPCNetArchitecture

    verified, status = {}, {}
    for name, constructor in (("Murray-style CNN", MurrayCNN), ("TPCNet", TPCNetArchitecture)):
        relative, expected, n, spacing = CHECKPOINTS[name]
        path = ARCHIVE / relative
        status[name] = dict(checkpoint=relative, sha256=expected, channels=n, spacing_kms=spacing,
                            torch_version=str(torch.__version__))
        try:
            if digest(path) != expected:
                raise ValueError("Checkpoint digest differs from the verified archive")
            net = constructor().eval()
            state = torch.load(path, map_location="cpu", weights_only=True)
            if name == "TPCNet":
                state = {k: v for k, v in state["net"].items() if k != "pos_embedding"}
            net.load_state_dict(state, strict=True)
            target = (np.arange(n) - (n - 1) / 2) * spacing
            grid = np.arange(test_spectra.shape[1]) * dv
            formatted = np.stack([
                np.interp(target, grid - grid[np.argmax(spectrum)], spectrum, left=0, right=0)
                for spectrum in test_spectra
            ]).astype(np.float32)
            if name == "TPCNet":
                encoding = np.sin(np.linspace(0, 1, n, dtype=np.float32))
                formatted = (formatted + encoding)[:, None, None, :]
            else:
                formatted = formatted[:, None, :]
            with torch.inference_mode():
                pred = np.concatenate([net(batch).numpy()
                                       for batch in torch.from_numpy(formatted).split(128)])
            if pred.shape != (len(test_spectra), 2) or not np.isfinite(pred).all():
                raise ValueError("Checkpoint did not return two finite predictions per test input")
            status[name]["bounded_predictions"] = {
                "f_cnm": int(((pred[:, 0] < 0) | (pred[:, 0] > 1)).sum()),
                "r_hi": int((pred[:, 1] < 1).sum()),
            }
            verified[name] = np.column_stack([np.clip(pred[:, 0], 0, 1), np.maximum(pred[:, 1], 1)])
            status[name]["status"] = "verified"
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            status[name].update(status="unavailable", reason=str(error))
        print(f"{name}: {status[name]['status']}", flush=True)
    return verified, status


def summarize(per_draw):
    result = []
    metrics = ("mae", "rmse", "bias", "pearson_r", "ccc")
    for (size, model, target), group in per_draw.groupby(["training_size", "model", "target"]):
        record = dict(training_size=int(size), model=model, target=target)
        for metric in metrics:
            series = group[metric].dropna()
            record[metric] = dict(valid_draws=len(series), mean=float(series.mean()) if len(series)
                                  else None, sd=float(series.std()) if len(series) > 1 else None)
        result.append(record)
    return result


def score_155(labels, models):
    result = []
    for name, pred in models.items():
        for j, target in enumerate(TARGETS):
            y, p = labels[:, j], pred[:, j]
            masks = {"all": np.ones(len(y), dtype=bool)}
            masks.update({key: criterion(y) for key, criterion in REGIMES[target].items()})
            for regime, mask in masks.items():
                result.append(dict(model=name, target=target, regime=regime, **scores(y[mask], p[mask])))
    return result


def main():
    fixture()
    output = ROOT / "runs/tigress" / RUN_ID
    if output.exists():
        raise FileExistsError(output)
    frame, descriptors, labels, spectra, valid, velocity, metadata = inputs()
    output.mkdir(parents=True)
    draws, selections, predictions, per_draw, ensemble = fit_draws(frame, descriptors, labels)
    mask = frame.score.to_numpy()
    test = frame[mask].reset_index(drop=True)
    models = {"Ridge, 155-draw ensemble": ensemble[mask]}
    cnns, status = cnn_predictions(spectra[mask], float(velocity[1] - velocity[0]))
    models.update(cnns)
    models["Unity"] = np.column_stack([np.full(len(test), np.nan), np.ones(len(test))])
    comparison = score_155(labels[mask], {name: p for name, p in models.items() if name != "Unity"})
    comparison += [row for row in score_155(labels[mask], {"Unity": models["Unity"]})
                   if row["target"] == "r_hi"]
    summary = summarize(per_draw)
    metadata.update(run_id=RUN_ID, seed=SEED, sizes=list(SIZES), draws_per_size=DRAWS,
                    ridge_alpha=10, python=sys.version.split()[0], sklearn=sklearn.__version__,
                    cnn=status, code_sha256={name: digest(ROOT / "scripts" / name) for name in
                        ("run_tigress_benchmark.py", "evaluate_development.py", "report_tigress.py",
                         "check_tigress_mock.py")})
    per_draw.to_csv(output / "tigress_draw_metrics.csv", index=False)
    draws.to_csv(output / "draws.csv", index=False)
    selections.to_csv(output / "training_draws.csv", index=False)
    frame.to_csv(output / "frozen_split.csv", index=False)
    np.savez_compressed(output / "matched_test_inputs.npz", xy=test[["x", "y"]].to_numpy(),
                        velocity_kms=velocity, tb_k=spectra[mask], valid=valid[mask])
    np.savez_compressed(output / "draw_predictions.npz", predictions=predictions,
                        models=np.array(["Ridge", "Train mean"]), targets=np.array(TARGETS),
                        xy=test[["x", "y"]].to_numpy(), truth=labels[mask],
                        training_size=draws.training_size.to_numpy(), draw=draws.draw.to_numpy())
    tables = []
    for name, pred in models.items():
        for j, target in enumerate(TARGETS):
            if name == "Unity" and j == 0:
                continue
            tables.append(test.assign(target=target, model=name, training_size=155 if name.startswith("Ridge") else None,
                                      draw_id="mean_of_100" if name.startswith("Ridge") else None,
                                      truth=labels[mask, j], prediction=pred[:, j]))
    pd.concat(tables, ignore_index=True).to_csv(output / "tigress_predictions.csv", index=False)
    mapped = frame.copy()
    for j, target in enumerate(TARGETS):
        mapped[f"{target}_truth"] = labels[:, j]
        mapped[f"{target}_prediction"] = ensemble[:, j]
    mapped.to_csv(output / "tigress_map_grid.csv", index=False)
    (output / "tigress_metrics.json").write_text(json.dumps(
        dict(settings=metadata, training_size_summary=summary, same_test_comparison=comparison),
        indent=2, allow_nan=False) + "\n")
    write_report(output, mapped, labels[mask], models, summary, comparison, metadata)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
