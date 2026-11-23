"""Extract a few native-resolution TIGRESS annular spectra and central references."""

import json
from pathlib import Path

from astropy.io import fits
import numpy as np

from check_inputs import DEFAULT_MANIFEST


DISTANCE_PC = 50_000
RADII_ARCSEC = (28, 56)
CENTERS_XY = ((128, 128), (256, 256), (384, 320))


def annulus_mask(dx, dy):
    inner, outer = DISTANCE_PC * np.deg2rad(np.array(RADII_ARCSEC) / 3600)
    dx, dy = abs(dx), abs(dy)
    rx, ry = int(outer / dx), int(outer / dy)
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    radius = np.hypot(xx * dx, yy * dy)
    return (radius >= inner) & (radius <= outer), inner, outer


def extract_annulus(cube, x, y, mask):
    ry, rx = (size // 2 for size in mask.shape)
    _, ny, nx = cube.shape
    if x - rx < 0 or x + rx >= nx or y - ry < 0 or y + ry >= ny:
        raise ValueError("Annulus extends beyond the cube")
    cutout = cube[:, y - ry:y + ry + 1, x - rx:x + rx + 1]
    pixels = cutout[:, mask]
    valid = np.isfinite(pixels).all(axis=1)
    spectrum = np.full(cube.shape[0], np.nan)
    spectrum[valid] = pixels[valid].mean(axis=1)
    return spectrum, valid


def main():
    inputs = json.loads(DEFAULT_MANIFEST.read_text())["tigress"]
    with (
        fits.open(DEFAULT_MANIFEST.parent / inputs["emission_cube"]) as cube,
        fits.open(DEFAULT_MANIFEST.parent / inputs["truth_file"]) as truth,
    ):
        tb, tau = cube["TB"], cube["TAU"]
        header = tb.header
        axes = [f"{key}{axis}" for axis in (1, 2, 3)
                for key in ("CTYPE", "CUNIT", "CRVAL", "CRPIX", "CDELT")]
        if tb.shape != tau.shape or any(header[key] != tau.header[key] for key in axes):
            raise ValueError("TB and TAU grids disagree")
        if tuple(header[f"CTYPE{i}"] for i in (1, 2, 3)) != ("x", "y", "v"):
            raise ValueError("Unexpected cube orientation")
        fcnm, rhi = truth[1].data, truth[2].data
        if fcnm.shape != tb.shape[1:] or rhi.shape != tb.shape[1:]:
            raise ValueError("Truth footprint differs from the emission grid")

        # The generator stores the first channel in VMIN and uses zero-based CRPIX3.
        primary = cube[0].header
        velocity = primary["VMIN"] + primary["DV"] * np.arange(tb.shape[0])
        np.testing.assert_allclose(
            [velocity[0], velocity[-1], primary["DV"]],
            [header["CRVAL3"], primary["VMAX"], header["CDELT3"]],
        )

        mask, inner, outer = annulus_mask(dx=header["CDELT1"], dy=header["CDELT2"])
        ry, rx = (size // 2 for size in mask.shape)

        fixture = np.broadcast_to(np.array([1.0, 2.0])[:, None, None], (2, *mask.shape)).copy()
        fixture[:, ry, rx] = 100
        np.testing.assert_array_equal(extract_annulus(fixture, rx, ry, mask)[0], [1, 2])
        try:
            extract_annulus(fixture, 0, 0, mask)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete aperture accepted")

        x, y = np.array(CENTERS_XY).T
        labels = np.column_stack((fcnm[y, x], rhi[y, x]))
        if not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels[:, 0] > 1):
            raise ValueError("Invalid central truth")
        spectra, valid = map(np.stack, zip(*(
            extract_annulus(tb.data, px, py, mask) for px, py in CENTERS_XY
        )))
        central_tau = tau.data[:, y, x].T
        if not valid.all() or not np.isfinite(central_tau).all():
            raise ValueError("Incomplete spectral coverage")

        output = DEFAULT_MANIFEST.parents[1] / "data/derived/tigress/annulus-smoke-v01"
        output.mkdir(parents=True, exist_ok=True)
        with (output / "sightlines.npz").open("xb") as file:
            np.savez_compressed(
                file, xy=np.array(CENTERS_XY), velocity_kms=velocity, tb_k=spectra,
                valid=valid, tau=central_tau, f_cnm=labels[:, 0], r_hi=labels[:, 1],
                annulus_mask=mask,
            )
        print(f"Annulus {inner:.3f}–{outer:.3f} pc, {mask.sum()} pixels")
        print(f"Velocity {velocity[0]} to {velocity[-1]} km/s in {len(velocity)} channels")
        for xy, spectrum, optical_depth, (fraction, correction) in zip(
            CENTERS_XY, spectra, central_tau, labels
        ):
            print(f"{xy}  peak TB={spectrum.max():.4f} K  peak tau={optical_depth.max():.4f}"
                  f"  f_CNM={fraction:.5f}  R_HI={correction:.5f}")


if __name__ == "__main__":
    main()
