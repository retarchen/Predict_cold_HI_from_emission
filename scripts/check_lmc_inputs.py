"""Audit matched LMC labels and compare saved emission with three FITS cutouts."""

import json

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import numpy as np
import pandas as pd

from check_inputs import DEFAULT_MANIFEST


CUTOUT_SOURCES = ("J043855-672153", "J044047-695217", "J044056-662423")


def main():
    root = DEFAULT_MANIFEST.parent
    paths = {key: root / value for key, value in
             json.loads(DEFAULT_MANIFEST.read_text())["lmc"].items()}
    labels = pd.read_csv(paths["absorption_catalog"]).merge(
        pd.read_csv(paths["thin_column_catalog"]), left_on="Name", right_on="Source",
        how="left", validate="one_to_one",
    )
    phases = labels[["NHI_c", "NHI_w", "NHI_u"]]
    total = phases.sum(axis=1, min_count=3)
    labels["recomputed_f_c"] = labels["NHI_c"] / total.where(total > 0)
    labels["f_c_difference"] = labels["f_c"] - labels["recomputed_f_c"]
    thin = labels["NHI_uncorr_mean"]
    labels["recomputed_r_hi"] = total * 1e20 / thin.where(thin > 0)

    saved = {name: np.loadtxt(paths["emission_spectra_dir"] / f"{name}.txt", ndmin=2)
             for name in labels["Name"]}
    for name, spectrum in saved.items():
        if spectrum.shape[1] != 3 or not np.isfinite(spectrum).all():
            raise ValueError(f"Invalid three-column spectrum for {name}")
        np.testing.assert_allclose(np.diff(spectrum[:, 0]), 1.0, atol=2e-6)
    labels["saved_tb_integral_cm2"] = [
        1.823e18 * np.trapezoid(s[:, 1], s[:, 0]) for s in saved.values()
    ]
    labels["saved_thin_fractional_difference"] = labels["saved_tb_integral_cm2"] / thin - 1

    reference = paths["emission_spectra_dir"].parent / (
        "LMC_abs_new/abs_v1.0/new_abs_catalog/gaskap_abs_spectra_v1.1.vot"
    )
    catalog = Table.read(reference)
    comparison = []
    with fits.open(paths["emission_cube"], memmap=True) as hdus:
        cube = hdus[0]
        header = cube.header
        celestial = WCS(header).celestial
        velocity = (header["CRVAL3"] + header["CDELT3"] *
                    (np.arange(cube.shape[0]) + 1 - header["CRPIX3"])) / 1000
        for name in CUTOUT_SOURCES:
            rows = catalog[catalog["comp_name"] == name]
            if len(rows) != 1:
                raise ValueError(f"Expected one precise coordinate for {name}")
            x, y = celestial.world_to_pixel_values(rows["ra"][0], rows["dec"][0])
            ix, iy = int(round(float(x))), int(round(float(y)))
            radius_pix = 9
            yy, xx = np.mgrid[iy-radius_pix:iy+radius_pix+1, ix-radius_pix:ix+radius_pix+1]
            radius_arcsec = np.hypot(xx - x, yy - y) * abs(header["CDELT2"]) * 3600
            annulus = (radius_arcsec >= 28) & (radius_arcsec <= 56)
            pixels = cube.data[:, yy[annulus], xx[annulus]]
            spectrum = saved[name]
            cube_channels = np.abs(spectrum[:, 0, None] - velocity).argmin(axis=1)
            offset = spectrum[:, 0] - velocity[cube_channels]
            valid = (np.abs(offset) < 1e-4) & np.isfinite(pixels[cube_channels]).all(axis=1)
            if not valid.any():
                raise ValueError(f"No aligned channels for {name}")
            mean = pixels[cube_channels[valid]].mean(axis=1)
            difference = mean - spectrum[valid, 1]
            comparison.append({
                "source": name, "annulus_pixels": int(annulus.sum()),
                "matched_channels": int(valid.sum()),
                "max_velocity_difference_kms": float(np.max(np.abs(offset[valid]))),
                "tb_rms_difference_k": float(np.sqrt(np.mean(difference**2))),
                "tb_max_difference_k": float(np.max(np.abs(difference))),
            })

    output = DEFAULT_MANIFEST.parents[1] / "runs/lmc/input-validation-v01"
    output.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output / "label_audit.csv", index=False, mode="x")
    with (output / "cutouts.json").open("x") as file:
        json.dump(comparison, file, indent=2)
    print(labels[["f_c_difference", "recomputed_r_hi", "saved_thin_fractional_difference"]].describe())
    print("Missing phase values:", phases.isna().sum().to_dict())
    print("Zero phase values:", phases.eq(0).sum().to_dict())
    print("Missing/nonpositive thin columns:", int((thin.isna() | (thin <= 0)).sum()))
    print("Ratios below unity:", int((labels["recomputed_r_hi"] < 1).sum()))
    print("Absolute fraction discrepancies above 0.01:", int((labels["f_c_difference"].abs() > .01).sum()))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
