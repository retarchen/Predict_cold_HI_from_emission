"""Inventory local inputs and cross-match catalog IDs without loading FITS arrays."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "config/data_sources.json"


def catalog_ids(path, id_column, required_columns):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")
        ids = [(row[id_column] or "").strip() for row in reader]
    counts = Counter(ids)
    return {
        "rows": len(ids),
        "unique_nonblank_ids": len(counts) - int("" in counts),
        "blank_ids": counts[""],
        "duplicate_nonblank_ids": sum(
            1 for source, count in counts.items() if source and count > 1
        ),
    }, {source for source in counts if source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    manifest = parser.parse_args().manifest.resolve()

    try:
        sources = json.loads(manifest.read_text(encoding="utf-8"))
        file_specs = (
            ("lmc", "absorption_catalog"),
            ("lmc", "thin_column_catalog"),
            ("lmc", "emission_cube"),
            ("tigress", "emission_cube"),
            ("tigress", "truth_file"),
        )
        paths = {
            (domain, name): manifest.parent / sources[domain][name]
            for domain, name in file_specs
        }
        spectra_dir = manifest.parent / sources["lmc"]["emission_spectra_dir"]
        report = {"files": {}}
        for (domain, name), path in paths.items():
            size = path.stat().st_size if path.is_file() else 0
            report["files"][f"{domain}.{name}"] = {
                "exists": path.is_file(),
                "bytes": size,
            }

        if not spectra_dir.is_dir() or any(
            item["bytes"] == 0 for item in report["files"].values()
        ):
            report["spectra_directory_exists"] = spectra_dir.is_dir()
            report["status"] = "missing_or_empty_inputs"
            print(json.dumps(report, indent=2))
            return 1

        absorption, lmc_ids = catalog_ids(
            paths["lmc", "absorption_catalog"],
            "Name",
            {"Name", "RA_deg", "Dec_deg", "f_c", "NHI_c", "NHI_w", "NHI_u"},
        )
        thin, thin_ids = catalog_ids(
            paths["lmc", "thin_column_catalog"],
            "Source",
            {"Source", "NHI_uncorr_mean"},
        )
        spectra_ids = {path.stem for path in spectra_dir.glob("*.txt") if path.is_file()}
        missing_spectra = len(lmc_ids - spectra_ids)
        missing_thin = len(lmc_ids - thin_ids)
        report.update({
            "absorption_catalog": absorption,
            "thin_column_catalog": thin,
            "spectra_files": len(spectra_ids),
            "lmc_ids_with_both_matches": len(lmc_ids & thin_ids & spectra_ids),
            "lmc_ids_missing_spectrum": missing_spectra,
            "lmc_ids_missing_thin_column": missing_thin,
        })
        incomplete = (
            not lmc_ids or missing_spectra or missing_thin
            or any(catalog[key] for catalog in (absorption, thin)
                   for key in ("blank_ids", "duplicate_nonblank_ids"))
        )
        report["status"] = "review_catalog_links" if incomplete else "inventory_pass"
        print(json.dumps(report, indent=2))
        return int(bool(incomplete))
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
        parser.exit(1, f"Input inventory failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
