"""Vendor the real tabular datasets into the repository, once, as CSV.

Why vendor rather than depend. This repository's value is that a reader can
re-derive every number in it from committed files. A dataset fetched at run
time from a host that may change, rate-limit or disappear is not that, and a
dataset loaded through a library is a dataset whose preprocessing depends on
that library's version. The container these sessions run in cannot reach UCI or
OpenML at all (the network policy answers 403 to both), which settles it: the
data is extracted once, written as plain CSV with a checksum, and committed.

scikit-learn is used here as a *delivery mechanism* for datasets it bundles, and
for nothing else. It is a development dependency of this script, is not imported
by any module under `src/`, and no model, metric, split or statistic in this
repository comes from it. The bundled copies are themselves copies of UCI
datasets; the manifest records that chain so a reader can go to the original.

Run with `make vendor-data`. It is idempotent and refuses to overwrite a file
whose checksum already matches, so re-running it cannot silently change data a
release was built on.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tabular"

#: Provenance for each bundled dataset: where it really comes from, and what a
#: reader should cite. scikit-learn is the courier, not the source.
SOURCES = {
    "iris": {
        "title": "Iris plants",
        "origin": "UCI Machine Learning Repository, 'Iris' (Fisher, 1936)",
        "url": "https://archive.ics.uci.edu/dataset/53/iris",
        "note": "Four measurements of 150 flowers, three species, perfectly balanced.",
    },
    "wine": {
        "title": "Wine recognition",
        "origin": "UCI Machine Learning Repository, 'Wine' (Forina et al.)",
        "url": "https://archive.ics.uci.edu/dataset/109/wine",
        "note": "Thirteen chemical measurements of 178 wines from three cultivars.",
    },
    "breast_cancer": {
        "title": "Breast cancer Wisconsin (diagnostic)",
        "origin": "UCI Machine Learning Repository, 'WDBC' (Street, Wolberg, Mangasarian)",
        "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "note": "Thirty features computed from digitised images of 569 fine-needle "
                "aspirates; the binary member of the battery.",
    },
    "digits": {
        "title": "Optical recognition of handwritten digits",
        "origin": "UCI Machine Learning Repository, 'Optdigits' test set (Alpaydin, Kaynak)",
        "url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "note": "8x8 integer pixel intensities, 1797 samples, ten classes; the "
                "widest and the most classes in the battery.",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, X: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([*names, "target"])
        for row, label in zip(X, y):
            # repr round-trips a float64 exactly; "%.17g" would too, but repr
            # keeps the short form when the short form is exact, so the file
            # stays readable without losing a bit.
            w.writerow([repr(float(v)) for v in row] + [int(label)])


def main() -> int:
    try:
        from sklearn import datasets as skd
        from sklearn import __version__ as skversion
    except ImportError:
        print(
            "scikit-learn is needed to extract the datasets, and only to extract "
            "them: `uv pip install --python .venv/bin/python scikit-learn`.\n"
            "The committed CSVs under data/tabular/ are what every module reads, "
            "so this is a one-time step a reader never has to repeat.",
            file=sys.stderr,
        )
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "extracted_by": "bench/vendor_datasets.py",
        "courier": f"scikit-learn {skversion} (bundled copies; not used for anything else)",
        "datasets": [],
    }

    for name, source in SOURCES.items():
        bundle = getattr(skd, f"load_{name}")()
        X = np.asarray(bundle.data, dtype=np.float64)
        y = np.asarray(bundle.target, dtype=np.int64)
        names = [str(f).strip().replace(",", " ") for f in bundle.feature_names]
        assert len(names) == X.shape[1], name

        path = OUT / f"{name}.csv"
        before = _sha256(path) if path.exists() else None
        _write_csv(path, X, y, names)
        digest = _sha256(path)
        if before is not None and before != digest:
            print(
                f"REFUSED: {path.name} already existed with a different checksum.\n"
                f"  committed {before}\n  extracted {digest}\n"
                "A release may have been built on the committed file. Investigate "
                "rather than overwrite; restore it with `git checkout`.",
                file=sys.stderr,
            )
            return 1

        counts = np.bincount(y).tolist()
        manifest["datasets"].append(
            {
                "name": name,
                "file": f"{path.relative_to(ROOT)}",
                "sha256": digest,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_classes": int(len(counts)),
                "class_counts": counts,
                "feature_names": names,
                **source,
            }
        )
        print(f"{name:15s} {X.shape[0]:5d}x{X.shape[1]:<3d} {len(counts):2d} classes  {digest[:16]}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
