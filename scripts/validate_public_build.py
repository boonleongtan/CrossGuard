#!/usr/bin/env python
"""Validate a local public-data build before training or scoring.

The judge-controlled reproduction path consumes two local artifacts:

* a manifest parquet/csv with at least ``path``, ``label``, ``split`` and
  ``sha256`` columns;
* an image root containing the files referenced by ``path``.

This checker performs fast structural checks and an optional sha256 sample. It
does not contact Modal, Hugging Face, ModelScope, Kaggle, or any private store.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"path", "label", "split", "sha256"}
VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {0, 1}
FORBIDDEN_PATH_SUBSTRINGS = (
    "coco2017/val2017/",
    "/DALLE/Advanced/",
    "/Advanced/DALLE3/",
    "DALLE/Advanced/",
)


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def resolve_image_path(image_root: Path, manifest_path: str) -> Path:
    # Sharded rows look like full/shard-00000.parquet#123 and are not loose
    # files. The public README path uses loose canonical images, so fail
    # plainly if a judge points this checker at a pre-materialized manifest.
    if "#" in manifest_path:
        raise FileNotFoundError(
            f"{manifest_path!r} is a shard row, not a loose image path")
    return image_root / manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.parquet"))
    parser.add_argument("--image-root", type=Path, default=Path("data/full"))
    parser.add_argument("--sample", type=int, default=256,
                        help="number of rows to hash-check; 0 checks structure only")
    args = parser.parse_args()

    df = read_manifest(args.manifest)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"manifest missing required columns: {sorted(missing)}")
    if len(df) == 0:
        raise SystemExit("manifest is empty")

    splits = set(df["split"].astype(str))
    labels = {int(v) for v in df["label"].dropna().unique()}
    if not splits <= VALID_SPLITS:
        raise SystemExit(f"unexpected split values: {sorted(splits - VALID_SPLITS)}")
    if not labels <= VALID_LABELS:
        raise SystemExit(f"unexpected label values: {sorted(labels - VALID_LABELS)}")

    paths = df["path"].astype(str)
    forbidden = paths.str.replace("\\", "/", regex=False).apply(
        lambda p: any(token in p for token in FORBIDDEN_PATH_SUBSTRINGS))
    if bool(forbidden.any()):
        raise SystemExit(
            f"manifest contains {int(forbidden.sum())} forbidden validation paths")

    subset = df if args.sample <= 0 else df.sample(
        n=min(args.sample, len(df)), random_state=0)
    missing_files = []
    bad_hashes = []
    for row in subset.itertuples(index=False):
        rel = str(getattr(row, "path"))
        try:
            image_path = resolve_image_path(args.image_root, rel)
        except FileNotFoundError as exc:
            missing_files.append(str(exc))
            continue
        if not image_path.exists():
            missing_files.append(str(image_path))
            continue
        expected = str(getattr(row, "sha256"))
        if expected:
            got = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if got != expected:
                bad_hashes.append(f"{rel}: expected {expected[:12]}, got {got[:12]}")

    if missing_files:
        raise SystemExit(
            f"{len(missing_files)} sampled image paths are missing; first: "
            f"{missing_files[0]}")
    if bad_hashes:
        raise SystemExit(
            f"{len(bad_hashes)} sampled image hashes mismatch; first: "
            f"{bad_hashes[0]}")

    print(f"manifest rows: {len(df)}")
    print(f"splits: {df['split'].value_counts().to_dict()}")
    print(f"labels: {df['label'].value_counts().to_dict()}")
    print(f"sha256 sample checked: {len(subset)}")
    print("public build validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
