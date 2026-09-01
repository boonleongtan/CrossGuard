#!/usr/bin/env python
"""Merge local public-data manifest parts into one training manifest.

This script never contacts Hugging Face, Modal, or any team-controlled artifact
store. It is for judge-controlled reproduction after the public datasets have
been extracted/canonicalized into local manifest parts.

Examples:

    python scripts/merge_manifests.py --inputs data/manifests/*.parquet
    python scripts/merge_manifests.py --inputs data/sid/manifest.parquet data/wildfake/manifest.parquet

The output defaults to ``data/manifest.parquet``, which is the path used by the
README training commands.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigid.canon import MANIFEST_COLUMNS


def expand_inputs(patterns: list[str], output: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            p = Path(pattern)
            if p.exists():
                paths.append(p)
    out = output.resolve()
    unique = sorted({p.resolve() for p in paths if p.resolve() != out})
    if not unique:
        raise SystemExit(
            "no input manifests found. Pass --inputs with local .parquet or "
            ".csv manifests built from public dataset copies.")
    return unique


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"unsupported manifest extension: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", default=["data/manifests/*.parquet"],
        help="local manifest files or glob patterns")
    parser.add_argument(
        "--out", type=Path, default=Path("data/manifest.parquet"),
        help="merged manifest path")
    args = parser.parse_args()

    inputs = expand_inputs(args.inputs, args.out)
    frames = []
    for path in inputs:
        df = read_manifest(path)
        print(f"{path}: {len(df)} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    missing_required = {"path", "label", "split", "sha256"} - set(merged.columns)
    if missing_required:
        raise SystemExit(
            f"merged manifest is missing required columns: "
            f"{sorted(missing_required)}")

    for col in MANIFEST_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
    merged = merged[MANIFEST_COLUMNS]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f"\nwrote {len(merged)} rows -> {args.out}")
    print(f"splits: {merged['split'].value_counts().to_dict()}")
    print(f"labels: {merged['label'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
