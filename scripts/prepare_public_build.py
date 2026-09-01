#!/usr/bin/env python
"""Build the public CrossGuard training manifest and loose image root.

This is the reproducible data-build step between downloading the public source
datasets and running ``aigid.train``. It consumes public local mirrors only:

* Hugging Face parquet mirrors of ``saberzl/SID_Set`` and
  ``dragonintelligence/CIFAKE-image-dataset``;
* WildFake zip archives downloaded by ``scripts/download_wildfake.py``.

The output contract matches the README training commands:

* ``data/manifest.parquet``
* ``data/full/<manifest.path>``

Use ``--download-hf`` if SID_Set and CIFAKE are not already mirrored locally.
WildFake is intentionally handled by ``scripts/download_wildfake.py`` because
those archives are large and resumable range downloads matter.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigid import canon  # noqa: E402
from aigid.quarantine import Screen  # noqa: E402

FORBIDDEN_PATH_SUBSTRINGS = (
    "coco2017/val2017/",
    "/DALLE/Advanced/",
    "/Advanced/DALLE3/",
    "DALLE/Advanced/",
)


@dataclass(frozen=True)
class StagedImage:
    dataset: str
    source_key: str
    label: int
    generator: str
    architecture: str
    generator_licence: str
    content: str
    real_source: str
    upstream_split: str
    raw_sha256: str
    raw_id: str
    pool: str


@dataclass
class BuildStats:
    accepted: int = 0
    dropped_forbidden_path: int = 0
    dropped_quarantine: int = 0
    review_quarantine: int = 0
    decode_errors: int = 0
    skipped_labels: int = 0
    missing_archives: int = 0


def _safe_part(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "unknown"


def _forbidden_path(path: str) -> bool:
    norm = path.replace("\\", "/")
    return any(token in norm for token in FORBIDDEN_PATH_SUBSTRINGS)


def _open_raw_image(raw: bytes) -> tuple[Image.Image, bool]:
    img = Image.open(io.BytesIO(raw))
    fmt = (img.format or "").upper()
    img.load()
    return img.convert("RGB"), fmt in {"JPEG", "JPG"}


def _image_cell_bytes(cell, parquet_path: Path) -> bytes:
    """Return bytes from a Hugging Face parquet image cell."""
    if isinstance(cell, bytes):
        return cell
    if isinstance(cell, bytearray):
        return bytes(cell)
    if isinstance(cell, memoryview):
        return cell.tobytes()
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is not None:
            return bytes(raw)
        img_path = cell.get("path")
        if img_path:
            p = Path(img_path)
            if not p.is_absolute():
                p = parquet_path.parent / p
            return p.read_bytes()
    raise TypeError(f"unsupported image cell from {parquet_path}: {type(cell).__name__}")


def _parquet_split(path: Path) -> str:
    name = path.stem
    return name.split("-", 1)[0] if "-" in name else name


def _iter_parquet_candidates(dataset: str, root: Path,
                             batch_size: int) -> Iterable[tuple[StagedImage, bytes]]:
    import pyarrow.parquet as pq

    spec = canon.DATASETS[dataset]
    files = sorted((root / "data").glob("*.parquet")) or sorted(root.glob("*.parquet"))
    for parquet_path in files:
        upstream_split = _parquet_split(parquet_path)
        pf = pq.ParquetFile(parquet_path)
        top_level = set(pf.schema_arrow.names)
        cols = [c for c in spec.columns if c in top_level]
        if "image" not in cols or "label" not in cols:
            raise SystemExit(f"{parquet_path} must contain image and label columns")
        for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
            df = batch.to_pandas()
            for row_idx, row in df.iterrows():
                upstream_label = int(row["label"])
                source = spec.sources.get(upstream_label)
                if source is None:
                    continue
                raw = _image_cell_bytes(row["image"], parquet_path)
                raw_sha = hashlib.sha256(raw).hexdigest()
                raw_id = str(row.get("img_id", f"{parquet_path.name}:{row_idx}"))
                yield StagedImage(
                    dataset=dataset,
                    source_key=source.key,
                    label=source.label,
                    generator=source.generator,
                    architecture=source.architecture,
                    generator_licence=source.licence,
                    content=source.content,
                    real_source=source.real_source,
                    upstream_split=upstream_split,
                    raw_sha256=raw_sha,
                    raw_id=raw_id,
                    pool=spec.pool,
                ), raw


def _matching_source(prefixes: dict[str, canon.Source], member: str):
    for prefix, source in prefixes.items():
        if prefix == "" or member.startswith(prefix):
            return source
    return None


def _iter_wildfake_candidates(root: Path) -> Iterable[tuple[StagedImage, bytes]]:
    for dataset, spec in canon.DATASETS.items():
        if spec.kind != "zip":
            continue
        for archive_name, prefixes in spec.archives.items():
            archive_path = root / archive_name
            if not archive_path.exists():
                raise FileNotFoundError(archive_path)
            with zipfile.ZipFile(archive_path) as zf:
                for member in sorted(zf.namelist()):
                    if member.endswith("/") or member.startswith("__MACOSX/"):
                        continue
                    source = _matching_source(prefixes, member)
                    if source is None:
                        continue
                    raw = zf.read(member)
                    raw_sha = hashlib.sha256(raw).hexdigest()
                    yield StagedImage(
                        dataset=dataset,
                        source_key=source.key,
                        label=source.label,
                        generator=source.generator,
                        architecture=source.architecture,
                        generator_licence=source.licence,
                        content=source.content,
                        real_source=source.real_source,
                        upstream_split="",
                        raw_sha256=raw_sha,
                        raw_id=f"{archive_name}:{member}",
                        pool=spec.pool,
                    ), raw


def _maybe_download_hf(args) -> None:
    if not args.download_hf:
        return
    from huggingface_hub import snapshot_download

    for dataset, root in (("SID_Set", args.sid_root), ("CIFAKE", args.cifake_root)):
        spec = canon.DATASETS[dataset]
        snapshot_download(
            repo_id=spec.repo,
            repo_type="dataset",
            local_dir=root,
            allow_patterns=["data/*.parquet", "README.md", "config.json"],
        )


def _screened(items: Iterable[tuple[StagedImage, bytes]], screen: Screen | None,
              stats: BuildStats, limit: int) -> Iterable[tuple[StagedImage, bytes]]:
    seen_by_dataset: dict[str, int] = {}
    for staged, raw in items:
        if limit > 0 and seen_by_dataset.get(staged.dataset, 0) >= limit:
            continue
        if _forbidden_path(staged.raw_id):
            stats.dropped_forbidden_path += 1
            continue
        if screen is not None:
            verdict, _, _ = screen.verdict(raw)
            if verdict == "drop":
                stats.dropped_quarantine += 1
                continue
            if verdict == "review":
                stats.review_quarantine += 1
        seen_by_dataset[staged.dataset] = seen_by_dataset.get(staged.dataset, 0) + 1
        stats.accepted += 1
        yield staged, raw


def _all_candidates(args) -> Iterable[tuple[StagedImage, bytes]]:
    if not args.skip_sid:
        yield from _iter_parquet_candidates("SID_Set", args.sid_root, args.batch_size)
    if not args.skip_cifake:
        yield from _iter_parquet_candidates("CIFAKE", args.cifake_root, args.batch_size)
    if not args.skip_wildfake:
        yield from _iter_wildfake_candidates(args.wildfake_root)


def _collect_geometry(args, screen: Screen | None) -> tuple[dict[str, list[tuple[int, int]]], BuildStats]:
    stats = BuildStats()
    sizes: dict[str, list[tuple[int, int]]] = {}
    for staged, raw in _screened(_all_candidates(args), screen, stats, args.limit_per_dataset):
        try:
            with Image.open(io.BytesIO(raw)) as img:
                img.load()
                sizes.setdefault(staged.pool, []).append(img.size)
        except Exception:
            stats.decode_errors += 1
    return sizes, stats


def _target_pools(sizes: dict[str, list[tuple[int, int]]]) -> dict[str, tuple[list[float], list[int]]]:
    pools = {}
    for pool, pool_sizes in sizes.items():
        ars, ses = canon.pooled_targets(pool_sizes)
        if not ars or not ses:
            raise SystemExit(f"geometry pool {pool!r} has no decodable images")
        pools[pool] = (ars, ses)
    return pools


def _canonicalize(raw: bytes, staged: StagedImage,
                  pools: dict[str, tuple[list[float], list[int]]],
                  seed: int) -> tuple[bytes, str, int]:
    img, arrived_jpeg = _open_raw_image(raw)
    ars, ses = pools[staged.pool]
    rng = np.random.default_rng(seed ^ int(staged.raw_sha256[:16], 16))
    target_ar = float(ars[int(rng.integers(0, len(ars)))])
    target_se = int(ses[int(rng.integers(0, len(ses)))])
    out = canon.canonicalize(img, arrived_jpeg, target_ar, target_se, rng)
    sha, phash = canon.hashes(out)
    return out, sha, phash


def _write_build(args, screen: Screen | None,
                 pools: dict[str, tuple[list[float], list[int]]]) -> tuple[pd.DataFrame, BuildStats]:
    stats = BuildStats()
    rows = []
    for staged, raw in _screened(_all_candidates(args), screen, stats, args.limit_per_dataset):
        try:
            image_bytes, sha, phash = _canonicalize(raw, staged, pools, args.seed)
        except Exception as exc:  # noqa: BLE001 - bad public input should be counted and skipped
            stats.decode_errors += 1
            if args.strict_decode:
                raise SystemExit(f"{staged.raw_id}: {type(exc).__name__}: {exc}") from exc
            continue

        source_dir = _safe_part(staged.source_key)
        rel = Path(staged.dataset) / source_dir / sha[:2] / f"{sha}.jpg"
        dest = args.image_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(image_bytes)

        rows.append({
            "path": rel.as_posix(),
            "shard": "",
            "row_idx": "",
            "label": staged.label,
            "source": staged.source_key,
            "generator": staged.generator,
            "architecture": staged.architecture,
            "generator_licence": staged.generator_licence,
            "content": staged.content,
            "real_source": staged.real_source,
            "upstream_split": staged.upstream_split,
            "split": canon.assign_split(
                staged.dataset, staged.upstream_split, staged.raw_sha256,
                staged.label, staged.generator),
            "sha256": sha,
            "phash": str(phash),
        })

    df = pd.DataFrame(rows)
    for col in canon.MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[canon.MANIFEST_COLUMNS], stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-hf", action="store_true",
                        help="download/mirror SID_Set and CIFAKE from Hugging Face")
    parser.add_argument("--sid-root", type=Path, default=Path("data/sources/SID_Set"))
    parser.add_argument("--cifake-root", type=Path, default=Path("data/sources/CIFAKE"))
    parser.add_argument("--wildfake-root", type=Path, default=Path("data/wildfake_zips"))
    parser.add_argument("--image-root", type=Path, default=Path("data/full"))
    parser.add_argument("--out", type=Path, default=Path("data/manifest.parquet"))
    parser.add_argument("--quarantine", type=Path, default=Path("manifest/quarantine.npz"))
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="parquet rows per read batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-per-dataset", type=int, default=0,
                        help="smoke-test cap per dataset; 0 builds everything")
    parser.add_argument("--skip-sid", action="store_true")
    parser.add_argument("--skip-cifake", action="store_true")
    parser.add_argument("--skip-wildfake", action="store_true")
    parser.add_argument("--strict-decode", action="store_true",
                        help="fail on the first undecodable source image")
    args = parser.parse_args()

    _maybe_download_hf(args)
    screen = Screen(str(args.quarantine)) if args.quarantine.exists() else None
    if screen is None:
        print(f"warning: quarantine index not found at {args.quarantine}; content screen disabled",
              file=sys.stderr)

    print("pass 1/2: collecting pooled geometry")
    sizes, geo_stats = _collect_geometry(args, screen)
    for pool, pool_sizes in sorted(sizes.items()):
        print(f"  {pool}: {len(pool_sizes)} decodable image(s)")
    pools = _target_pools(sizes)
    if geo_stats.decode_errors:
        print(f"  skipped {geo_stats.decode_errors} undecodable image(s) in pass 1",
              file=sys.stderr)

    print("pass 2/2: writing canonical images and manifest")
    manifest, write_stats = _write_build(args, screen, pools)
    if manifest.empty:
        raise SystemExit("no manifest rows written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out, index=False)

    print(f"\nwrote {len(manifest)} rows -> {args.out}")
    print(f"images -> {args.image_root}")
    print(f"splits: {manifest['split'].value_counts().to_dict()}")
    print(f"labels: {manifest['label'].value_counts().to_dict()}")
    print(f"quarantine drops: {write_stats.dropped_quarantine}")
    print(f"quarantine reviews kept: {write_stats.review_quarantine}")
    if write_stats.decode_errors:
        print(f"decode errors skipped: {write_stats.decode_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
