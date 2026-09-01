"""The data-access contract for local training and evaluation.

  load_manifest(split)  → the manifest rows for that split (pandas DataFrame)
  open_image(row)        → PIL.Image, RGB

Plus ``BranchADataset``, which turns rows into the clean/distorted tensor pairs
the robustness objective needs, applying the spatial policy used by Branch A:
RandomResizedCrop + hflip during training, squish-resize at inference.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from aigid.constants import IMAGENET_MEAN, IMAGENET_STD

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifest.parquet"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "full"          # loose files: <root>/<path>

# Where materialize_shards() unpacks the full build. Override with $AIGID_CACHE
# to put the ~10 GB somewhere other than the repo checkout.
FULL_CACHE = Path(os.environ.get("AIGID_CACHE", ROOT / "data" / "full_cache"))

_VALID_SPLITS = {"train", "val", "test"}

def load_manifest(split: str, manifest_path: str | Path | None = None) -> pd.DataFrame:
    """Rows for one split from a local parquet manifest."""
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    df = pd.read_parquet(path)
    rows = df[df["split"] == split].reset_index(drop=True)
    if len(rows) == 0:
        raise RuntimeError(
            f"0 rows for split={split!r} in {path}. The manifest must carry "
            f"real train/val/test values."
        )
    return rows


# ─────────────────────────────────────────────────── full-build resolution ───
# Some local builds can pack images into ~512 MB zstd parquet shards with image
# bytes inline, and each shard as a SINGLE row group (5-7k rows, ~537 MB
# decompressed). That shape is fine to stream sequentially, but it is hostile to
# the random access a shuffling DataLoader does: parquet cannot seek to one row
# inside a row group, so every __getitem__ that misses the cache decompresses
# the whole shard. With many shards and a shuffled sampler, that is hundreds of
# MB of zstd per image.
#
# So the supported path is to unpack once (materialize_shards) into loose files
# laid out exactly like the dev subset, after which open_image is the same cheap
# file read it always was. Reading straight from the parquet still works and is
# correct -- it keeps a one-shard LRU, so a shard-ordered pass is fast -- but it
# warns, because a shuffled epoch on that path is not something you want to
# discover from the wall clock.

# `<shard path>#<row_idx>`. The shard path may be namespaced by data source,
# such as `full/...` or `wildfake/full/...`.
_SHARD_KEY = re.compile(r"^((?:[^#/]+/)*full/[^#/]+\.parquet)#(\d+)$")
_shard_cache: tuple[str, object] | None = None   # (shard, pyarrow ChunkedArray)
_warned_slow = False


def _shard_file(shard: str, image_root: str | Path | None = None) -> Path:
    """Local path to a shard in a judge-controlled checkout or data mirror."""
    if image_root:
        local = Path(image_root) / shard
        if local.exists():
            return local
    local = FULL_CACHE / shard
    if local.exists():
        return local
    raise FileNotFoundError(
        f"missing shard {shard!r}. Public reproduction must provide local "
        "data/manifest.parquet and the referenced shards or materialized loose "
        "images; this repo does not download from any team-controlled artifact "
        "store.")


def _read_shard_row(shard: str, row_idx: int, image_root=None) -> bytes:
    global _shard_cache, _warned_slow
    import pyarrow.parquet as pq

    if not _warned_slow:
        warnings.warn(
            f"reading {shard} straight from parquet: each cache miss "
            f"decompresses the whole ~537 MB row group. Run "
            f"aigid.data.materialize_shards() once for a shuffled epoch.",
            stacklevel=3,
        )
        _warned_slow = True

    if _shard_cache is None or _shard_cache[0] != shard:
        col = pq.read_table(_shard_file(shard, image_root), columns=["image"])["image"]
        _shard_cache = (shard, col)          # replaces, so at most one shard is resident
    return _shard_cache[1][row_idx].as_py()


def materialize_shards(shards=None, image_root=None, dest: str | Path | None = None,
                       verify: bool = True) -> Path:
    """Unpack the full build into loose ``<dest>/full/<shard-stem>/<row_idx>.jpg``.

    One pass per shard, so this is the sequential access the layout is good at:
    ~10 GB and one 537 MB row group resident at a time. Idempotent -- a shard
    whose files are already all present is skipped, so an interrupted run
    resumes. Returns the root to hand back as ``image_root``.
    """
    import pyarrow.parquet as pq

    dest = Path(dest) if dest else FULL_CACHE
    man = pd.read_parquet(_manifest_for_shards(image_root))
    shards = list(shards) if shards else sorted(man["shard"].dropna().unique())

    for shard in shards:
        rows = man[man["shard"] == shard]
        # Mirror the shard's OWN path, namespace included. `Path(shard).stem`
        # Preserve the shard namespace. Flattening to Path(shard).stem can
        # collide when different data sources reuse names like shard-00000.
        out = dest / Path(shard).parent / Path(shard).stem
        if out.is_dir() and len(list(out.glob("*.jpg"))) == len(rows):
            continue
        out.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(_shard_file(shard, image_root),
                              columns=["image", "sha256"])
        images, digests = table["image"], table["sha256"]
        for i in range(len(images)):
            raw = images[i].as_py()
            if verify:
                got = hashlib.sha256(raw).hexdigest()
                if got != digests[i].as_py():
                    raise RuntimeError(
                        f"{shard}#{i}: sha256 mismatch (shard says "
                        f"{digests[i].as_py()[:12]}, bytes hash to {got[:12]}). "
                        f"The shard is corrupt -- re-download it."
                    )
            (out / f"{i}.jpg").write_bytes(raw)
        del table, images, digests
        print(f"materialized {shard} ({len(rows)} images)")
    return dest


def _manifest_for_shards(image_root=None) -> Path:
    """The full-build manifest from a local checkout or judge data mirror."""
    local = Path(image_root or FULL_CACHE) / "manifest.parquet"
    if local.exists():
        return local
    raise FileNotFoundError(
        f"no manifest.parquet found at {local}. Build or provide the public "
        "dataset manifest locally before calling materialize_shards().")


def open_image(row, image_root: str | Path | None = None) -> Image.Image:
    """Resolve a manifest row to pixels.

    Public reproduction uses loose files under ``data/full/<path>``. Local
    sharded builds can also use the logical key
    ``full/<shard>.parquet#<row_idx>``, served from the unpacked cache when
    materialize_shards() has run and from the parquet otherwise.
    """
    path = str(row["path"])
    m = _SHARD_KEY.match(path)
    if m:
        shard, row_idx = m.group(1), int(m.group(2))
        root = Path(image_root) if image_root else FULL_CACHE
        unpacked = root / Path(shard).parent / Path(shard).stem / f"{row_idx}.jpg"
        if unpacked.exists():
            img = Image.open(unpacked)
        else:
            img = Image.open(io.BytesIO(_read_shard_row(shard, row_idx, image_root)))
        img.load()
        return img.convert("RGB")

    full = (Path(image_root) if image_root else DEFAULT_IMAGE_ROOT) / path
    if not full.exists():
        raise FileNotFoundError(full)
    img = Image.open(full)
    img.load()
    return img.convert("RGB")


# ───────────────────────────────────────────────────────────── dataset ───────
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # data.py stays importable without torch (canon-style)
    Dataset = object


class BranchADataset(Dataset):
    """Rows → ``{"clean", "distorted", "label"}``.

    Spatial policy (§4.2):
      train:     RandomResizedCrop(scale=(0.6, 1.0)) + horizontal flip
      inference: squish-resize to (size, size) — full image, no crop

    The clean and distorted views share the *same* spatial crop, so the §5
    consistency terms (KL, feature-MSE) compare like with like. ``grid_cell``
    applies one fixed §6.1 evaluation cell instead of the training tiers — for
    M4's robustness sweep.
    """

    def __init__(self, split: str, size: int = 448, train: bool = True,
                 distort: bool = True, grid_cell: str | None = None,
                 manifest_path=None, image_root=None,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, seed: int = 0):
        if Dataset is object:
            raise ImportError("BranchADataset needs torch; pip install -r requirements.txt")
        self.rows = load_manifest(split, manifest_path)
        self.size = size
        self.train = train
        self.distort = distort
        self.grid_cell = grid_cell
        self.image_root = image_root
        self.seed = seed
        self.epoch = 0
        self._mean = torch.tensor(mean).view(3, 1, 1)
        self._std = torch.tensor(std).view(3, 1, 1)

    def set_epoch(self, epoch: int) -> None:
        """Vary augmentation across epochs while staying reproducible."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    # ── spatial policy ──────────────────────────────────────────────────────
    def _rrc(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        w, h = img.size
        area = w * h
        for _ in range(10):
            target = rng.uniform(0.6, 1.0) * area
            ar = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
            tw, th = int(round(np.sqrt(target * ar))), int(round(np.sqrt(target / ar)))
            if tw <= w and th <= h:
                x0, y0 = int(rng.integers(0, w - tw + 1)), int(rng.integers(0, h - th + 1))
                img = img.crop((x0, y0, x0 + tw, y0 + th))
                break
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _squish(self, img: Image.Image) -> Image.Image:
        return img.resize((self.size, self.size), Image.Resampling.BICUBIC)

    def _to_tensor(self, img: Image.Image):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return (t - self._mean) / self._std

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        img = open_image(row, self.image_root)

        if self.grid_cell is not None:
            from aigid.distort import apply_grid_cell
            view = self._squish(apply_grid_cell(img, self.grid_cell))
            clean = distorted = self._to_tensor(view)
        else:
            # Distort BEFORE the squish, at the crop's own resolution. That is
            # the order both the eval grid above and the deployed path take (a
            # transform lands on the delivered image, and predict.py resizes
            # after), so a JPEG block or a blur σ means the same thing at train
            # time as at test time. Squishing first would fix every artifact to
            # the 448² scale and quietly re-introduce the train/eval drift this
            # module exists to prevent.
            base = self._rrc(img, rng) if self.train else img
            clean = self._to_tensor(self._squish(base))
            if self.distort:
                from aigid.distort import sample_train_distortion
                d, _ = sample_train_distortion(base, rng)
                distorted = self._to_tensor(self._squish(d))
            else:
                distorted = clean

        return {"clean": clean, "distorted": distorted,
                "label": torch.tensor(float(row["label"]))}
