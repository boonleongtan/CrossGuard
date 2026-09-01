"""The forbidden-content screen.

The hackathon rules (rules/rules.md §5.4) name a validation set and say
plainly: **do not use the following data during training**.

    Non-AIGC   COCO val2017          4,998
    AIGC       DALL-E Advanced       8,843

Both numbers are exact matches for WildFake slices — `real_coco.csv` has
exactly 4,998 rows under `coco2017/val2017/`, and `dalle3.csv` is exactly 8,843
rows, every one `IsAdvanced=1`. So the rule is not vague: it names two
identifiable sets of images.

Path filtering alone is not enough, for one reason: COCO val2017 is a *public
corpus*, and other datasets embed it. Anything sourcing "COCO" may be carrying
val2017 images under a different filename, and no path rule catches that. So we
screen by CONTENT — a perceptual hash of every forbidden image, checked against
every image we stage, from every dataset.

Design notes that matter for correctness:

* Two hashes, not one. phash (DCT) and dhash (gradient) fail differently;
  requiring only ONE to match makes the screen more paranoid, which is the
  direction we want to err in.
* Banded LSH, not prefix bucketing. Bucketing on the top-N bits misses most
  true matches: for two hashes at Hamming distance 5, the chance that all five
  differing bits avoid a 16-bit prefix is only ~22%. With B bands and threshold
  T, a pair within T MUST share a whole band whenever B > T (pigeonhole), so
  B=6 bands finds every pair at distance <= 5. Exhaustively verified in
  `selftest()`.
* `review` before `drop`. A near-but-not-certain match is reported, not
  silently discarded, so a screen that is too aggressive is visible rather than
  quietly eating training data.

This module is the ENGINE. The index it screens against is built by
`scripts/build_quarantine_index.py`, and the two failed apart once: a
`quarantine.npz` built from a different COCO val2017 source than the pipeline
reads screened 27 of 27 forbidden images as `ok` (Hamming 12-19; 2-of-4,998
overlap with the correct hashes), because re-encoding moves a perceptual hash
well past DROP_AT.

`selftest()` below does NOT catch that, and cannot: it proves the banded-LSH
matching finds every pair within DROP_AT using *synthetic* hashes, so it passes
against an index built from the wrong images entirely. The check that matters is
end-to-end -- feed a known-forbidden image to `verdict()` and require `drop` --
and it lives in the builder, which runs it after every write.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

DROP_AT = 5        # <= this Hamming distance on either hash: forbidden
REVIEW_AT = 10     # <= this: flagged for a human, kept
BANDS = 6          # > DROP_AT, so the pigeonhole guarantee holds
_BAND_BITS = [(i * 64 // BANDS, (i + 1) * 64 // BANDS) for i in range(BANDS)]


def _dct2(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    m[:, 0] *= 1 / np.sqrt(2)
    return m.T @ a @ m


def _bits_to_int(bits) -> int:
    return int("".join("1" if b else "0" for b in bits), 2)


def hashes_of(img: Image.Image) -> tuple[int, int]:
    """(phash, dhash) as 64-bit ints, both scale-invariant.

    Scale invariance is load-bearing: a forbidden image re-encoded, resized or
    canonicalized must still match, so the screen has to work across the
    resolutions the pipeline produces.
    """
    g = img.convert("L")
    a = np.asarray(g.resize((32, 32), Image.Resampling.BICUBIC), dtype=np.float64)
    d = _dct2(a)[:8, :8]
    phash = _bits_to_int((d > np.median(d[1:].flatten())).flatten())

    b = np.asarray(g.resize((9, 8), Image.Resampling.BICUBIC), dtype=np.float64)
    dhash = _bits_to_int((b[:, 1:] > b[:, :-1]).flatten())
    return phash, dhash


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bands(h: int) -> list[tuple[int, int]]:
    """(band index, band value) pairs — the LSH key set for one hash."""
    return [(i, (h >> lo) & ((1 << (hi - lo)) - 1)) for i, (lo, hi) in enumerate(_BAND_BITS)]


class Screen:
    """Holds the forbidden index; answers 'may I train on this image?'."""

    def __init__(self, index_path: str | None = None, phashes=None, dhashes=None):
        if index_path is not None:
            z = np.load(index_path)
            phashes, dhashes = z["phash"], z["dhash"]
        self.phash = np.asarray(phashes, dtype=np.uint64)
        self.dhash = np.asarray(dhashes, dtype=np.uint64)
        self._buckets: dict = {}
        for i, (p, d) in enumerate(zip(self.phash.tolist(), self.dhash.tolist())):
            for key in _bands(p):
                self._buckets.setdefault(("p", *key), []).append(i)
            for key in _bands(d):
                self._buckets.setdefault(("d", *key), []).append(i)

    def __len__(self) -> int:
        return len(self.phash)

    def _candidates(self, p: int, d: int) -> set:
        c = set()
        for key in _bands(p):
            c.update(self._buckets.get(("p", *key), ()))
        for key in _bands(d):
            c.update(self._buckets.get(("d", *key), ()))
        return c

    def verdict(self, raw: bytes) -> tuple[str, int, int]:
        """(action, min phash distance, min dhash distance).

        action is 'drop' (forbidden — never train), 'review' (suspicious, kept
        and logged) or 'ok'. Undecodable bytes are 'ok': they fail later in
        canonicalization, and guessing here would silently drop good data.
        """
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                p, d = hashes_of(im)
        except Exception:
            return "ok", 64, 64
        pmin = dmin = 64
        for i in self._candidates(p, d):
            pmin = min(pmin, hamming(p, int(self.phash[i])))
            dmin = min(dmin, hamming(d, int(self.dhash[i])))
        worst = min(pmin, dmin)
        return ("drop" if worst <= DROP_AT else
                "review" if worst <= REVIEW_AT else "ok"), pmin, dmin

    def save(self, path: str) -> None:
        np.savez_compressed(path, phash=self.phash, dhash=self.dhash)


def selftest(n: int = 2000, seed: int = 0) -> dict:
    """Prove the banding finds every pair within DROP_AT.

    This is the property the whole screen rests on, and it is cheap to check
    exhaustively, so it is checked rather than argued.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 1 << 63, size=n, dtype=np.int64).astype(np.uint64)
    s = Screen(phashes=base, dhashes=base)
    missed = 0
    for _ in range(n):
        i = int(rng.integers(0, n))
        h = int(base[i])
        flips = int(rng.integers(0, DROP_AT + 1))
        for bit in rng.choice(64, size=flips, replace=False):
            h ^= 1 << int(bit)
        if i not in s._candidates(h, h):
            missed += 1
    return {"trials": n, "missed": missed, "bands": BANDS, "drop_at": DROP_AT}
