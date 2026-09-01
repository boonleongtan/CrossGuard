#!/usr/bin/env python
"""Build `manifest/quarantine.npz` - the forbidden-content index.

    python scripts/build_quarantine_index.py                 # build + write
    python scripts/build_quarantine_index.py --verify-only   # check, write nothing
    python scripts/build_quarantine_index.py --out /tmp/q.npz

The challenge rules name a validation set and say not to use the following data
during training: COCO val2017 (4,998) and DALL-E Advanced (8,843). Both are excluded
twice over -- by path (exact, primary) and by content (this index, defence in
depth for the one case a path rule cannot see: those images reappearing inside
another corpus under different filenames).

**This index covers COCO val2017 only, and that is deliberate.** DALL-E Advanced
is excluded structurally: `plan_zip` admits only the `DALLE/Typical/` prefix of
`DALLE.zip`, so Advanced members are never selected, and `FORBIDDEN_PATH_SUBSTRINGS`
is a second net behind that. Those images exist in exactly one place in the
pipeline, so there is no "same image under another name" case for a content hash
to catch -- hashing them would guard a scenario that cannot occur. COCO val2017
is different: it is a widely redistributed public corpus that other datasets
embed, which is precisely why it needs content screening.

Source: WildFake's `Images/Real/coco.zip` on ModelScope, read over HTTP ranges
(`aigid/rangezip`) rather than downloaded -- the archive is 2.35 GB and we need
4,998 of its 163,846 members. Verified 29 Aug: members are named
`coco/coco2017/val2017/imgNNNNNN.jpg`, exactly 4,998 of them, and all 4,998 are
caught by `is_forbidden_path()`.

**Why the index holds fewer entries than 4,998.** Hashes are deduplicated: two
byte-identical or perceptually identical val2017 images produce the same
(phash, dhash) pair and collapse to one entry. That loses nothing -- the screen
asks "is this image within Hamming 5 of ANY forbidden hash", so a duplicate
entry can never change an answer. The count is reported at build time so the
collapse is visible rather than mysterious.
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigid.quarantine import Screen, hashes_of          # noqa: E402
from aigid.rangezip import open_remote_zip              # noqa: E402

REPO = "hy2628982280/WildFake"
ARCHIVE = "Images/Real/coco.zip"
MARKER = "val2017"
# The rules' own count for the COCO val2017 slice. A mismatch here means the
# upstream archive changed, and the build should stop rather than quietly index
# a different set than the rules name.
EXPECTED_MEMBERS = 4998

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "manifest" / "quarantine.npz"


def _member_bytes(url: str, info, session) -> bytes:
    """One member's decompressed bytes, fetched as a single explicit byte range.

    zipfile.read() would be the obvious call, and it is what this used to do --
    but ModelScope rate-limits a single connection hard: a sequential pass over
    4,998 members settled at ~2 KB/s and would have taken hours. Fetching each
    member as its own ranged GET lets the pool below run several in flight,
    which finishes in ~15 min.

    That means parsing the local file header by hand, so the layout is spelled
    out: 30 fixed bytes, then the filename (length at offset 26), then the extra
    field (length at offset 28), then `compress_size` bytes of payload. The
    central directory already told us compress_size and compress_type, so
    nothing here has to guess. Verified byte-identical to zipfile.read() across
    20 members spanning the archive before this replaced it.
    """
    start = info.header_offset
    # +256 covers the local header's filename and extra field, which may differ
    # in length from the central directory's copy of them.
    end = start + 30 + len(info.filename.encode()) + 256 + info.compress_size
    r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
    r.raise_for_status()
    b = r.content
    off = 30 + int.from_bytes(b[26:28], "little") + int.from_bytes(b[28:30], "little")
    data = b[off:off + info.compress_size]
    if info.compress_type != 0:                 # 0 = stored, 8 = deflate
        data = zlib.decompress(data, -15)       # -15: raw stream, no zlib header
    return data


def collect_hashes(workers: int = 16, progress_every: int = 500):
    """(phashes, dhashes, failures) for every val2017 member of the archive."""
    import requests

    z, rf = open_remote_zip(REPO, ARCHIVE)
    members = sorted(n for n in z.namelist()
                     if MARKER in n and not n.endswith("/"))
    print(f"{ARCHIVE}: {rf.size / 1e9:.2f} GB, "
          f"{len(members)} members matching {MARKER!r}")
    if len(members) != EXPECTED_MEMBERS:
        raise SystemExit(
            f"expected {EXPECTED_MEMBERS} {MARKER} members from the challenge rules, found "
            f"{len(members)}. The upstream archive has changed -- do not index "
            f"a different set than the rules name without settling why.")

    infos = {n: z.getinfo(n) for n in members}
    url = rf.url
    local = threading.local()

    def session():
        if not hasattr(local, "s"):
            local.s = requests.Session()        # one per thread; not shareable
        return local.s

    def one(name):
        try:
            with Image.open(io.BytesIO(_member_bytes(url, infos[name], session()))) as im:
                im.load()
                return name, hashes_of(im)
        except Exception as exc:                          # noqa: BLE001
            # An image we cannot hash is an image we cannot screen for. Collect
            # it and fail loudly at the end rather than silently shipping a
            # short index.
            return name, f"{type(exc).__name__}: {exc}"

    ph, dh, failures = [], [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (name, res) in enumerate(pool.map(one, members), 1):
            if isinstance(res, tuple):
                ph.append(res[0])
                dh.append(res[1])
            else:
                failures.append({"member": name, "error": res})
            if i % progress_every == 0:
                print(f"  hashed {i}/{len(members)}  ({time.time() - t0:.0f}s)",
                      flush=True)
    return ph, dh, failures


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"where to write the index (default {DEFAULT_OUT})")
    p.add_argument("--verify-only", action="store_true",
                   help="rebuild and compare against the committed index, "
                        "writing nothing")
    p.add_argument("--workers", type=int, default=16,
                   help="parallel range fetches (default 16; ModelScope "
                        "rate-limits a single connection to ~2 KB/s)")
    p.add_argument("--verify-sample", type=int, default=50,
                   help="how many random forbidden images to round-trip through "
                        "the finished index (default 50; every one must 'drop')")
    p.add_argument("--allow-failures", type=int, default=0,
                   help="tolerate up to N unhashable members (default 0: any "
                        "failure is fatal, since an unhashed image is unscreened)")
    args = p.parse_args()

    ph, dh, failures = collect_hashes(workers=args.workers)
    print(f"\nhashed {len(ph)}, failed {len(failures)}")
    if failures:
        for f in failures[:10]:
            print(f"  {f['member']}: {f['error']}")
        if len(failures) > args.allow_failures:
            raise SystemExit(
                f"{len(failures)} member(s) could not be hashed and would be "
                f"unscreened. Fix the source or pass --allow-failures.")

    # Deduplicate on the PAIR: two images collapse only if they agree on both
    # hashes, which is the same condition under which a duplicate entry could
    # never change a verdict.
    pairs = sorted(set(zip(ph, dh)))
    up = np.array([a for a, _ in pairs], dtype=np.uint64)
    ud = np.array([b for _, b in pairs], dtype=np.uint64)
    print(f"unique (phash, dhash) pairs: {len(pairs)}  "
          f"(collapsed {len(ph) - len(pairs)} duplicate image(s))")

    committed = Path(DEFAULT_OUT)
    if committed.exists():
        z = np.load(committed)
        old = set(zip(z["phash"].tolist(), z["dhash"].tolist()))
        new = set(pairs)
        print(f"\ncommitted index: {len(old)} entries")
        print(f"  in both      : {len(old & new)}")
        print(f"  only in new  : {len(new - old)}")
        print(f"  only in old  : {len(old - new)}")
        if old == new:
            print("  -> IDENTICAL: the committed index is reproducible from source.")
        else:
            print("  -> DIFFERS. Inspect before overwriting.")

    if args.verify_only:
        print("\n--verify-only: nothing written")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Screen(phashes=up, dhashes=ud).save(str(out))
    print(f"\nwrote {len(pairs)} entries -> {out}")

    # ---- Verify the ARTIFACT, not the process. This is the check whose absence
    # let a wrong index sit in the repo: the committed index held 4,952 hashes
    # of *something*, but screening WildFake's own val2017 images against it
    # returned 'ok' at Hamming 12-19 -- it had been built from a different COCO
    # source, and re-encoding had moved every hash past the threshold. Nothing
    # caught that, because nothing ever asked "does this index drop the images
    # it exists to drop?". Now something does, on a random sample rather than
    # one member, because one member cannot distinguish a good index from an
    # index that happens to contain that one image.
    z2, _ = open_remote_zip(REPO, ARCHIVE)
    members = sorted(n for n in z2.namelist()
                     if MARKER in n and not n.endswith("/"))
    sample = random.Random(0).sample(members, min(args.verify_sample, len(members)))
    screen = Screen(str(out))
    verdicts = {"drop": 0, "review": 0, "ok": 0}
    worst = []
    for name in sample:
        action, pmin, dmin = screen.verdict(z2.read(name))
        verdicts[action] += 1
        if action != "drop":
            worst.append((name, action, pmin, dmin))
    print(f"\nround-trip verification on {len(sample)} random members:")
    print(f"  drop={verdicts['drop']}  review={verdicts['review']}  ok={verdicts['ok']}")
    if verdicts["drop"] != len(sample):
        for name, action, pmin, dmin in worst[:10]:
            print(f"  NOT DROPPED: {name} -> {action} (phash d={pmin}, dhash d={dmin})")
        raise SystemExit(
            f"{len(sample) - verdicts['drop']} of {len(sample)} known-forbidden "
            f"images did not screen as 'drop'. The index does not match the data "
            f"it must screen -- do NOT ship it.")
    print("  PASS: every sampled forbidden image screens as 'drop'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
