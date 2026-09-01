#!/usr/bin/env python
"""What is in the build, read from the registry itself.

    python scripts/show_datasets.py            # summary
    python scripts/show_datasets.py --full     # + per-source detail and notes

The registry in aigid/canon.py is the single source of truth for what gets
ingested, so this prints that rather than a maintained list which could drift
out of date the first time someone adds a dataset and forgets the docs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigid.canon import DATASETS, GEOMETRY_POOLS, assign_split  # noqa: E402

FULL = "--full" in sys.argv
DASH = "-"


def sources_of(spec):
    """Every Source in a dataset, whatever its kind."""
    if spec.kind == "zip":
        return [s for group in (spec.archives or {}).values() for s in group.values()]
    return list(spec.sources.values())


def split_of(name, spec, src):
    """What the split rule does with this source, stated the way it behaves."""
    if spec.holdout_split:
        return f"upstream train->train, {spec.holdout_split}->val/test"
    if src.label == 0:
        return "80/10/10 by content hash"
    return assign_split(name, "", "ab" * 32, src.label, src.generator) + " (whole generator)"


def main() -> None:
    rows = []
    for name, spec in DATASETS.items():
        for src in sources_of(spec):
            rows.append((name, spec, src))

    gens = sorted({s.generator for _, _, s in rows if s.generator})
    archs = sorted({s.architecture for _, _, s in rows if s.architecture})
    reals = sorted({s.key for _, _, s in rows if s.label == 0})

    print(f"\n{len(DATASETS)} datasets, {len(gens)} generators, "
          f"{len(archs)} architectures, {len(reals)} real sources\n")

    hdr = f"{'dataset':14s} {'':1s} {'source':26s} {'architecture':16s} {'content':8s} split"
    print(hdr)
    print(DASH * len(hdr))
    for name, spec in DATASETS.items():
        for src in sorted(sources_of(spec), key=lambda s: (s.label, s.key)):
            mark = "R" if src.label == 0 else "F"
            print(f"{name:14s} {mark:1s} {src.key:26s} "
                  f"{src.architecture or '-':16s} {src.content:8s} "
                  f"{split_of(name, spec, src)}")
        if FULL and spec.note:
            print(f"{'':16s}note: {spec.note}")
    print()

    print("GENERATORS")
    for g in gens:
        print(f"  {g}")

    all_held = set()
    all_trained = set()
    for sp in DATASETS.values():
        all_held |= sp.held_out_generators
        all_trained |= sp.trained_generators

    print("\nHELD-OUT GENERATORS (never in train - the unseen-generator axis)")
    for g in sorted(all_held):
        print(f"  {g}")
    bucketed = {s.generator for _, sp, s in rows if s.generator and not sp.holdout_split}
    unpinned = bucketed - all_held - all_trained
    if unpinned:
        print("\n  WARNING - generators decided by hash, not by an explicit list:")
        for g in sorted(unpinned):
            print(f"    {g}")

    print("\nGEOMETRY POOLS")
    for pool, why in GEOMETRY_POOLS.items():
        members = sorted({n for n, sp in DATASETS.items() if sp.pool == pool})
        print(f"  {pool:10s} {', '.join(members) or '(none)'}")
        if FULL:
            print(f"{'':13s}{why}")

    print("\nEXCLUDED FROM TRAINING (challenge rules validation set)")
    print("  COCO val2017      4998   by path + perceptual index")
    print("  DALL-E Advanced   8843   by path; also not hosted on ModelScope")
    idx = Path(__file__).resolve().parents[1] / "manifest" / "quarantine.npz"
    print(f"  quarantine index: {'present' if idx.exists() else 'MISSING - build it'} "
          f"({idx})")
    print()


if __name__ == "__main__":
    main()
