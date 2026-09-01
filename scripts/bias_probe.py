"""Acceptance gate: can a model separate real from fake using *only* file-level
statistics -- no pixel semantics, no generator artifacts?

If it can, the dataset is teaching a compression/resolution shortcut and any
downstream AUROC is a mirage. The acceptance gate is <= 0.55 AUROC.

    python scripts/bias_probe.py data/manifest.parquet --root data/full

--root is the local directory the manifest paths are relative to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GATE = 0.55

FEATURES = [
    "file_size", "width", "height", "pixels", "aspect_ratio", "short_edge",
    "bytes_per_pixel", "jpeg_quality", "n_quant_tables", "q_mean", "q_std",
    "dct_energy", "dct_hf_ratio", "dct_kurtosis",
]

# Geometry + compression history: what canonicalization is responsible for.
ACQUISITION = ["width", "height", "pixels", "aspect_ratio", "short_edge",
               "jpeg_quality", "n_quant_tables", "q_mean", "q_std"]
# Image complexity: part genuine forensic cue, part real-source confound.
CONTENT = ["file_size", "bytes_per_pixel", "dct_energy", "dct_hf_ratio",
           "dct_kurtosis"]


def extract(path: Path) -> dict | None:
    try:
        raw = path.read_bytes()
        img = Image.open(path)
        w, h = img.size
        # Quantisation tables are the compression fingerprint the probe hunts for.
        qt = getattr(img, "quantization", {}) or {}
        qvals = np.concatenate([np.asarray(v, dtype=float).ravel()
                                for v in qt.values()]) if qt else np.array([0.0])
        g = np.asarray(img.convert("L"), dtype=float)
        # 8x8 block DCT statistics over a centre crop, cheap and resolution-free.
        s = min(256, g.shape[0] - g.shape[0] % 8, g.shape[1] - g.shape[1] % 8)
        c = g[:s, :s] if s >= 8 else g[:8, :8]
        d = dct(dct(c.T, norm="ortho").T, norm="ortho")
        a = np.abs(d)
        hf = a[a.shape[0] // 2:, a.shape[1] // 2:]
        return {
            "file_size": len(raw), "width": w, "height": h, "pixels": w * h,
            "aspect_ratio": w / h, "short_edge": min(w, h),
            "bytes_per_pixel": len(raw) / (w * h),
            "jpeg_quality": float(np.mean(qvals)),
            "n_quant_tables": float(len(qt)),
            "q_mean": float(qvals.mean()), "q_std": float(qvals.std()),
            "dct_energy": float(a.mean()),
            "dct_hf_ratio": float(hf.mean() / (a.mean() + 1e-9)),
            "dct_kurtosis": float(((a - a.mean()) ** 4).mean() / (a.std() ** 4 + 1e-9)),
        }
    except Exception as e:
        print(f"  skip {path.name}: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--root", default="data/full",
                    help="dir the manifest paths are relative to")
    ap.add_argument("--gate", choices=("pooled", "acquisition"), default="pooled",
                    help="'pooled' gates the full file-statistics probe as the acceptance "
                         "criterion specifies; 'acquisition' gates only what "
                         "canonicalization controls (an explicit override)")
    ap.add_argument("--content-shortcut", action="store_true",
                    help="run the content-shortcut probe instead: restrict to "
                         "fakes whose real_source includes COCO, scored against COCO "
                         "reals, so content is held constant")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    if args.content_shortcut:
        if "real_source" not in df.columns:
            print("manifest has no `real_source` column -- the content-held-constant "
                  "slice cannot be identified. Rebuild with the current schema.")
            return 2
        rs = df["real_source"].fillna("").str.lower()
        keep = ((df.label == 1) & rs.str.contains("coco")) | \
               ((df.label == 0) & df["source"].str.contains("COCO", case=False, na=False))
        df = df[keep]
        n1, n0 = int((df.label == 1).sum()), int((df.label == 0).sum())
        print(f"content-shortcut slice: {n1} COCO-sourced fakes vs {n0} COCO reals")
        if min(n1, n0) < 50:
            print("slice too small to be meaningful (need >= 50 per class).")
            return 2
    root = Path(args.root)
    rows, labels = [], []
    for _, r in df.iterrows():
        # Manifest paths are repo-relative ("dev/images/..."); strip the prefix.
        rel = r["path"].split("/", 1)[1] if r["path"].startswith("dev/") else r["path"]
        f = extract(root / rel)
        if f:
            rows.append(f)
            labels.append(int(r["label"]))
    X = pd.DataFrame(rows)[FEATURES].to_numpy()
    y = np.asarray(labels)
    print(f"probed {len(y)} images ({y.sum()} fake / {len(y) - y.sum()} real)")

    def auroc(cols: list[str]) -> float:
        idx = [FEATURES.index(c) for c in cols]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        p = cross_val_predict(clf, X[:, idx], y, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(y, p)

    # Per-feature AUROC names the culprit when the gate fails.
    print("\nsingle-feature AUROC (0.5 = no signal):")
    for i, name in enumerate(FEATURES):
        a = roc_auc_score(y, X[:, i])
        print(f"  {name:16s} {max(a, 1 - a):.3f}")

    acq = auroc(ACQUISITION)
    content = auroc(CONTENT)
    pooled = auroc(FEATURES)
    print(f"\nACQUISITION AUROC {acq:.4f}   gate <= {GATE}")
    print(f"CONTENT     AUROC {content:.4f}   diagnostic")
    print(f"POOLED      AUROC {pooled:.4f}   <-- pooled acceptance gate")

    # Only the acquisition block gates. It measures what canonicalization
    # controls -- geometry and compression history -- so a failure there is a
    # pipeline bug we can fix. The content block (file size at fixed
    # resolution/quality, DCT energy) measures image complexity, which is
    # partly the genuine forensic cue (diffusion under-produces high
    # frequencies) and partly a real-source confound. Canonicalization cannot
    # fix it and should not try; real-source diversity and held-out-generator
    # evaluation are the controls for that.
    # Two verdicts, both stated. The acceptance criterion gates the
    # POOLED probe -- "(file size, quality estimate, resolution, aspect ratio,
    # DCT stats) only must score <= 0.55". Reporting only the acquisition block
    # would announce PASS while the pooled criterion is failing, which is
    # how a dataset ships with a shortcut nobody agreed to accept.
    pooled_ok, acq_ok = pooled <= GATE, acq <= GATE
    print(f"\n  pooled gate                {'PASS' if pooled_ok else 'FAIL'}"
          f"   {pooled:.4f} vs {GATE}")
    print(f"  acquisition (canonicalization) {'PASS' if acq_ok else 'FAIL'}"
          f"   {acq:.4f} vs {GATE}")

    if not acq_ok:
        print("\nFAIL -- canonicalization left an acquisition shortcut: geometry or "
              "compression history separates the classes. That is a pipeline bug. "
              "Do not train on it.")
        return 1
    if not pooled_ok:
        print(f"\nFAIL (pooled gate) -- acquisition is clean at {acq:.3f}, so "
              f"canonicalization is doing its job, but pixel statistics carry "
              f"{content:.3f} and the pooled gate catches it.")
        print("  Before overriding, run the content-shortcut probe (--content-shortcut):")
        print("  it holds content constant, so a low number there means this is the "
              "real-source mismatch that content pairing addresses, and a high number "
              "means a genuine shortcut.")
        print("  --gate acquisition overrides, deliberately and on the record.")
        return 0 if args.gate == "acquisition" else 1
    print("\nPASS -- both the pooled gate and the acquisition block are clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
