"""Canonicalization and the split rule — the dataset-building contract.

Rebuilt from scratch (28 Aug) after the previous pipeline was deleted. Scope is
deliberately one dataset: SID_Set. Everything dataset-specific lives in the
``DATASETS`` registry at the bottom, so the next source is a new entry rather
than a new code path.

Two jobs, and they are the only two:

  canonicalize()  — make a real and a fake indistinguishable by anything except
                    the generation itself: same geometry distribution, same
                    compression history. Without this the model learns
                    "PNG = fake, JPEG = real" (arXiv 2403.17608) and inverts
                    under any JPEG-augmented benchmark.
  assign_split()  — decide train/val/test ONCE. Nobody calls train_test_split.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

# The manifest schema. aigid/data.py reads `path`, `shard`, `label`, `split`
# and `sha256`; the rest exist so a slice can be named after the fact (per
# generator, per source, per upstream split) without rebuilding.
MANIFEST_COLUMNS = [
    "path", "shard", "row_idx", "label", "source", "generator",
    "architecture", "generator_licence", "content", "real_source",
    "upstream_split", "split", "sha256", "phash",
]

# One shared quality distribution for BOTH classes. A per-class quality range
# is itself a shortcut.
JPEG_QUALITY_RANGE = (75, 96)


# ─────────────────────────────────────────────────────────── hashing ────────

def hashes(raw: bytes) -> tuple[str, int]:
    """(sha256 hex, 64-bit perceptual hash) for one encoded image.

    phash is returned as an int so dedup is a popcount, not a string compare.
    """
    sha = hashlib.sha256(raw).hexdigest()
    img = Image.open(io.BytesIO(raw)).convert("L").resize((32, 32), Image.Resampling.BICUBIC)
    # DCT-II via numpy: no scipy dependency in the core build path.
    a = np.asarray(img, dtype=np.float64)
    d = _dct2(a)[:8, :8]
    med = np.median(d[1:].flatten() if d.size > 1 else d)
    bits = (d > med).flatten()
    return sha, int("".join("1" if b else "0" for b in bits), 2)


def _dct2(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    m[:, 0] *= 1 / np.sqrt(2)
    return m.T @ a @ m


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ────────────────────────────────────────────────────── canonicalization ────

def read_image(raw: bytes) -> tuple[Image.Image, bool]:
    """Decode to RGB. Returns (image, arrived_as_jpeg)."""
    img = Image.open(io.BytesIO(raw))
    fmt = (img.format or "").upper()
    img.load()
    return img.convert("RGB"), fmt in ("JPEG", "JPG")


def canonicalize(img: Image.Image, arrived_jpeg: bool, target_ar: float,
                 target_se: int, rng: np.random.Generator) -> bytes:
    """One image → canonical JPEG bytes.

    Geometry: centre-crop to ``target_ar``, then resize so the short edge is
    ``target_se``. Both targets are drawn by the caller from the POOLED
    real+fake distribution, so neither class carries its native geometry.

    Compression: every image leaves as JPEG at a quality drawn from one shared
    distribution, and every image has been JPEG-encoded the same number of
    times — one generation for arrivals that were already JPEG, two for
    lossless arrivals, which is what equalises the history rather than the
    format. Counting generations, not formats, is the whole point: a PNG
    re-encoded once still has one fewer generation than a JPEG re-encoded once.
    """
    w, h = img.size
    cur = w / h
    if cur > target_ar:                      # too wide → trim width
        nw = max(1, int(round(h * target_ar)))
        x0 = (w - nw) // 2
        img = img.crop((x0, 0, x0 + nw, h))
    else:                                    # too tall → trim height
        nh = max(1, int(round(w / target_ar)))
        y0 = (h - nh) // 2
        img = img.crop((0, y0, w, y0 + nh))

    w, h = img.size
    scale = target_se / min(w, h)
    img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                     Image.Resampling.BICUBIC)

    lo, hi = JPEG_QUALITY_RANGE
    generations = 1 if arrived_jpeg else 2
    for _ in range(generations):
        q = int(rng.integers(lo, hi + 1))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        raw = buf.getvalue()
        img = Image.open(io.BytesIO(raw)); img.load(); img = img.convert("RGB")
    return raw


def pooled_targets(sizes: list[tuple[int, int]]) -> tuple[list[float], list[int]]:
    """Aspect ratios and short edges observed across BOTH classes together.

    Returned as the empirical pools to sample from. Pooling is what makes the
    geometry uninformative: sampling each class from its own distribution would
    preserve exactly the cue we are trying to destroy.
    """
    ars = [w / h for w, h in sizes if w > 0 and h > 0]
    ses = [min(w, h) for w, h in sizes if w > 0 and h > 0]
    return ars, ses


# ──────────────────────────────────────────────────────────── the split ─────

_HOLDOUT_SALT = "crossguard/valtest/v1"


def assign_split(dataset: str, upstream_split: str, sha256_hex: str,
                 label: int = 0, generator: str = "") -> str:
    """train/val/test, honouring each upstream dataset's own boundary.

    Every source we ingest ships a split and withholds something. We respect
    that rather than pooling and re-splitting, because their held-out material
    is the only part guaranteed not to have leaked into upstream training:

      upstream train    → our train
      upstream holdout  → our val (half) and our test (half)

    Halving the holdout rather than mapping it wholly to val is deliberate. A
    source that contributes no test rows drops out of the test set entirely,
    which silently changes what the test set is made of — for SID_Set that
    would have made our test reals 100% COCO, a train/test content shift in the
    reals that we would later have misread as a false-positive result. Both
    halves are still images upstream never trained on, so the contamination
    guarantee is untouched; only our own val/test line is drawn by us.
    """
    spec = DATASETS[dataset]

    # WildFake ships no split of its own, so we draw one -- and this is the
    # first dataset where the interesting rule is possible. With six generators
    # we can split FAKES BY GENERATOR, so a test generator is one the model has
    # never seen, which is the only number that predicts behaviour on a
    # generator that did not exist at training time.
    #
    # Deny-list first, bucketing second. Hash bucketing alone cannot honour an
    # intent it cannot see: it would scatter the generators we chose as the
    # axis across train, and the axis would silently be empty while still being
    # reported. So the held-out set is pinned, and only the remainder is
    # bucketed.
    if not spec.holdout_split:
        if label == 1:
            if generator in spec.held_out_generators:
                return "test"
            if generator in spec.trained_generators:
                b = int(sha256_hex, 16) % 20
                return "train" if b < 18 else "val"
            b = int(hashlib.sha256(generator.encode()).hexdigest(), 16) % 20
            return "train" if b < 16 else ("val" if b < 18 else "test")
        b = int(sha256_hex, 16) % 10          # reals: by content hash, 80/10/10
        return "train" if b < 8 else ("val" if b < 9 else "test")

    if upstream_split == "train":
        return "train"
    if upstream_split != spec.holdout_split:
        raise ValueError(
            f"{dataset}: unknown upstream split {upstream_split!r} "
            f"(expected 'train' or {spec.holdout_split!r})")
    h = hashlib.sha256((_HOLDOUT_SALT + sha256_hex).encode()).hexdigest()
    return "val" if int(h, 16) % 2 == 0 else "test"


# ───────────────────────────────────────────────────────── the registry ─────

@dataclass(frozen=True)
class Source:
    """One upstream label of one dataset, as one homogeneous pool of rows."""
    key: str
    label: int                  # OUR label: 0 real, 1 fake. NOT the upstream
                                # value — see CIFAKE below, which inverts them.
    generator: str              # "" for reals
    architecture: str
    licence: str
    content: str                # what the pixels depict
    real_source: str            # for fakes: where the prompts came from
    note: str = ""


@dataclass(frozen=True)
class Dataset:
    repo: str
    sources: dict               # UPSTREAM label value -> Source
    holdout_split: str          # the upstream split that is not "train"
    columns: list
    pool: str                   # geometry pool this dataset canonicalizes within
    axis: str = ""              # non-empty => reportable as its own eval axis
    kind: str = "parquet"       # "parquet" (HF shards) or "zip" (remote ranges)
    archives: dict = None       # zip only: archive path -> {member prefix: Source}
    held_out_generators: frozenset = frozenset()   # fakes split to test
    trained_generators: frozenset = frozenset()     # fakes pinned to train
    note: str = ""


# ── SID_Set ──────────────────────────────────────────────────────────────────
# saberzl/SID_Set, CC BY 4.0. Facts from the SIDA paper (arXiv 2412.04292 §4),
# NOT the dataset card, which documents none of them:
#   label 0  100K real, OpenImages V7
#   label 1  100K synthetic, FLUX, prompted from Flickr30k + COCO captions
#   label 2  100K tampered, latent-diffusion inpainting — never ingested
# The paper cites FLUX only as "Flux model team, Flux model" and never names
# the variant, so we record what is documented and leave the variant open.
#
# MEASURED 28 Aug: after canonicalization a 4x4 grey thumbnail still separates
# these classes at AUC 0.885. SID reals vs COCO reals is 0.584, so that is not
# a corpus mismatch — FLUX output is globally distinguishable from photographs.
# It is generator-specific aesthetic signal and will not transfer. Safe to
# train on; inflates any val/test slice it lands in.
SID_REAL = Source(
    key="SID_Set/OpenImagesV7", label=0, generator="", architecture="",
    licence="CC-BY-4.0", content="scene", real_source="",
    note="OpenImages V7; img_id matches upstream, so this and any direct "
         "OpenImages pull are ONE source and must not be double-counted.",
)
SID_FAKE = Source(
    key="SID_Set/FLUX", label=1, generator="SID_Set/FLUX",
    architecture="RectifiedFlow", licence="CC-BY-4.0",
    content="scene", real_source="Flickr30k+COCO",
    note="FLUX variant undocumented upstream. Carries a strong content "
         "shortcut (AUC 0.885 from a 4x4 thumbnail) that real-source pairing "
         "does not fix.",
)

# ── CIFAKE ───────────────────────────────────────────────────────────────────
# dragonintelligence/CIFAKE-image-dataset (Kaggle: birdy654/cifake-...).
# 120,000 images, upstream 100k train / 20k test, exactly balanced.
# Reals are CIFAR-10; fakes are Stable Diffusion 1.4 downscaled to 32x32.
#
# THE LABELS ARE INVERTED relative to SID_Set and to our convention. The card
# declares 0=FAKE, 1=REAL, and that is confirmed empirically: 400/400 sampled
# label-1 images phash-match (distance <= 4) genuine CIFAR-10 test images,
# 0/400 label-0 images do. `Source.label` below carries OUR value, so the
# ingest never touches the upstream integer after the filter.
#
# MEASURED 28 Aug: the cleanest data we have looked at — both classes are 100%
# JPEG (no format shortcut at all, unique so far) and a 4x4 thumbnail gets only
# AUC 0.652, against 0.885 for SID_Set.
#
# But every image is 32x32: 1,024 pixels against 200,704 in a 448 training
# crop, and the fakes were downscaled from SD's 512x512, which destroys the
# high-frequency artifacts a detector needs. Hence `pool="lowres32"` — see
# GEOMETRY_POOLS.
# LICENCE: MIT, confirmed 30 Aug from the source the rules link (§5.4's Kaggle
# page). Kaggle's own API reports `licenseNameNullable: "Other (specified in
# description)"`, and that description points at the authors' repo, which
# releases CIFAKE under the same MIT licence as CIFAR-10. An earlier note here
# and in the plan said CC BY-SA 4.0; that was wrong. Attribution is still owed
# to both Krizhevsky & Hinton (2009) and Bird & Lotfi (2024), IEEE Access.
CIFAKE_REAL = Source(
    key="CIFAKE/CIFAR-10", label=0, generator="", architecture="",
    licence="MIT", content="object", real_source="",
    note="CIFAR-10. Verified by phash against uoft-cs/cifar10. Anything else "
         "sourcing CIFAR-10 would double-count and leak against this.",
)
CIFAKE_FAKE = Source(
    key="CIFAKE/SD-1.4", label=1, generator="CIFAKE/SD-1.4",
    architecture="LatentDiffusion", licence="MIT",
    content="object", real_source="CIFAR-10",
    note="SD 1.4 rendered then downscaled to 32x32. Content-matched to the "
         "reals by CIFAR-10 class, which is why its content shortcut is weak.",
)

# ── WildFake / GAN_based ─────────────────────────────────────────────────────
# hy2628982280/WildFake on ModelScope, named in rules/rules.md 5.4. Served as
# zip archives read over HTTP ranges (aigid/rangezip.py) rather than downloaded.
#
# GAN_based.zip is 47 GB / 493,218 images / 6 generators. Verified by reading
# the archive's own central directory, not its CSVs, whose Category and Weight
# columns are degenerate (every row just repeats the generator name):
#
#   DF-GAN    191,980   Advanced/DF-GAN/samples/<class>/    text-to-image, CUB/COCO classes
#   GALIP     162,646   Advanced/GALIP/{cc12m,coco}/        text-to-image
#   styleGAN   80,000   Typical/styleGAN/sg3/<weights>/     StyleGAN3, 10 weight sets
#   GigaGAN    27,610   Advanced/GigaGAN/fake_images/
#   BigGAN     15,540   Typical/BigGAN/<imagenet class>/    class-conditional
#   starGAN    15,442   Typical/starGAN/<attribute>/        CelebA attribute edits
#
# CONTENT IS THE TRAP HERE. styleGAN's 80k are FFHQ/MetFaces faces and AFHQv2
# animals; starGAN's 15k are CelebA faces. That is 95k face images on the fake
# side. Our existing real pool (OpenImages scenes, CIFAR-10 objects) contains no
# faces at all, so pairing these against it would teach "face => fake" —
# the same content shortcut that makes SID_Set's fakes 0.885-separable, rebuilt
# deliberately. WildFake ships the matched reals, so they are ingested with it:
# ffhq and afhq for styleGAN, celebahq for starGAN, imagenet for BigGAN, coco
# for GALIP. `content` below is what drives that pairing.
_WF = "WildFake"
_GAN = "Images/GAN_based.zip"


def _wf_fake(name, prefix, archive, content, real_source, arch="GAN"):
    return Source(key=f"{_WF}/{name}", label=1, generator=f"{_WF}/{name}",
                  architecture=arch, licence="unknown", content=content,
                  real_source=real_source, note=f"{archive}::{prefix}")


def _wf_real(name, prefix, archive, content):
    return Source(key=f"{_WF}/real_{name}", label=0, generator="",
                  architecture="", licence="unknown", content=content,
                  real_source="", note=f"{archive}::{prefix}")


WILDFAKE_GAN_FAKES = {
    "GAN_based/Advanced/DF-GAN/":   _wf_fake("DF-GAN", "GAN_based/Advanced/DF-GAN/", _GAN, "object", "CUB+COCO"),
    "GAN_based/Advanced/GALIP/":    _wf_fake("GALIP", "GAN_based/Advanced/GALIP/", _GAN, "web", "CC12M+COCO"),
    "GAN_based/Advanced/GigaGAN/":  _wf_fake("GigaGAN", "GAN_based/Advanced/GigaGAN/", _GAN, "web", "LAION"),
    "GAN_based/Typical/styleGAN/":  _wf_fake("styleGAN3", "GAN_based/Typical/styleGAN/", _GAN, "face", "FFHQ+MetFaces+AFHQv2"),
    "GAN_based/Typical/BigGAN/":    _wf_fake("BigGAN", "GAN_based/Typical/BigGAN/", _GAN, "object", "ImageNet"),
    "GAN_based/Typical/starGAN/":   _wf_fake("starGAN", "GAN_based/Typical/starGAN/", _GAN, "face", "CelebA"),
}

# The matched reals. Each exists to answer a specific fake source, which is why
# faces are pulled at all -- we do not add a face corpus for its own sake.
WILDFAKE_GAN_REALS = {
    "Images/Real/ffhq.zip":      _wf_real("ffhq", "ffhq/", "Images/Real/ffhq.zip", "face"),
    "Images/Real/celebahq.zip":  _wf_real("celebahq", "celebahq/", "Images/Real/celebahq.zip", "face"),
    "Images/Real/afhq.zip":      _wf_real("afhq", "afhq/", "Images/Real/afhq.zip", "face"),
    "Images/Real/imagenet.zip":  _wf_real("imagenet", "imagenet/", "Images/Real/imagenet.zip", "object"),
    "Images/Real/coco.zip":      _wf_real("coco", "coco/", "Images/Real/coco.zip", "scene"),
}

# The unseen-GAN axis, as two explicit lists. This is the point of ingesting six
# generators at once, and both halves are PINNED -- assigned before any hashing,
# because a hash cannot honour an intent it cannot see.
#
# Measured: sha256(name) % 20 puts BigGAN in test. Letting that stand would have
# left training with no class-conditional GAN at all, and would have reported a
# 2-generator axis that actually contained 3. An axis is only as honest as its
# membership list, so the list decides, not the hash.
#
# Each held-out generator has a trained counterpart of the same kind, so the
# axis measures an unseen GENERATOR rather than unseen CONTENT:
#
#   GigaGAN (text-to-image)  held out against DF-GAN + GALIP, also text-to-image
#   starGAN (face)           held out against styleGAN3, also faces
#
# Without that pairing, "unseen GAN" would be measuring whether the model had
# ever seen a face, which is a different claim entirely.
WILDFAKE_HELD_OUT = frozenset({f"{_WF}/GigaGAN", f"{_WF}/starGAN"})
WILDFAKE_TRAINED = frozenset({f"{_WF}/DF-GAN", f"{_WF}/GALIP",
                              f"{_WF}/styleGAN3", f"{_WF}/BigGAN"})


# ── WildFake / Diffusion_based ───────────────────────────────────────────────
# 12 diffusion generators across single-zip and multi-part archives, plus
# Midjourney (11 multi-part zips with flat filenames). Verified counts against
# the label CSVs and archive central directories (28 Aug).
#
# DALLE.zip carries DALLE2 (Typical, 55,638) and DALLE3 (Advanced, 8,843).
# DALLE3 Advanced is FORBIDDEN by rules.md — excluded by prefix filter
# (only DALLE/Typical/ is ingested) and by FORBIDDEN_PATH_SUBSTRINGS.

_DIFF = "Images/Diffusion_based"

WILDFAKE_DIFF_FAKES: dict[str, dict[str, Source]] = {
    f"{_DIFF}/ADM.zip": {
        "ADM/": _wf_fake("ADM", "ADM/", f"{_DIFF}/ADM.zip",
                          "scene", "LSUN+ImageNet", arch="Diffusion"),
    },
    f"{_DIFF}/DALLE.zip": {
        "DALLE/Typical/": _wf_fake("DALLE2", "DALLE/Typical/", f"{_DIFF}/DALLE.zip",
                                    "web", "LAION", arch="Diffusion"),
    },
    f"{_DIFF}/DDIM.zip": {
        "DDIM/": _wf_fake("DDIM", "DDIM/", f"{_DIFF}/DDIM.zip",
                           "scene", "LSUN+ImageNet", arch="Diffusion"),
    },
    f"{_DIFF}/DDPM.zip": {
        "DDPM/": _wf_fake("DDPM", "DDPM/", f"{_DIFF}/DDPM.zip",
                           "scene", "LSUN+ImageNet", arch="Diffusion"),
    },
    f"{_DIFF}/Imagen.zip": {
        "Imagen/": _wf_fake("Imagen", "Imagen/", f"{_DIFF}/Imagen.zip",
                             "object", "LAION", arch="Diffusion"),
    },
    f"{_DIFF}/VQDM.zip": {
        "VQDM/": _wf_fake("VQDM", "VQDM/", f"{_DIFF}/VQDM.zip",
                           "scene", "LSUN+ImageNet", arch="VQDiffusion"),
    },
    # SD personalizedSD: DreamBooth + finetune in one archive
    f"{_DIFF}/SD/personalizedSD.zip": {
        "personalizedSD/dreambooth/": _wf_fake(
            "pSD-dreambooth", "personalizedSD/dreambooth/",
            f"{_DIFF}/SD/personalizedSD.zip", "web", "LAION", arch="Diffusion"),
        "personalizedSD/finetune/": _wf_fake(
            "pSD-finetune", "personalizedSD/finetune/",
            f"{_DIFF}/SD/personalizedSD.zip", "web", "LAION", arch="Diffusion"),
    },
    # SD with adaptors: ControlNet, LoRA, LyCORIS in one archive
    f"{_DIFF}/SD/SDwithAdaptor.zip": {
        "SDwithAdaptor/controlnet/": _wf_fake(
            "SD-controlnet", "SDwithAdaptor/controlnet/",
            f"{_DIFF}/SD/SDwithAdaptor.zip", "web", "LAION", arch="Diffusion"),
        "SDwithAdaptor/lora/": _wf_fake(
            "SD-lora", "SDwithAdaptor/lora/",
            f"{_DIFF}/SD/SDwithAdaptor.zip", "web", "LAION", arch="Diffusion"),
        "SDwithAdaptor/lycris/": _wf_fake(
            "SD-lycris", "SDwithAdaptor/lycris/",
            f"{_DIFF}/SD/SDwithAdaptor.zip", "web", "LAION", arch="Diffusion"),
    },
}

# Multi-part archives: each part_N.zip maps to the same Source via one prefix.
_mj_adv = _wf_fake("Midjourney-Advanced", "", "MJ/Adv", "web", "LAION", arch="Diffusion")
for _i in range(1, 8):
    WILDFAKE_DIFF_FAKES[f"{_DIFF}/Midjourney/Advanced/part_{_i}.zip"] = {"": _mj_adv}

_mj_typ = _wf_fake("Midjourney-Typical", "", "MJ/Typ", "web", "LAION", arch="Diffusion")
for _i in range(1, 5):
    WILDFAKE_DIFF_FAKES[f"{_DIFF}/Midjourney/Typical/part_{_i}.zip"] = {"": _mj_typ}

_sd_typ = _wf_fake("SD-original", "", "SD/Typ", "web", "LAION", arch="Diffusion")
for _i in range(1, 4):
    WILDFAKE_DIFF_FAKES[f"{_DIFF}/SD/originalSD/Typical/part_{_i}.zip"] = {"": _sd_typ}

_sdxl = _wf_fake("SDXL", "SDXL/", "SD/Adv", "web", "LAION", arch="Diffusion")
for _i in range(1, 8):
    WILDFAKE_DIFF_FAKES[f"{_DIFF}/SD/originalSD/Advanced/part_{_i}.zip"] = {"SDXL/": _sdxl}

WILDFAKE_DIFF_REALS: dict[str, dict[str, Source]] = {
    "Images/Real/church.zip":  {"church/":  _wf_real("church", "church/", "Images/Real/church.zip", "scene")},
    "Images/Real/laion5b.zip": {"laion5b/": _wf_real("laion5b", "laion5b/", "Images/Real/laion5b.zip", "web")},
}

WILDFAKE_DIFF_HELD_OUT = frozenset({f"{_WF}/Imagen", f"{_WF}/VQDM"})
WILDFAKE_DIFF_TRAINED = frozenset({
    f"{_WF}/ADM", f"{_WF}/DALLE2", f"{_WF}/DDIM", f"{_WF}/DDPM",
    f"{_WF}/pSD-dreambooth", f"{_WF}/pSD-finetune",
    f"{_WF}/SD-controlnet", f"{_WF}/SD-lora", f"{_WF}/SD-lycris",
    f"{_WF}/Midjourney-Advanced", f"{_WF}/Midjourney-Typical",
    f"{_WF}/SD-original", f"{_WF}/SDXL",
})


# ── WildFake / Other_based ──────────────────────────────────────────────────
# 4 generators in one archive: MAE and MAGE (masked autoencoders, ImageNet),
# VQGAN and VQVAE (vector-quantised, ImageNet/COCO).

_OTH = "Images/Other_based.zip"

WILDFAKE_OTHER_FAKES: dict[str, dict[str, Source]] = {
    _OTH: {
        "Other_based/Advanced/MAE/":  _wf_fake("MAE", "Other_based/Advanced/MAE/",
                                                _OTH, "object", "ImageNet", arch="MaskedAutoencoder"),
        "Other_based/Advanced/MAGE/": _wf_fake("MAGE", "Other_based/Advanced/MAGE/",
                                                _OTH, "object", "ImageNet", arch="MaskedAutoencoder"),
        "Other_based/Typical/VQGAN/": _wf_fake("VQGAN", "Other_based/Typical/VQGAN/",
                                                _OTH, "object", "ImageNet", arch="VQ"),
        "Other_based/Typical/VQVAE/": _wf_fake("VQVAE", "Other_based/Typical/VQVAE/",
                                                _OTH, "object", "COCO+ImageNet", arch="VQ"),
    },
}

WILDFAKE_OTHER_REALS: dict[str, dict[str, Source]] = {
    "Images/Real/imagenet.zip": {"imagenet/": _wf_real("imagenet", "imagenet/", "Images/Real/imagenet.zip", "object")},
    "Images/Real/coco.zip":     {"coco/":     _wf_real("coco", "coco/", "Images/Real/coco.zip", "scene")},
}

WILDFAKE_OTHER_HELD_OUT = frozenset({f"{_WF}/MAGE"})
WILDFAKE_OTHER_TRAINED = frozenset({f"{_WF}/MAE", f"{_WF}/VQGAN", f"{_WF}/VQVAE"})


DATASETS: dict[str, Dataset] = {
    "SID_Set": Dataset(
        repo="saberzl/SID_Set",
        sources={0: SID_REAL, 1: SID_FAKE},   # label 2 (tampered) absent on
                                              # purpose: mostly-real pixels
                                              # poison a binary label in both
                                              # directions.
        holdout_split="validation",
        columns=["img_id", "image", "label"],  # `mask` skipped: unused, and it
                                               # is most of the 140 GB
        pool="native",
    ),
    "CIFAKE": Dataset(
        repo="dragonintelligence/CIFAKE-image-dataset",
        sources={1: CIFAKE_REAL, 0: CIFAKE_FAKE},   # <- upstream keys, inverted
        holdout_split="test",
        columns=["image", "label"],
        pool="lowres32",
        axis="lowres32",
        note="32x32. Isolated pool; see GEOMETRY_POOLS.",
    ),
    "WildFake_GAN": Dataset(
        repo="hy2628982280/WildFake",
        sources={},
        holdout_split="",
        columns=[],
        pool="native",
        axis="unseen_gan",
        kind="zip",
        archives={_GAN: WILDFAKE_GAN_FAKES, **{k: {v.note.split("::")[1]: v}
                                               for k, v in WILDFAKE_GAN_REALS.items()}},
        held_out_generators=WILDFAKE_HELD_OUT,
        trained_generators=WILDFAKE_TRAINED,
        note="6 GAN generators + their matched reals. Read over HTTP ranges.",
    ),
    "WildFake_Diffusion": Dataset(
        repo="hy2628982280/WildFake",
        sources={},
        holdout_split="",
        columns=[],
        pool="native",
        axis="unseen_diffusion",
        kind="zip",
        archives={**WILDFAKE_DIFF_FAKES, **WILDFAKE_DIFF_REALS},
        held_out_generators=WILDFAKE_DIFF_HELD_OUT,
        trained_generators=WILDFAKE_DIFF_TRAINED,
        note="Diffusion generators (ADM, DALLE2, DDIM, DDPM, Imagen, VQDM, "
             "SD variants, Midjourney) + matched reals.",
    ),
    "WildFake_Other": Dataset(
        repo="hy2628982280/WildFake",
        sources={},
        holdout_split="",
        columns=[],
        pool="native",
        axis="unseen_other",
        kind="zip",
        archives={**WILDFAKE_OTHER_FAKES, **WILDFAKE_OTHER_REALS},
        held_out_generators=WILDFAKE_OTHER_HELD_OUT,
        trained_generators=WILDFAKE_OTHER_TRAINED,
        note="Non-GAN non-diffusion generators (MAE, MAGE, VQGAN, VQVAE) + "
             "matched reals.",
    ),
}


# ───────────────────────────────────────────────────── geometry pools ───────
#
# canonicalize() draws its target aspect ratio and short edge from a POOLED
# distribution, so that geometry carries no label information. Pooling has to
# happen across both classes — that is the whole point — but NOT across
# datasets of wildly different scale.
#
# CIFAKE is 32x32 and SID_Set is ~1024. One shared pool would do two bad
# things: SID images would sometimes be canonicalized to a 32-pixel short edge,
# throwing away 99.9% of their pixels, and CIFAKE images would sometimes be
# upsampled 32x, which is not a resize but an invention. Either way a dataset
# we added would silently degrade one we already had.
#
# So each dataset names a pool, and pooling happens within a pool. Both classes
# of a dataset always share one, which preserves the property that matters:
# within any comparison the model actually makes, geometry is uninformative.
GEOMETRY_POOLS = {
    "native": "Full-resolution sources. Pooled across every dataset in it, so "
              "cross-dataset geometry is uninformative too.",
    "lowres32": "CIFAKE only. 32x32 in, 32x32 out — no upsampling, ever. "
                "Isolated so it cannot pull the native pool's targets down.",
}


def pools_of(records) -> dict:
    """Group staged records by geometry pool. Canonicalization runs per pool."""
    out: dict[str, list] = {}
    for r in records:
        out.setdefault(DATASETS[r["dataset"]].pool, []).append(r)
    return out
