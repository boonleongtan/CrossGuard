"""Distortion pipeline — one module, imported by both training and evaluation so
the two can never drift.

Two entry points:

  * ``apply_grid_cell(img, name)`` — the 14 deterministic evaluation cells of
    §6.1, one transform at one severity, reproducing rules §5.2 exactly. Used by
    M4's robustness harness and the WildFake demo benchmark.

  * ``sample_train_distortion(img, rng)`` — the four training tiers of §5. Draws
    a tier by its share, then that many transforms from *distinct* groups. The
    training distribution is heavier than the eval grid on purpose (L3/L4 chain
    2–5 transforms), but L2 (35% of images) is exactly the single-transform view
    the grid measures, so the graded regime stays well represented.

PIL in, PIL out; numpy for the pixel math. No torch — the eval harness and a
torch-free ``predict --stub`` both import this.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ─────────────────────────────────────────────────────────── primitives ──────
# Each takes a PIL RGB image plus one severity value and returns a PIL RGB image
# at the same size — except centre crop, whose whole point is to change framing.


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0")
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    # PIL's GaussianBlur radius is effectively the kernel std-dev.
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


_RESAMPLE = (Image.Resampling.NEAREST, Image.Resampling.BILINEAR,
             Image.Resampling.BICUBIC, Image.Resampling.LANCZOS)


def resize_back(img: Image.Image, scale: float, down=Image.Resampling.BICUBIC,
                up=Image.Resampling.BICUBIC) -> Image.Image:
    """Downscale by ``scale`` then back up to the original size — the rules'
    thumbnail-generation analog. Bicubic both ways by default, which is what the
    §6.1 grid cells use; the training group draws ``down``/``up`` independently
    from ``_RESAMPLE`` (§5: "mixed interpolation")."""
    w, h = img.size
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), down)
    return small.resize((w, h), up)


def gaussian_noise(img: Image.Image, sigma: float,
                   rng: np.random.Generator) -> Image.Image:
    """Additive Gaussian noise; ``sigma`` in [0,1] units of pixel intensity
    (rules §5.2: σ = 0.02, 0.05, 0.10)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr + rng.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "RGB")


def webp(img: Image.Image, quality: int) -> Image.Image:
    """The other lossy codec platforms re-encode with (§5 JPEG group)."""
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def _kernel5(weights: np.ndarray) -> ImageFilter.Kernel:
    """A 5×5 PIL convolution kernel from a numpy weight array, normalised."""
    total = float(weights.sum()) or 1.0
    return ImageFilter.Kernel((5, 5), (weights / total).ravel().tolist(), scale=1.0)


_KY, _KX = np.mgrid[-2:3, -2:3]


def defocus_blur(img: Image.Image, radius: float) -> Image.Image:
    """Out-of-focus (disc) blur — a flat-topped kernel, unlike Gaussian, so it
    kills high frequencies differently. §5 blur group."""
    k = ((_KX ** 2 + _KY ** 2) <= float(radius) ** 2).astype(np.float64)
    if k.sum() == 0:
        k[2, 2] = 1.0
    return img.filter(_kernel5(k))


def motion_blur(img: Image.Image, length: float, angle_deg: float) -> Image.Image:
    """Mild linear motion blur at ``angle_deg``. §5 blur group."""
    a = np.deg2rad(float(angle_deg))
    perp = np.abs(-np.sin(a) * _KX + np.cos(a) * _KY)
    along = np.abs(np.cos(a) * _KX + np.sin(a) * _KY)
    k = ((perp <= 0.5) & (along <= float(length) / 2.0)).astype(np.float64)
    if k.sum() == 0:
        k[2, 2] = 1.0
    return img.filter(_kernel5(k))


def shot_noise(img: Image.Image, photons: float,
               rng: np.random.Generator) -> Image.Image:
    """Poisson (photon) noise — signal-dependent, unlike additive Gaussian.
    Lower ``photons`` is noisier. §5 noise group."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    out = rng.poisson(arr * float(photons)) / float(photons)
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), "RGB")


def impulse_noise(img: Image.Image, amount: float,
                  rng: np.random.Generator) -> Image.Image:
    """Salt-and-pepper. ``amount`` is the total fraction of corrupted pixels.
    §5 noise group."""
    arr = np.asarray(img).copy()
    m = rng.random(arr.shape[:2])
    arr[m < float(amount) / 2] = 0
    arr[m > 1.0 - float(amount) / 2] = 255
    return Image.fromarray(arr, "RGB")


def tone_curve(img: Image.Image, gamma: float, s_strength: float) -> Image.Image:
    """Gamma plus a gentle S-curve, applied through a 256-entry LUT — the
    non-linear tone mapping every phone camera and photo app applies. §5 colour
    group."""
    x = np.linspace(0.0, 1.0, 256) ** float(gamma)
    x = x + float(s_strength) * (x - 0.5) * (1.0 - np.abs(2.0 * x - 1.0))
    lut = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(lut[np.asarray(img)], "RGB")


def clahe(img: Image.Image, clip_limit: float = 2.0, grid: int = 8) -> Image.Image:
    """Contrast-limited adaptive histogram equalisation on the luminance channel,
    with the tile LUTs bilinearly interpolated (the standard formulation). §5
    colour group — it is what "auto-enhance" does to a photo before upload."""
    ycc = np.asarray(img.convert("YCbCr")).copy()
    y = ycc[..., 0]
    h, w = y.shape
    gy, gx = max(1, min(grid, h)), max(1, min(grid, w))
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)

    luts = np.empty((gy, gx, 256), np.float32)
    for i in range(gy):
        for j in range(gx):
            tile = y[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float32)
            limit = max(1.0, float(clip_limit) * max(tile.size, 1) / 256.0)
            excess = float(np.maximum(hist - limit, 0.0).sum())
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = np.cumsum(hist)
            luts[i, j] = cdf / max(cdf[-1], 1e-9) * 255.0

    # Bilinear blend between the four tile centres nearest each pixel.
    cy, cx = (ys[:-1] + ys[1:]) / 2.0, (xs[:-1] + xs[1:]) / 2.0
    rr = np.clip(np.interp(np.arange(h), cy, np.arange(gy)), 0, gy - 1)
    cc = np.clip(np.interp(np.arange(w), cx, np.arange(gx)), 0, gx - 1)
    i0 = np.floor(rr).astype(int)[:, None]
    j0 = np.floor(cc).astype(int)[None, :]
    i1 = np.minimum(i0 + 1, gy - 1)
    j1 = np.minimum(j0 + 1, gx - 1)
    fy = (rr - np.floor(rr))[:, None]
    fx = (cc - np.floor(cc))[None, :]
    top = luts[i0, j0, y] * (1 - fx) + luts[i0, j1, y] * fx
    bot = luts[i1, j0, y] * (1 - fx) + luts[i1, j1, y] * fx
    ycc[..., 0] = np.clip(top * (1 - fy) + bot * fy, 0, 255).astype(np.uint8)
    return Image.fromarray(ycc, "YCbCr").convert("RGB")


def aspect_jitter(img: Image.Image, ax: float, ay: float) -> Image.Image:
    """Slight non-uniform rescale — the aspect half of §5's crop group."""
    w, h = img.size
    return img.resize((max(1, round(w * float(ax))), max(1, round(h * float(ay)))),
                      Image.Resampling.BICUBIC)


def _perspective_coeffs(out_corners, in_corners):
    """PIL's PERSPECTIVE transform maps *output* coords back to *input* coords,
    so solve for the 8 coefficients in that direction."""
    a, b = [], []
    for (x, y), (u, v) in zip(out_corners, in_corners):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    return tuple(np.linalg.solve(np.asarray(a, float), np.asarray(b, float)))


def perspective(img: Image.Image, jitter: float,
                rng: np.random.Generator) -> Image.Image:
    """Slight perspective warp — corners displaced by up to ``jitter`` × the
    short edge. The perspective half of §5's crop group."""
    w, h = img.size
    d = float(jitter) * min(w, h)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(x + rng.uniform(-d, d), y + rng.uniform(-d, d)) for x, y in src]
    return img.transform((w, h), Image.Transform.PERSPECTIVE,
                         _perspective_coeffs(dst, src),
                         resample=Image.Resampling.BICUBIC)


def color_jitter(img: Image.Image, brightness: float, contrast: float,
                 saturation: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, round(w * frac)), max(1, round(h * frac))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def _seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


# ───────────────────────────────────────────── the 14 evaluation cells ───────
# One transform × one severity per cell, reproducing rules §5.2 exactly (§6.1).
# The colour-jitter cell is one fixed representative point inside the ±20% box;
# the grid is deterministic, so noise is seeded from the cell name.

_GRID = [
    ("jpeg_q90",     "jpeg",   lambda im, r: jpeg(im, 90)),
    ("jpeg_q70",     "jpeg",   lambda im, r: jpeg(im, 70)),
    ("jpeg_q50",     "jpeg",   lambda im, r: jpeg(im, 50)),
    ("jpeg_q30",     "jpeg",   lambda im, r: jpeg(im, 30)),
    ("blur_s0.5",    "blur",   lambda im, r: gaussian_blur(im, 0.5)),
    ("blur_s1.0",    "blur",   lambda im, r: gaussian_blur(im, 1.0)),
    ("blur_s2.0",    "blur",   lambda im, r: gaussian_blur(im, 2.0)),
    ("resize_0.50x", "resize", lambda im, r: resize_back(im, 0.50)),
    ("resize_0.25x", "resize", lambda im, r: resize_back(im, 0.25)),
    ("noise_s0.02",  "noise",  lambda im, r: gaussian_noise(im, 0.02, r)),
    ("noise_s0.05",  "noise",  lambda im, r: gaussian_noise(im, 0.05, r)),
    ("noise_s0.10",  "noise",  lambda im, r: gaussian_noise(im, 0.10, r)),
    ("jitter_20pct", "jitter", lambda im, r: color_jitter(im, 1.2, 0.8, 1.2)),
    ("crop_80pct",   "crop",   lambda im, r: center_crop(im, 0.80)),
]
assert len(_GRID) == 14, "rules §5.2 → 14 cells"

GRID_CELL_NAMES = [n for n, _, _ in _GRID]
GRID_CELL_GROUP = {n: g for n, g, _ in _GRID}
_GRID_FN = {n: fn for n, _, fn in _GRID}


def apply_grid_cell(img: Image.Image, name: str) -> Image.Image:
    """Apply one named evaluation cell. Deterministic."""
    try:
        fn = _GRID_FN[name]
    except KeyError:
        raise KeyError(f"unknown grid cell {name!r}; one of {GRID_CELL_NAMES}")
    return fn(img.convert("RGB"), np.random.default_rng(_seed(name)))


# ────────────────────────────────────────────── the four training tiers ──────
# §5: sampled per image. "heavy" (L4 only) shifts every group to its upper range.
#
# The six groups and their severity ranges are §5's distortion policy verbatim:
#   JPEG Q ∈ [30,95] + double-JPEG + WebP
#   Gaussian σ ∈ [0.3,2.5] + defocus + mild motion
#   downscale 0.25–0.7× and back with mixed interpolation
#   noise σ ∈ [0.01,0.12] + shot + impulse
#   colour ±25% + tone curve + CLAHE
#   crop 70–95% + slight aspect/perspective
# Every group is applied identically to both classes.

def _g_jpeg(im, r, heavy):
    lo, hi = (30, 55) if heavy else (55, 95)
    kind = str(r.choice(["single", "double", "webp"], p=[0.5, 0.3, 0.2]))
    if kind == "webp":
        return webp(im, int(r.integers(lo, hi + 1)))
    im = jpeg(im, int(r.integers(lo, hi + 1)))
    if kind == "double":                      # a re-share: two generations, two Qs
        im = jpeg(im, int(r.integers(lo, hi + 1)))
    return im


def _g_blur(im, r, heavy):
    kind = str(r.choice(["gauss", "defocus", "motion"], p=[0.6, 0.2, 0.2]))
    if kind == "gauss":
        return gaussian_blur(im, r.uniform(1.4, 2.5) if heavy else r.uniform(0.3, 1.4))
    if kind == "defocus":
        return defocus_blur(im, r.uniform(1.5, 2.5) if heavy else r.uniform(0.8, 1.6))
    return motion_blur(im, r.uniform(3.0, 5.0) if heavy else r.uniform(2.0, 3.5),
                       float(r.uniform(0.0, 180.0)))


def _g_resize(im, r, heavy):
    scale = r.uniform(0.25, 0.40) if heavy else r.uniform(0.40, 0.70)
    return resize_back(im, float(scale),
                       down=_RESAMPLE[int(r.integers(len(_RESAMPLE)))],
                       up=_RESAMPLE[int(r.integers(len(_RESAMPLE)))])


def _g_noise(im, r, heavy):
    kind = str(r.choice(["gauss", "shot", "impulse"], p=[0.6, 0.25, 0.15]))
    if kind == "gauss":
        return gaussian_noise(
            im, r.uniform(0.06, 0.12) if heavy else r.uniform(0.01, 0.06), r)
    if kind == "shot":
        return shot_noise(im, r.uniform(15, 45) if heavy else r.uniform(45, 180), r)
    return impulse_noise(
        im, r.uniform(0.02, 0.05) if heavy else r.uniform(0.002, 0.02), r)


def _g_jitter(im, r, heavy):
    """§5's colour group: brightness/contrast/saturation ±25%, tone curve, CLAHE."""
    kind = str(r.choice(["jitter", "tone", "clahe"], p=[0.6, 0.25, 0.15]))
    if kind == "jitter":
        span = 0.25 if heavy else 0.15
        b, c, s = 1.0 + r.uniform(-span, span, 3)
        return color_jitter(im, float(b), float(c), float(s))
    if kind == "tone":
        return tone_curve(im, float(r.uniform(0.75, 1.35)),
                          float(r.uniform(-0.35, 0.35) if heavy
                                else r.uniform(-0.20, 0.20)))
    return clahe(im, clip_limit=float(r.uniform(2.0, 4.0) if heavy
                                      else r.uniform(1.5, 2.5)))


def _g_crop(im, r, heavy):
    im = center_crop(im, float(r.uniform(0.70, 0.82) if heavy
                               else r.uniform(0.82, 0.95)))
    if r.random() < 0.5:
        im = aspect_jitter(im, float(r.uniform(0.93, 1.07)),
                           float(r.uniform(0.93, 1.07)))
    if r.random() < 0.35:
        im = perspective(im, 0.030 if heavy else 0.015, r)
    return im


_GROUPS = {"jpeg": _g_jpeg, "blur": _g_blur, "resize": _g_resize,
           "noise": _g_noise, "jitter": _g_jitter, "crop": _g_crop}

# (name, share, (min transforms, max transforms))
TIERS = [
    ("L1", 0.10, (0, 0)),   # clean — identity
    ("L2", 0.35, (1, 1)),   # mild — the eval-cell view
    ("L3", 0.40, (2, 3)),   # moderate — distinct groups
    ("L4", 0.15, (4, 5)),   # heavy — upper severity range
]
assert abs(sum(s for _, s, _ in TIERS) - 1.0) < 1e-9


def sample_train_distortion(img: Image.Image,
                            rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Return ``(distorted_image, tier_name)``. At most one transform per group."""
    img = img.convert("RGB")
    r, cum = rng.random(), 0.0
    tier, lo, hi = TIERS[-1][0], *TIERS[-1][2]
    for name, share, (a, b) in TIERS:
        cum += share
        if r <= cum:
            tier, lo, hi = name, a, b
            break
    k = int(rng.integers(lo, hi + 1))
    if k == 0:
        return img, tier
    groups = list(_GROUPS)
    rng.shuffle(groups)
    heavy = tier == "L4"
    for g in groups[:k]:
        img = _GROUPS[g](img, rng, heavy)
    return img, tier
