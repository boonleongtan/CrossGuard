# CrossGuard Final Results

Source: `runs/ft_full_calibrated/test_test.json`

Checkpoint: `/data/runs/ft_full_calibrated/best.pt`

Split: `test`

Weights: `swa`

Calibration: temperature `0.8613`, shipped balanced-accuracy threshold `0.8976`

Inference: horizontal-flip TTA on; 15 views scored for clean plus the 14-cell
robustness grid.

## Headline Metrics

| Metric | Value |
|---|---:|
| Clean AUROC | 0.9921 |
| Clean AUROC 95% CI | [0.9915, 0.9927] |
| Average precision | 0.9981 |
| Balanced accuracy at shipped threshold | 0.9424 |
| Shipped threshold | 0.8976 |
| TPR at shipped threshold | 0.8937 |
| FPR at shipped threshold | 0.0089 |
| Brier score | 0.0404 |
| ECE | 0.0445 |
| Test rows | 41,729 |

## Deployment Operating Points

The operating thresholds were fitted on validation by `scripts/calibrate.py` and
applied unchanged to test by `scripts/score_test.py`. They are not re-derived
from the final split.

| Operating point | Validation threshold | Test TPR | Test FPR | FPR gap vs target |
|---|---:|---:|---:|---:|
| 1% FPR | 0.9491 | 0.8667 | 0.0055 | -0.0045 |
| 5% FPR | 0.3779 | 0.9496 | 0.0334 | -0.0166 |

The deployment page uses the 1% FPR point as the headline because a false
positive wrongly flags a real creator. The shipped threshold remains the
balanced-accuracy threshold embedded in the checkpoint for auditability.

## Robustness Grid

| Cell | AUROC |
|---|---:|
| `jpeg_q90` | 0.9922 |
| `jpeg_q70` | 0.9912 |
| `jpeg_q50` | 0.9890 |
| `jpeg_q30` | 0.9847 |
| `blur_s0.5` | 0.9922 |
| `blur_s1.0` | 0.9900 |
| `blur_s2.0` | 0.9775 |
| `resize_0.50x` | 0.9892 |
| `resize_0.25x` | 0.9728 |
| `noise_s0.02` | 0.9906 |
| `noise_s0.05` | 0.9868 |
| `noise_s0.10` | 0.9740 |
| `jitter_20pct` | 0.9895 |
| `crop_80pct` | 0.9907 |

| Aggregate | Value |
|---|---:|
| Macro robust AUROC | 0.9865 |
| Worst-cell AUROC | 0.9728 |
| Worst cell | `resize_0.25x` |
| Clean-to-worst drop | 0.0192 |

The weakest transformation is `resize_0.25x`. This is the expected failure mode:
aggressive downscale-then-upscale destroys high-frequency forensic detail before
the 448 px model sees the image.

## Held-Out Generator Axes

| Axis | Generators | AUROC | 95% CI | Rows |
|---|---|---:|---:|---:|
| Held-out all | All five held-out generators | 0.9913 | [0.9905, 0.9920] | 37,867 |
| Unseen diffusion | WildFake/Imagen, WildFake/VQDM | 0.9963 | [0.9957, 0.9968] | 20,098 |
| Unseen GAN | WildFake/GigaGAN, WildFake/starGAN | 0.9831 | [0.9818, 0.9844] | 20,062 |
| Unseen other | WildFake/MAGE | 0.9978 | [0.9972, 0.9983] | 13,939 |

The GAN axis is the weakest held-out generator family. The training mix contains
less GAN exposure than diffusion exposure, so this axis should be described as
measured transfer to two withheld WildFake GANs, not universal GAN coverage.

## Named Rows

| Row | Result |
|---|---:|
| CIFAKE thumbnail regime | AUROC 0.9971 [0.9959, 0.9982] |

CIFAKE compares 32x32 CIFAR-10 reals against SD-1.4 synthetics upscaled to the
model input size. It is reported as a bounded thumbnail-regime check, not as a
headline result.

## Absent Rows

| Row | Status |
|---|---|
| SID_Set label 2 tampered slice | Not measured. The final build carries labels 0 and 1 only. |
| Hard reals | Not measured. No screenshot/CGI/render source cleared licence review by the 29 Aug cutoff. |

These rows are stated as limitations instead of filled with improvised data.

## Error Analysis Notes

Expected false-positive risks remain screenshots, UI-heavy images, heavily
filtered photos, CGI/renders, and AI-upscaled real photos. The hard-real slice
that would have quantified those risks did not clear licence review in time, so
the deployment page treats the model as a triage tool rather than an automatic
enforcement system.

Expected false-negative risks are strongest under severe information loss:
`resize_0.25x`, `noise_s0.10`, and `blur_s2.0` are the three weakest robustness
cells. Heavily degraded fakes and generators unlike the WildFake/SID_Set mix
should be routed through review, provenance checks, or a refreshed evaluation
slice before enforcement.
