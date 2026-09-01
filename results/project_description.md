# CrossGuard

CrossGuard detects AI-generated images and keeps working after the image has
been compressed, cropped, blurred, or shrunk to a thumbnail.

## How our solution addresses the problem statement

A detector that only works on pristine images does not work in practice. By the
time a synthetic image is worth catching, it has been screenshotted, re-encoded
by a messaging app, cropped to a profile frame, and reposted at reduced scale.
Each of those steps strips away the fine-grained generation artefacts that
detectors rely on.

CrossGuard makes robustness the training objective rather than an evaluation
afterthought. Every training step sees each image twice: once clean, once
through a randomly sampled distortion drawn from the same transform families the
problem statement names. The loss has three terms: a focal classification loss
on both views, a KL term tying the distorted view's prediction to the clean
view's, and a feature-space MSE, applied through a small residual correction
network, that pulls the two representations together. The model is penalised
directly for changing its answer when an image is degraded.

Across the 14-cell transform grid on the held-out test split, CrossGuard holds a
macro AUROC of 0.9865 against a clean baseline of 0.9921. The largest single
drop is 0.0192, at 0.25x downscaling.

| | AUROC |
|---|---:|
| Clean | 0.9921 |
| Macro across all 14 transform cells | 0.9865 |
| Worst cell (`resize_0.25x`) | 0.9728 |

**Calibrated output with thresholds fixed in advance.** CrossGuard returns a
calibrated probability, not a raw score. Temperature and operating thresholds
were fitted on the validation split and applied unchanged to test; no threshold
was derived from the final evaluation. At the deployed 1% false-positive
operating point (threshold 0.9491) the model reaches TPR 0.8667 at FPR 0.0055.

**Designed around the cost of a false accusation.** A missed synthetic image
stays up and remains catchable by other signals. A false positive tells a real
creator that their own work is machine-made. CrossGuard ships the stricter
threshold: 45 false positives across 8,116 real test images, or 1 in 180, at the
cost of missing 13.3% of synthetic images. Scores fall into three triage bands
so uncertain cases route to human review instead of automated action.

**Generalisation to unseen generators.** Five generators were withheld from
training entirely. On those generators CrossGuard scores 0.9913 AUROC overall:
0.9963 on unseen diffusion models, 0.9831 on unseen GANs, 0.9978 on other
architectures.

## Development tools

- **VS Code** with the WSL2 remote extension, on a Windows 11 host running
  Ubuntu.
- A cloud GPU runner for training, calibration, and final scoring during the
  hackathon. The public reproduction path is local to judge-controlled servers
  and does not require our cloud workspace or credentials.
- **Git** and **GitHub** for source control.
- **Claude Code** as an AI pair-programmer during development.
- **NVIDIA H100 80GB** for training. A B200 run was benchmarked and reverted:
  the workload is bound by network-volume read throughput rather than GPU
  compute, so the faster device returned roughly 10% for a 58% price premium.

## Models and APIs

- **DINOv2 ViT-L/14** (`vit_large_patch14_dinov2.lvd142m`, Apache-2.0) as the
  backbone, at 448×448 input.
- A single-logit classification head over globally average-pooled final patch
  tokens, plus a residual correction network used only by the training-time
  consistency loss.
- **LoRA** (rank 32, applied to attention and MLP projections) for the
  fine-tuning stage, after an initial linear-probe stage on frozen features.
- **306,114,561 parameters** in total, within the 2B limit.
- No external inference APIs. The model runs offline from a single checkpoint.

An architecture change during the event required both training stages to be
re-run on DINOv2 within the remaining day.

## Libraries and frameworks

- **PyTorch** and **torchvision**: training and inference
- **timm**: backbone architectures and pretrained weights
- **peft**: LoRA fine-tuning
- **scikit-learn**: AUROC, average precision, bootstrap confidence intervals
- **Pillow**: image decoding and the distortion grid
- **NumPy**, **pandas** and **PyArrow**: dataset manifest and sharded storage
- **imagehash**: perceptual-hash screening
- **huggingface-hub**: public dataset access where applicable
- **tqdm**: progress reporting

## Datasets and assets

| Asset | Licence | Use |
|---|---|---|
| SID_Set | CC BY 4.0 | Training and evaluation: real images, FLUX.1-dev synthetics |
| WildFake (GAN / Diffusion / Other) | Apache 2.0 per uploader | Training, and the held-out generator axes |
| CIFAKE | MIT, from CIFAR-10 and SD-1.4 | Evaluation only, thumbnail-regime row |
| COCO train2017 | CC BY 4.0 annotations, per-image Flickr terms | Enters the build through WildFake's real slices |
| DINOv2 ViT-L/14 | Apache-2.0 | Backbone weights |

The final build is 327,311 images: 257,433 training, 28,149 validation, 41,729
test, split so that no generator or source appears in more than one split.

### Assumption on the restricted WildFake slices

> The rules ban exactly two WildFake slices from training: the COCO val2017 reals and
> the DALL·E Advanced fakes that form the validation benchmark. We train on other
> WildFake slices. Both banned slices are excluded structurally, by path, at the point
> the archives are read: `coco2017/val2017/` for the reals, and `DALLE/Advanced/` for the
> DALL·E fakes, which share an archive with the DALLE2 images we do train on. A
> perceptual-hash screen over COCO val2017 runs alongside the path exclusion as a second
> check, covering the case of those images reappearing elsewhere under different
> filenames. The banned slices are used only as the demo-benchmark evaluation row and
> never touch training, model selection, calibration, or threshold choice.

## Limitations and next steps

CrossGuard ships a single model branch. Two additional branches were built and
tested but never trained on the final dataset, so branch fusion is unmeasured.

The false-positive classes that matter most in deployment, namely screenshots,
CGI and renders, heavily filtered photographs and AI-upscaled real images, could
not be
quantified, because no corpus covering them cleared licence review before the
project's data cutoff. They are characterised by mechanism in the error analysis
rather than by measurement.

The transform grid covers accidental degradation from ordinary redistribution.
It is not an adversarial evaluation: no attacker optimised against this model.
With more time, the priorities are a licence-cleared hard-real evaluation slice,
per-cell false-positive rates alongside the AUROC figures, and an adversarial
pass covering re-compression chains and deliberate thumbnail evasion.
