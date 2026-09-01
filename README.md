# CrossGuard

Robust detection of AI-generated images under real-world transformations.
TikTok TechJam 2026, problem statement 5.

## Judge / TikTok reproduction and inference

Judges should reproduce the training pipeline from the public source datasets
using the steps below. No model weights or private checkpoint are submitted with
this repository. All checkpoint paths in this README refer to artifacts produced
locally by that reproduction run.

For the final inference-only environment, after the reproduced calibrated
checkpoint exists, install the minimal dependency set:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

The reproduced training pipeline writes the calibrated checkpoint to:

```text
runs/ft_full_calibrated/best.pt
```

Then score any image directory with that locally produced checkpoint:

```powershell
.\.venv\Scripts\python.exe -m aigid.predict --input path\to\images --output predictions.json
```

The output is a JSON array containing `image_path` and `pred` for every
supported image. The score is the calibrated probability that the image is
AI-generated.

## Project overview

CrossGuard detects AI-generated images and keeps working after the image has
been compressed, cropped, blurred, or shrunk to a thumbnail. It ships a single
DINOv2 ViT-L/14 branch trained at 448 px, temperature-calibrated on validation
data, and evaluated once on the held-out test split.

Robustness is the training objective rather than an evaluation afterthought.
Every training step sees each image twice, once clean and once through a
randomly sampled distortion from the same transform families the problem
statement names, with a consistency loss penalising the model for changing its
answer when an image is degraded.

**The graded interface** takes an image directory and writes one JSON file:

```powershell
python -m aigid.predict --input <image-directory> --output predictions.json --device auto
```

The output is a sorted JSON array and nothing else:

```json
[
  {"image_path": "example.jpg", "pred": 0.8734}
]
```

`pred` is the calibrated probability that the image is AI-generated. Supported
inputs are JPG, JPEG, PNG, WebP, and BMP. Undecodable files still receive
`pred: 0.5`; read errors are written to `predictions.json.errors.json`, outside
the graded file.

### Headline results

Source report: `runs/ft_full_calibrated/test_test.json`, produced by
`scripts/score_test.py` on the calibrated checkpoint (test split, n = 41,729).

| Metric | Value |
|---|---:|
| Clean AUROC | 0.9921 [0.9915, 0.9927] |
| Macro robust AUROC, 14 transform cells | 0.9865 |
| Worst-cell AUROC | 0.9728 (`resize_0.25x`) |
| Clean-to-worst drop | 0.0192 |
| Held-out generator AUROC | 0.9913 [0.9905, 0.9920] |

Deployed at the validation-fitted 1% FPR operating point (threshold **0.9491**),
the model reaches **TPR 0.8667 at FPR 0.0055** on test: 45 false positives across
8,116 real images, or 1 in 180.

Full detail lives in `results/`:

- `results/final_result.md`: complete metric tables, held-out axes, calibration.
- `results/error_analysis.md`: false positives, false negatives, trade-offs.
- `results/deployment_page.md`: trust-and-safety deployment framing.
- `results/project_description.md`: the written project description.
- `PROJECT_HISTORY.md`: consolidated engineering decisions, experiments, dropped directions, and lessons learned.

### Model and libraries

- Backbone: DINOv2 ViT-L/14, `vit_large_patch14_dinov2.lvd142m`, Apache-2.0.
- Training path: Branch A only, linear probe then fine-tune, 448 px input,
  horizontal-flip TTA.
- Parameters: 306,114,561 total, below the 2B cap.
- Main libraries: PyTorch, torchvision, timm, peft, Pillow, NumPy, pandas,
  PyArrow, scikit-learn, imagehash, Hugging Face Hub, tqdm.
- Branches B and C are retained under `dropped_models/` as unshipped work. They
  were cut for time and never trained on final bundles.

## Setup and installation

For judge/TikTok inference only, use the minimal dependency file:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

For CUDA inference, install the CUDA wheel matching the judge environment before
the same minimal requirements. For CUDA 12.8:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

For full local reproduction, calibration, scoring, and data preparation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No checkpoint is committed or submitted separately. The reproduction steps below
create the calibrated checkpoint at:

```text
runs/ft_full_calibrated/best.pt
```

The default `aigid.predict` checkpoint path points there after reproduction.

Run inference:

```powershell
.\.venv\Scripts\python.exe -m aigid.predict --input path\to\images --output predictions.json
```

Verify the install without a checkpoint (exercises the full output contract, no
torch required):

```powershell
.\.venv\Scripts\python.exe -m aigid.predict --input path\to\images --output predictions.json --stub
```

Print the parameter count and the 2B compliance check:

```powershell
.\.venv\Scripts\python.exe -m aigid.predict --report-params
```

This command loads the reproduced checkpoint, so run it after
`runs/ft_full_calibrated/best.pt` has been created. Before reproduction, use
`--stub` to verify only the file-discovery and JSON-output contract.

Optional deployment triage sidecar, which applies the 1%/5% FPR bands without
changing the graded output:

```powershell
.\.venv\Scripts\python.exe scripts\triage_predictions.py predictions.json --calibration runs\ft_full_calibrated\calibration.json
```

## Steps to reproduce your results

The public reproduction path runs on a judge-controlled GPU server. It does not
use any Modal workspace, private Hugging Face artifact repository, or team
credential. It does need the public source datasets listed below, or local
mirrors of those datasets inside the judge environment.

The training code consumes a local data contract:

```text
data/manifest.parquet
data/full/<image files referenced by manifest.path>
```

The manifest must contain at least `path`, `label`, `split`, and `sha256`.
`label` uses `1 = AI-generated` and `0 = authentic`; `split` is one of `train`,
`val`, or `test`. The full schema used by this project is
`aigid.canon.MANIFEST_COLUMNS`.

### Local training on a judge server

Requires a CUDA GPU with 80 GB (the shipped run used an H100 at batch 16, 448
px; smaller cards work with `--batch-size 8 --grad-checkpoint`). Install the
CUDA wheels as in Installation above, then:

**Backbone initialization.** The training commands below use DINOv2 ViT-L/14
through `timm`'s public model name `vit_large_patch14_dinov2.lvd142m`; this is
the default `--backbone dinov2-l14-448` in `aigid.train`. No CrossGuard-trained
weights are supplied. If the judge server allows normal `timm` / Hugging Face
model downloads, the first training command initializes the public DINOv2
backbone automatically. If the judge pipeline pre-caches upstream model weights
instead, pass that local public DINOv2 file with `--weights <path>` on both
training stages. The project checkpoints `runs/lp/*.pt`, `runs/ft_full/*.pt`,
and `runs/ft_full_calibrated/best.pt` are produced by the steps below.

**1. Obtain the public source datasets.** SID_Set and CIFAKE are public Hugging
Face/Kaggle-hosted datasets; WildFake is public on ModelScope. The dataset
ledger below records how each source is used.

For WildFake, this repository keeps a public downloader:

```bash
python scripts/download_wildfake.py --out-dir data/wildfake_zips --workers 4
```

SID_Set and CIFAKE can either be mirrored by the build script through Hugging
Face, or supplied as local parquet mirrors under `data/sources/SID_Set` and
`data/sources/CIFAKE`.

**2. Build or verify the forbidden-content index.** The repository includes the
index used by the submitted run at `manifest/quarantine.npz`. To rebuild it
from the public WildFake COCO archive instead:

```bash
python scripts/build_quarantine_index.py
```

This index screens COCO val2017 by content. DALL-E Advanced is excluded by
upstream path.

**3. Produce the local manifest and image root.** This command uses the dataset
registry and split/canonicalization functions in `aigid/canon.py` to extract the
public datasets into canonical JPEGs under `data/full/`, and writes one manifest
row per image:

```bash
python scripts/prepare_public_build.py \
    --download-hf \
    --wildfake-root data/wildfake_zips \
    --image-root data/full \
    --out data/manifest.parquet
```

If SID_Set and CIFAKE are already mirrored locally, omit `--download-hf` and
point the script at those mirrors:

```bash
python scripts/prepare_public_build.py \
    --sid-root data/sources/SID_Set \
    --cifake-root data/sources/CIFAKE \
    --wildfake-root data/wildfake_zips \
    --image-root data/full \
    --out data/manifest.parquet
```

The important invariants are:

- train/val/test assignments come from `aigid.canon.assign_split`;
- COCO val2017 and DALL-E Advanced paths are excluded before training;
- the committed quarantine index is applied to catch COCO val2017 content under
  renamed paths;
- canonical images use pooled geometry and shared JPEG quality via
  `aigid.canon.canonicalize`;
- `path` is relative to `data/full/`.

For a small wiring check without building the full dataset, add
`--limit-per-dataset 10`. That option validates decoding, quarantine screening,
canonicalization, manifest writing, and output layout only; it is not a training
or metric reproduction.

If an external judge pipeline writes per-source manifests instead of using this
script, merge them locally:

```bash
python scripts/merge_manifests.py --inputs data/manifests/*.parquet --out data/manifest.parquet
```

Validate the local build before training:

```bash
python scripts/validate_public_build.py --manifest data/manifest.parquet --image-root data/full
```

**4. Stage one, linear probe.** Trains only the classification head on frozen
backbone features, which gives the fine-tune a stable starting point.

```bash
python -m aigid.train --stage lp \
    --manifest data/manifest.parquet --image-root data/full \
    --out runs/lp --epochs 2 --workers 16
```

Produces `runs/lp/best.pt` and `runs/lp/last.pt`. Roughly 1.5 to 2 hours on an
H100.

**5. Stage two, fine-tune.** Warm-starts from the linear probe and trains the
full network, including the consistency loss over the clean and distorted views.

```bash
python -m aigid.train --stage ft --ft-path full \
    --manifest data/manifest.parquet --image-root data/full \
    --resume runs/lp/last.pt \
    --out runs/ft_full --epochs 2 --batch-size 16 --workers 16
```

Produces `runs/ft_full/best.pt`, selected on worst-cell robust AUROC over the
validation split. Roughly 1.5 to 2 hours on an H100.

**6. Calibrate on validation.** Fits the temperature and the 1% / 5% FPR
operating thresholds. This reads the validation split only.

```bash
python scripts/calibrate.py \
    --checkpoint runs/ft_full/best.pt --split val \
    --manifest data/manifest.parquet --image-root data/full \
    --out runs/ft_full_calibrated/best.pt --cap 8000
```

Writes the calibrated checkpoint plus `runs/ft_full_calibrated/calibration.json`
beside it. The thresholds are fixed here and are never re-derived later.

**7. Score the test split, once.** Applies the validation-fitted thresholds
unchanged.

```bash
python scripts/score_test.py \
    --checkpoint runs/ft_full_calibrated/best.pt --split test \
    --manifest data/manifest.parquet --image-root data/full \
    --report runs/ft_full_calibrated/test_test.json
```

This regenerates `test_test.json`, the source of every number in this README and
in `results/`.

**8. Run inference with the checkpoint you just built.**

```bash
python -m aigid.predict \
    --input path/to/images --output predictions.json \
    --checkpoint runs/ft_full_calibrated/best.pt
```

Optionally apply the deployment triage bands, which do not alter the graded
output:

```bash
python scripts/triage_predictions.py predictions.json \
    --calibration runs/ft_full_calibrated/calibration.json
```

The build manifest is derived from SID_Set and CIFAKE on Hugging Face plus
WildFake on ModelScope.

### Dataset and licence ledger

| Asset | Licence posture | Use |
|---|---|---|
| SID_Set | CC BY 4.0 | Training/evaluation; label 0 reals and label 1 FLUX.1-dev synthetics |
| CIFAKE | MIT, derived from CIFAR-10 and SD-1.4 outputs | Evaluation-only thumbnail row |
| WildFake GAN, Diffusion, Other | Apache 2.0 by uploader declaration | Training/evaluation for unbanned slices; derivatives are rebuilt locally for reproduction |
| COCO train2017 | CC BY 4.0 annotations plus per-Flickr image terms | Reaches the build through WildFake's real slices |
| DINOv2 ViT-L/14 | Apache-2.0 | Shipped backbone |

WildFake assumption, stated for README and Devpost:

> The rules ban exactly two WildFake slices from training: the COCO val2017 reals and
> the DALL·E Advanced fakes that form the validation benchmark. We train on other
> WildFake slices. Both banned slices are excluded structurally, by path, at the point
> the archives are read: `coco2017/val2017/` for the reals, and `DALLE/Advanced/` for the
> DALL·E fakes, which share an archive with the DALLE2 images we do train on. A
> perceptual-hash screen over COCO val2017 runs alongside the path exclusion as a second
> check, covering the case of those images reappearing elsewhere under different
> filenames. The banned slices are used only as the demo-benchmark evaluation row and
> never touch training, model selection, calibration, or threshold choice.

## Limitations and what we would improve

CrossGuard ships a single model branch. Two further branches were built and
tested but never trained on the final dataset, so branch fusion is unmeasured
rather than failed.

The held-out generator axes cover five WildFake generators, so transfer outside
that corpus is unproven. Held-out GAN performance (0.9831) is measurably weaker
than held-out diffusion (0.9963), so architectures further from the training mix
should be assumed harder until measured.

The false-positive classes that matter most in deployment, namely screenshots,
CGI and renders, heavily filtered photographs and AI-upscaled real images, could
not be quantified: no corpus covering them cleared licence review before the
data cutoff. They are characterised by mechanism in `results/error_analysis.md`
rather than by measurement. This is the largest gap in the evaluation, and it is
why the deployment framing treats the score as triage rather than enforcement.

The transform grid covers accidental degradation from ordinary redistribution.
It is not an adversarial evaluation: no attacker optimised against this model.
The weakest cells all destroy high-frequency detail, which means a thumbnail is
the cheapest available evasion.

**Given more time**, in priority order: a licence-cleared hard-real evaluation
slice to turn the false-positive discussion into counts; per-cell false-positive
rates alongside the AUROC figures, which `scripts/score_test.py` already computes
but the shipped report predates; branch fusion, measured against the
pre-registered gate; and an adversarial pass covering re-compression chains and
deliberate thumbnail evasion.

## Team member contributions

| Member | Contribution |
|---|---|
| Perrin-Owen Heng Cheng Wei | Data ingestion, canonical manifest, quarantine, licence ledger |
| Brandon Jay-Han Bok and Feryan Krishany Jonandri | Branch A training path, checkpointing, DINOv2 migration |
| Tan Boon Leong | Transform grid, robustness harness, dropped branch prototypes |
| Brandon Jay-Han Bok, Tan Boon Leong and Lew Yik Chin | Calibration, final scoring, robustness/held-out reports |
| Tan Boon Leong, Lew Yik Chin and Perrin-Owen Heng Cheng Wei | Inference contract, deployment framing, README, demo triage sidecar |
