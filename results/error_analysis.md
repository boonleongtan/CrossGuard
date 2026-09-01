# CrossGuard Error Analysis Note

Rules §5.5.5. Source: `runs/ft_full_calibrated/test_test.json` (test split,
n = 41,729: 33,613 AI-generated, 8,116 real). Every count below is derived from
that report; nothing here is estimated.

## 1. What the errors cost at each threshold

The threshold is the whole trade. The same model produces very different error
profiles depending on where it is set:

| Operating point | Threshold | False negatives | False positives | Total errors |
|---|---:|---:|---:|---:|
| 1% FPR (**deployed**) | 0.9491 | 4,482 | **45** | 4,527 |
| Balanced accuracy (checkpoint metadata) | 0.8976 | 3,573 | 72 | 3,645 |
| 5% FPR (review band floor) | 0.3779 | 1,693 | 271 | 1,964 |

Read the last column carefully: the deployed threshold has the **worst** total
error count of the three. That is deliberate. Moving from the 5% point to the 1%
point trades 2,789 additional missed fakes for 226 fewer wrongly-flagged real
creators, roughly twelve missed fakes bought per false accusation avoided.

We take that trade because the two errors are not symmetric in consequence. A
false negative means one synthetic image stays up, and it remains catchable by
every other signal in a moderation stack. A false positive tells a real creator
their genuine work is machine-made: a direct accusation, made by an automated
system, against someone who did nothing wrong. At platform scale the second
error is the one that produces appeals, press, and lost trust.

45 false positives out of 8,116 reals is 1 in 180. That is the number to quote
when someone asks what the model costs innocent users.

## 2. Representative false positives

The hard-real slice that would have characterised these by content type did not
clear licence review before the 29 Aug cutoff, so we report the mechanism and
say plainly what was not measured.

The 45 false positives at the deployed threshold are real images the model
scored above 0.9491. Based on what the training distribution contains, the
expected sources are images whose low-level statistics already resemble
generated content:

- **Screenshots and UI-heavy images**: synthetic-looking by construction, with
  flat regions, hard edges and no sensor noise.
- **Heavily filtered photos**: beauty filters and auto-enhance suppress exactly
  the sensor-level detail the model reads as evidence of a camera.
- **CGI, renders, and game captures**: genuinely rendered, so "real photo" is
  arguably the wrong label rather than the model being wrong.
- **AI-upscaled real photographs**: a real capture with generated pixels. The
  binary label itself breaks down here.

The last two are worth naming as a limitation of the task framing, not just of
the model: "is this AI-generated" is not binary once a real photo has been
through a generative upscaler.

**Not measured.** No screenshot/CGI/render corpus cleared licence review, so
these categories are reasoned from the training distribution rather than counted.
This is the single largest gap in the evaluation, and it is why the deployment
page treats the score as triage rather than enforcement.

## 3. Representative false negatives

4,482 fakes fall below the deployed threshold. The robustness grid shows where
they concentrate. The model's discrimination degrades most under severe
information loss:

| Weakest cell | AUROC | Drop vs clean |
|---|---:|---:|
| `resize_0.25x` | 0.9728 | −0.0192 |
| `noise_s0.10` | 0.9740 | −0.0180 |
| `blur_s2.0` | 0.9775 | −0.0146 |
| `jpeg_q30` | 0.9847 | −0.0074 |

All three of the worst cells destroy high-frequency detail. That is a coherent
single failure mode rather than three separate ones: the model reads
generation artefacts that live in fine detail, and aggressive downscaling,
heavy noise, and strong blur all erase that band before the image reaches the
448 px input.

The practical consequence is specific: **a thumbnail is the adversary's cheapest
evasion.** Reposting at quarter scale costs an attacker nothing and is the
single most effective degradation in the grid. Any deployment should score the
highest-resolution copy available rather than a generated thumbnail.

Generators unlike the training mix are the second expected source. Held-out GAN
transfer (0.9831) is measurably weaker than held-out diffusion (0.9963), so
architectures further from the SID_Set/WildFake mix should be assumed harder
until measured.

## 4. Calibration quality

| Metric | Value |
|---|---:|
| Brier score | 0.0404 |
| Expected calibration error | 0.0445 |

ECE of 0.0445 means the confidence values are usable but not exact. A score of
0.90 corresponds to roughly 0.86 to 0.94 empirical precision. Scores are sound for
ranking and thresholding; they should not be quoted to a user as a precise
probability.

## 5. Trade-offs we accepted

1. **Recall sacrificed for precision.** Deployed at 1% FPR, we miss 13.3% of
   fakes to hold false accusations at 1 in 180 reals. Stated, not hidden.
2. **One branch, not three.** Branches B and C were cut for time; fusion is
   unmeasured rather than failed.
3. **Robustness measured, adversaries not.** The grid covers accidental
   degradation from real redistribution. It is not an adversarial evaluation:
   no attacker optimised against this model.
4. **The hard-real gap is the honest weak point.** The error class that matters
   most for deployment is the one we could not quantify.

## 6. What we would measure next

In priority order, and all blocked on data rather than method:

1. A licence-cleared hard-real slice covering screenshots, CGI and filtered
   photos, to turn §2 from reasoning into counts.
2. Per-cell false-positive rates, so robustness is reported in error counts and
   not only AUROC. `scripts/score_test.py` already computes this
   (`cell_metrics`); the shipped report predates that code path and does not
   contain it. One rescore closes the gap.
3. An adversarial pass: re-compression chains, generative upscaling, and
   deliberate thumbnail evasion.
