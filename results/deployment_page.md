# CrossGuard Deployment Framing

CrossGuard is designed for trust-and-safety triage, not automatic removal.

## Headline Operating Point

Use the **1% FPR operating point** for deployment framing:

| Field | Value |
|---|---:|
| Validation-fitted threshold | 0.9491 |
| Test TPR at this threshold | 0.8667 |
| Test FPR at this threshold | 0.0055 |
| Target FPR | 0.0100 |

This threshold is stricter than the balanced-accuracy threshold shipped in the
checkpoint (`0.8976`). That difference is intentional. The checkpoint threshold
records the balanced-accuracy operating point; the deployment page and live demo
lead with the stricter 1% FPR policy to reduce false accusations against real
creators.

## How To Interpret A Score

`pred` is the calibrated probability that an image is AI-generated. It is not
provenance, authorship, or policy intent.

| Score band | Decision | Rationale |
|---|---|---|
| `pred >= 0.9491` | Likely AI-generated | Above the validation-fitted 1% FPR threshold |
| `0.3779 <= pred < 0.9491` | Uncertain, route to review | Between the 5% and 1% FPR operating thresholds |
| `pred < 0.3779` | Lower risk | Below the broader 5% FPR review threshold |

The optional sidecar script applies this policy without changing the graded
prediction file:

```powershell
python scripts\triage_predictions.py predictions.json --calibration runs\ft_full_calibrated\calibration.json
```

## What The Model Is Good At

CrossGuard keeps high discrimination under the full robustness grid:

| Metric | Value |
|---|---:|
| Clean AUROC | 0.9921 [0.9915, 0.9927] |
| Macro robust AUROC | 0.9865 |
| Worst-cell AUROC | 0.9728 |
| Held-out all AUROC | 0.9913 [0.9905, 0.9920] |

The worst robustness cell is **`resize_0.25x`**, with AUROC `0.9728` and a
clean-to-worst drop of `0.0192`. That is the main robustness caveat to name in
deployment: very small thumbnails or aggressively downscaled reposts erase some
of the forensic signal.

## What The Score Is Not

CrossGuard does not read C2PA, SynthID, EXIF, watermark, account-history, or
reverse-search evidence. Those are production signals that would sit beside the
model score, not inside this offline benchmark.

The model should not be the only trigger for takedown or creator penalties. A
safe production flow is:

1. Score incoming media with CrossGuard.
2. Auto-prioritize items above `0.9491` for review queues.
3. Route the abstention band through reviewer context and provenance checks.
4. Keep low-risk items out of enforcement unless another independent signal
   raises concern.

## Known Limits

The shipped model is one branch. Branches B and C were cut for time and never
trained on final bundles, so fusion is unmeasured rather than failed.

The held-out generator results cover five WildFake generators:
GigaGAN, starGAN, Imagen, VQDM, and MAGE. They support a transfer claim within
that evaluation design, not a guarantee for every future generator.

The hard-real row is absent because no screenshot/CGI/render source cleared
licence review by the pre-registered cutoff. That is why the page leads with the
1% FPR operating point on the general test distribution instead of a hard-real
false-positive number.
