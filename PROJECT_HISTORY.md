# CrossGuard Project History

This document preserves the useful engineering history of CrossGuard after the hackathon: what shipped, what was measured, what was deliberately not shipped, and the lessons that affected the final result.

It is a post-hackathon record, not a reproduction plan. For setup and final metrics, use `README.md` and `results/final_result.md`.

## Final Outcome

CrossGuard shipped as a single-branch detector for AI-generated images under real-world transformations. The submitted path uses DINOv2 ViT-L/14 at 448 px, horizontal-flip test-time augmentation, and validation-fitted temperature scaling. It produces one calibrated probability per input image.

The final evaluation recorded clean AUROC 0.9921, macro robust AUROC 0.9865, worst-cell AUROC 0.9728 under `resize_0.25x`, and held-out-generator AUROC 0.9913. The full metric table and confidence intervals are in `results/final_result.md`.

## Compliance and Data Decisions

Data provenance was treated as an engineering constraint. Training used whole-image real and synthetic labels only; SID_Set label 2 tampered images were excluded because they do not have a reliable binary target. CIFAKE was evaluation-only because its 32x32 images are useful as a thumbnail stress test but not as a main 448 px training signal.

WildFake's COCO val2017 reals and DALL·E Advanced fakes were excluded from training. The ingestion path used exact path exclusions plus a perceptual-hash quarantine over the 4,998 original COCO val2017 images. End-to-end checks confirmed that sampled prohibited images were dropped and sampled non-prohibited images passed.

The final build contained 327,311 rows: 257,433 train, 28,149 validation, and 41,729 held-out rows. Held-out evaluation was by generator rather than a random image split, with unseen GAN, diffusion, and other-generator axes paired with content-matched reals where possible.

## Robustness Design

The fixed evaluation grid has 14 cells: JPEG compression at four quality levels, Gaussian blur at three severities, downscale-and-upscale at 0.5x and 0.25x, Gaussian noise at three severities, colour jitter, and centre crop.

Training used paired views and consistency losses so transformed views of the same image retained the same target. Inference uses horizontal-flip TTA by averaging logits before calibration. The weakest final condition was aggressive downscale-and-upscale, consistent with removal of high-frequency forensic evidence.

## Model Directions Considered

### Final branch

The final model was Branch A: DINOv2 ViT-L/14 with a classification head, linear-probe-to-full-fine-tune training, global-average pooling, and 448 px inputs. It has 306,114,561 parameters, comfortably below the challenge limit.

### Multi-branch fusion

Earlier designs included a frozen CLIP linear probe for a second semantic representation and a small SRM-style residual CNN for low-level forensic cues. The planned fusion was a logistic stacker fitted only on validation outputs, and each component had to pass a pre-registered robustness gate; ties favoured the simpler system.

Neither auxiliary branch completed training on the final data, so no final bundle or measured fusion gain exists. The team shipped Branch A alone rather than imply an ensemble benefit that was never measured. The prototypes in `dropped_models/` remain as engineering reference only and are not imported by the released model.

### Alternative backbone

EVA02-L/14 was retained as an eligible fallback, but was not trained for the final submission. It needs CLIP-style normalization rather than the ImageNet normalization used by the shipped branch, so switching it would have required a separate controlled run rather than a configuration-only swap.

## Infrastructure Experiments and Lessons

### GPU and dataloader experiments

The workload was limited primarily by image reads and decoding from network storage, not arithmetic throughput. A faster GPU improved end-to-end throughput by only about 10% while costing substantially more. Increasing from 16 to 48 workers made the run roughly 40% slower because additional workers contended for the same storage path.

The resulting operating point was an H100-class GPU, batch size 16, and 16 workers. Synthetic-tensor benchmarks were rejected as planning evidence because they bypassed image reads, decoding, and augmentation.

### Materialization

The published dataset uses large parquet shards. Random access repeatedly decompressed shards and left the accelerator underutilized. Materializing shards to loose files before full training was therefore made a required stage. The central systems lesson was to benchmark the complete dataloader path, not only the model forward pass.

### Training schedule

The final run used two linear-probe epochs followed by two full-fine-tuning epochs. Checkpoints were persisted at epoch boundaries, and the calibrated fine-tuning checkpoint is the submitted model artifact.

## Calibration and Operating Policy

Calibration was separate from model training and used validation data only. It evaluated clean images plus all 14 transform cells, fit a temperature by minimizing negative log likelihood, and selected the shipped threshold by balanced accuracy.

The calibrated checkpoint records validation-fitted 1% and 5% false-positive-rate operating points. Those thresholds were applied unchanged during the one held-out evaluation; they were never re-fitted there. The deployment policy uses the stricter 1% FPR threshold for review-queue prioritization, while scores between the 5% and 1% thresholds are routed for human review.

## Evaluation Scope and Known Limits

The final evaluation includes the 14-cell robustness grid, held-out-generator axes, calibration metrics, confidence intervals, and the CIFAKE thumbnail regime. It does not claim coverage that the project did not measure.

- The hard-real slice was dropped because no screenshot, CGI, or render source completed licence review in time.
- Tampered-image detection was not measured because the final binary build excluded that class.
- Fusion is unmeasured because the auxiliary branches did not complete final runs.
- Transfer beyond the evaluated WildFake generator families remains unproven.

These boundaries are deliberate reporting choices, not zero-valued results.

## Repository Guide

| Location | Purpose |
|---|---|
| `README.md` | Public overview, setup, dataset ledger, and headline results |
| `results/final_result.md` | Complete final metric table and limitations |
| `results/deployment_page.md` | Trust-and-safety triage framing |
| `rules/rules.md` | Local copy of the challenge rules used by the project |
| `aigid/` | Shipped data, model, training, and prediction code |
| `scripts/` | Build, training, calibration, evaluation, and demo entrypoints |
| `tests/` | Shipped-path and calibration safety tests |
| `dropped_models/` | Unshipped research prototypes; not imported by inference |

## Takeaways

1. Robustness work is constrained as much by data access and I/O as by model capacity.
2. Generator-held-out evaluation is more informative than a random image split when the aim is transfer to unfamiliar generators.
3. A smaller, fully measured model is preferable to an unmeasured ensemble.
4. Calibration and operating thresholds must be fitted on validation data and carried unchanged into final evaluation.
5. Honest negative results and explicit limits make the final system easier to trust and reproduce.
