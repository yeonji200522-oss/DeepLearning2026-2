# Project 2 — DCGAN Latent Optimization

**Goal:** Find 200 latent vectors of shape `(128,)` for the supplied, already-trained generator. The final submission is one `z.pt` tensor of shape `(200,128)`.

**Important:** The course-provided `targets200.npy` contains restricted face images. Do not upload this file, generated comparison figures that contain these images, or the supplied model weights to this public repository. Use the assets **locally only**.

## Required course files

From the [instructor's project2 folder](https://github.com/maygodwithu/Class2026_DL/tree/main/project2), copy these files into this folder alongside `z_generator.py`:

- `generator.py`, `classifier.py`, `score.py` (supplied code, do not modify)
- `G.pt`, `classifier.pt`, `targets200.npy` (local/private only)

Install `torch`, `torchvision`, and `numpy`.

## Run

```bash
cd project2
python z_generator.py --n_random 1000 --topk 2 --steps 50 --lr 0.04 --feature_weight 0.5
python score.py --z z.pt
```

`z_generator.py` outputs `z.pt`, `compare.png`, and `worst.png`; keep all outputs containing course images private. The reference score in `score.py` is **not** the private leaderboard score.

## Method

1. Normalize the supplied target images to `[-1,1]`.
2. Generate a reproducible random reference for the evaluator's pixel and feature normalization constants.
3. Sample random latent vectors, retaining the top `--topk` starting points per image.
4. Freeze all weights of the provided generator `G` and classifier `C`; use Adam to optimize **only z**.
5. Loss: `(1-w) * MSE(G(z),target)/random_mse + w * (1-cosine(features(G(z)),features(target)))/(1-random_cosine)`, where `w=--feature_weight`.
6. After each optimizer step, project each z vector's L2 norm to `[8.6,14.0]`. The grading rule allows `[8.5,14.1]`.
7. Preserve the best-scoring z for every target across optimization steps and random initializations.

## Parameters and reproducibility

| Argument | Default | Meaning |
|---|---:|---|
| `--n_random` | 1000 | Initial random samples |
| `--topk` | 2 | Independent restarts per target |
| `--steps` | 50 | Adam iterations |
| `--lr` | 0.04 | Adam learning rate |
| `--feature_weight` | 0.5 | Feature-loss weight |
| `--batch_size` | 8 | Number of target faces per optimization batch |
| `--seed` | 0 | Random seed |
| `--device` | auto | CPU or CUDA |

The initial version was smoke-tested on **two** target images. It has **not yet been scored on all 200**. Record experiment scores, parameters, timings, and failure cases for the report.

**Deadline:** 2026-11-17 23:59 KST. Submit `z.pt` to leaderboard and a PDF report (at least two A4 pages) to e-class.

## Pixel-priority refinement (v2)

If you have already produced a strong `z.pt` (e.g., v1), **keep that file** and run a second pass without searching from scratch:

```bash
cp z.pt z_v1_backup.pt
python z_generator.py --resume_z z_v1_backup.pt --out z_pixel_v2.pt --pixel_steps 80 --pixel_lr 0.015 --pixel_feature_weight 0.20 --pixel_restarts 2 --max_feature_drop 0.02
python score.py --z z_pixel_v2.pt
```

For each target, v2 accepts a new vector only if its pixel score is higher, its combined **public-classifier** score is not below the starting score, and its feature score has declined by no more than 0.02. This is a **local reference-score safeguard**; it does not guarantee the hidden leaderboard feature score will improve. The original tensor is preserved as `z_v1_backup.pt` and an additional `z_before_pixel.pt` is written locally. Generated images and all `.pt` files remain gitignored.

If you prefer to start from scratch, first run v1 settings with `--pixel_steps 0`, then refine that output using `--resume_z`. Review pixel, feature, and combined scores from `score.py` before submitting.

## V4 — balanced multistart search (experimental)

A new branch retains V3 unchanged while testing candidate diversity and targeted refinement. The final deliverable is still **only** `z.pt`; do not commit model weights or the private face images.

Use the professor's supplied `generator.py`, `classifier.py`, `score.py`, `G.pt`, `classifier.pt`, and `targets200.npy` locally. Replace **only** `z_generator.py` with this branch's version, then simply run:

```bash
python z_generator.py
python score.py
```

V4 defaults: 3,000 initial latent samples, 4 starting candidates (2 balanced, 2 pixel-oriented), 120 Adam updates with candidate-specific feature loss weights, 80 pixel-focused refinement updates, and 60 extra updates for the lowest-scoring 25% of targets. Weights of G and C remain frozen. Latent L2 norm is constrained to [8.6, 14.0].

Small tests (6–12 targets) showed that changing feature weight alters pixel/feature tradeoffs, but **V4 has not been validated on all 200 and has no confirmed leaderboard improvement**. The public classifier feature score can differ substantially from the hidden leaderboard's feature score, so keep and compare previous submissions. The score target 91 is a goal, not a guaranteed result.

This program is a research experiment. Submit only after the `score.py` shape, NaN, and norm checks pass. Report the actual run parameters and results, including unsuccessful experiments.

## Multi-seed merge — default in the current V4 script

Running `python z_generator.py` now performs a full optimization for seed 0, additional shorter searches for seeds 1 and 2, and combines candidates **per target image**. The merge requires an improved local total score, no more than a 0.002 drop in local pixel score, and no more than a 0.015 drop in public-classifier feature score. Pixel and hard-face refinement run once on the merged candidates. Only one submission tensor `z.pt` is saved (shape `200 x 128`); there are no intermediate `.pt` outputs by default.

Use `--seeds 0,1,2,3` to override the seeds; `--extra_random`, `--extra_topk`, and `--extra_steps` control the additional search passes. More seeds cost more compute. Retain previous submissions privately and evaluate the actual leaderboard score: local reference improvement does not guarantee improvement against the private classifier.

The generator/classifier architecture and pretrained weights remain unchanged; do not upload the course-private targets or comparison images.
