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
