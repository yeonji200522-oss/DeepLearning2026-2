"""Optimize the 200 latent vectors, not the supplied GAN or classifier weights."""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from generator import load_generator
from classifier import load_classifier


def project_norm_(z):
    """Project each z into a safe inner interval of the score.py norm limits."""
    with torch.no_grad():
        norm = z.norm(dim=1, keepdim=True).clamp_min(1e-8)
        z.mul_(norm.clamp(8.6, 14.0) / norm)


def normalized_features(C, x):
    return F.normalize(C.features(x), dim=1)


@torch.no_grad()
def reference_stats(G, C, targets, target_features, batch=50):
    """Compute score.py reference with the same seeded 1000 random faces."""
    rng = torch.Generator().manual_seed(999)
    faces, features = [], []
    for _ in range(10):
        latents = torch.randn(100, 128, generator=rng).to(targets.device)
        for j in range(0, 100, batch):
            x = G(latents[j:j + batch])
            faces.append(x)
            features.append(normalized_features(C, x))
    faces = torch.cat(faces)
    features = torch.cat(features)
    rand_mse = torch.cdist(targets.flatten(1), faces.flatten(1)).square().mean(1)
    rand_mse = rand_mse / targets[0].numel()
    rand_cos = (target_features @ features.T).mean(1)
    return rand_mse, rand_cos


def scores(x, target, fx, ft, rand_mse, rand_cos):
    mse = (x - target).square().mean((1, 2, 3))
    cosine = (fx * ft).sum(1)
    pixel = (1 - mse / rand_mse).clamp(0, 1)
    feature = ((cosine - rand_cos) / (1 - rand_cos).clamp_min(1e-6)).clamp(0, 1)
    return (pixel + feature) / 2, pixel, feature, mse, cosine


@torch.no_grad()
def random_start(G, C, target, ft, rand_mse, rand_cos, n_random, topk, chunk=32):
    """Find the best topk initial z vectors independently for every image."""
    vals, pixels_all, latents = [], [], []
    for start in range(0, n_random, chunk):
        z = torch.randn(min(chunk, n_random - start), 128, device=target.device)
        project_norm_(z)
        x = G(z)
        fx = normalized_features(C, x)
        mse = torch.cdist(target.flatten(1), x.flatten(1)).square() / target[0].numel()
        cos = ft @ fx.T
        pixel = (1 - mse / rand_mse[:, None]).clamp(0, 1)
        feature = ((cos - rand_cos[:, None]) /
                   (1 - rand_cos[:, None]).clamp_min(1e-6)).clamp(0, 1)
        vals.append((pixel + feature) / 2)
        pixels_all.append(pixel)
        latents.append(z)
    vals, latents = torch.cat(vals, dim=1), torch.cat(latents)
    pixels_all = torch.cat(pixels_all, dim=1)
    # Mix high-total-score and high-pixel-score starting points.
    n_balanced = max(1, topk // 2)
    n_pixel = topk - n_balanced
    balanced = vals.topk(n_balanced, dim=1).indices
    if n_pixel:
        available_pixel = pixels_all.clone()
        available_pixel.scatter_(1, balanced, -1.0)
        pixel_best = available_pixel.topk(n_pixel, dim=1).indices
        idx = torch.cat([balanced, pixel_best], dim=1)
    else:
        idx = balanced
    return latents[idx], vals.gather(1, idx)


def optimize(G, C, target, ft, rand_mse, rand_cos, init, steps, lr, batch, weight, coarse_weight=0.2):
    n, k, dim = init.shape
    best_z = torch.empty((n, dim), device=target.device)
    best_scores = torch.empty(n, device=target.device)
    best_pixel = torch.empty(n, device=target.device)
    best_feature = torch.empty(n, device=target.device)

    for start in range(0, n, batch):
        stop = min(start + batch, n)
        count = stop - start
        z = torch.nn.Parameter(init[start:stop].reshape(count * k, dim).clone())
        optimizer = torch.optim.Adam([z], lr=lr)
        # Different starts explore different pixel/identity trade-offs.
        weights = torch.linspace(max(0.05, weight * 0.6),
                                 min(0.50, weight * 2.5),
                                 steps=k, device=target.device).repeat(count)
        t = target[start:stop].repeat_interleave(k, dim=0)
        f = ft[start:stop].repeat_interleave(k, dim=0)
        rm = rand_mse[start:stop].repeat_interleave(k)
        rc = rand_cos[start:stop].repeat_interleave(k)
        local_best = torch.full((count,), -1., device=target.device)
        local_z = z.detach().reshape(count, k, dim)[:, 0].clone()
        local_pixel = torch.zeros(count, device=target.device)
        local_feature = torch.zeros(count, device=target.device)

        for step in range(steps + 1):
            x = G(z)
            fx = normalized_features(C, x)
            val, pixel, feature, mse, cosine = scores(x, t, fx, f, rm, rc)
            with torch.no_grad():
                found, idx = val.reshape(count, k).max(1)
                keep = found > local_best
                rows = torch.arange(count, device=target.device)
                local_z[keep] = z.detach().reshape(count, k, dim)[rows, idx][keep]
                local_pixel[keep] = pixel.detach().reshape(count, k)[rows, idx][keep]
                local_feature[keep] = feature.detach().reshape(count, k)[rows, idx][keep]
                local_best[keep] = found[keep]
            if step == steps:
                break
            # Differentiable, unclipped normalized surrogate of the scoring rule.
            coarse = (F.avg_pool2d(x, 4) - F.avg_pool2d(t, 4)).square().mean((1, 2, 3)) / rm
            loss = ((1 - weights) * (mse / rm + coarse_weight * coarse) +
                    weights * (1 - cosine) / (1 - rc).clamp_min(1e-6)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            project_norm_(z)

        best_z[start:stop] = local_z
        best_scores[start:stop] = local_best
        best_pixel[start:stop] = local_pixel
        best_feature[start:stop] = local_feature
        print(f"{stop}/{n}: local score {best_scores[:stop].mean().item()*100:.2f}", flush=True)

    return best_z, best_scores, best_pixel, best_feature


@torch.no_grad()
def evaluate_existing(G, C, target, ft, rand_mse, rand_cos, z, batch=20):
    """Compute the same reference scores as score.py for an existing submission."""
    values, pixels, features = [], [], []
    for start in range(0, len(z), batch):
        stop = min(start + batch, len(z))
        x = G(z[start:stop])
        fx = normalized_features(C, x)
        value, pixel, feature, _, _ = scores(
            x, target[start:stop], fx, ft[start:stop],
            rand_mse[start:stop], rand_cos[start:stop]
        )
        values.append(value)
        pixels.append(pixel)
        features.append(feature)
    return torch.cat(values), torch.cat(pixels), torch.cat(features)


def refine_pixels(G, C, target, ft, rand_mse, rand_cos,
                  base_z, base_scores, base_pixel, base_feature,
                  steps, lr, feature_weight, batch_size,
                  restarts=2, noise=0.03, max_feature_drop=0.02,
                  coarse_weight=0.2, improve="pixel"):
    """Refine latent vectors under guardrails for pixel or total improvements.

    The original candidate is kept unless a candidate improves pixel score,
    preserves the starting combined public score and keeps feature score
    within max_feature_drop of its starting value.
    """
    n, dim = base_z.shape
    refined_z = base_z.clone()
    chosen_score = base_scores.clone()
    chosen_pixel = base_pixel.clone()
    chosen_feature = base_feature.clone()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        b = stop - start
        init = base_z[start:stop, None, :].repeat(1, restarts, 1)
        if restarts > 1 and noise > 0:
            init[:, 1:] += torch.randn_like(init[:, 1:]) * noise
        z = torch.nn.Parameter(init.reshape(b * restarts, dim).clone())
        project_norm_(z)
        optimizer = torch.optim.Adam([z], lr=lr)
        t = target[start:stop].repeat_interleave(restarts, 0)
        f = ft[start:stop].repeat_interleave(restarts, 0)
        rm = rand_mse[start:stop].repeat_interleave(restarts)
        rc = rand_cos[start:stop].repeat_interleave(restarts)
        baseline_total = base_scores[start:stop, None]
        feature_floor = base_feature[start:stop, None] - max_feature_drop

        for iteration in range(steps + 1):
            x = G(z)
            fx = normalized_features(C, x)
            total, pixel, feature, mse, cosine = scores(x, t, fx, f, rm, rc)
            with torch.no_grad():
                totals = total.detach().reshape(b, restarts)
                pixels = pixel.detach().reshape(b, restarts)
                features = feature.detach().reshape(b, restarts)
                if improve == "pixel":
                    eligible = (
                        (pixels > chosen_pixel[start:stop, None] + 1e-6)
                        & (features >= feature_floor)
                        & (totals + 1e-7 >= baseline_total)
                    )
                    criterion = pixels
                else:
                    # Hard targets: allow only a negligible pixel sacrifice.
                    eligible = (
                        (totals > chosen_score[start:stop, None] + 1e-6)
                        & (pixels >= base_pixel[start:stop, None] - 0.003)
                        & (features >= feature_floor)
                    )
                    criterion = totals
                candidates = criterion.masked_fill(~eligible, -1.0)
                val, index = candidates.max(dim=1)
                keep = val >= 0
                row = torch.arange(b, device=z.device)
                chosen_z = z.detach().reshape(b, restarts, dim)[row, index]
                refined_z[start:stop][keep] = chosen_z[keep]
                chosen_score[start:stop][keep] = totals[row, index][keep]
                chosen_pixel[start:stop][keep] = pixels[row, index][keep]
                chosen_feature[start:stop][keep] = features[row, index][keep]

            if iteration == steps:
                break
            pix_loss = mse / rm
            coarse_loss = (F.avg_pool2d(x, 4) - F.avg_pool2d(t, 4)).square().mean((1, 2, 3)) / rm
            feature_loss = (1 - cosine) / (1 - rc).clamp_min(1e-6)
            loss = ((1 - feature_weight) * (pix_loss + coarse_weight * coarse_loss) +
                    feature_weight * feature_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            project_norm_(z)

        changed = int((chosen_pixel[start:stop] > base_pixel[start:stop] + 1e-6).sum())
        print(f"Pixel stage {stop}/{n}, improved {changed}/{b}, "
              f"pixel={chosen_pixel[:stop].mean().item():.4f}, "
              f"feature={chosen_feature[:stop].mean().item():.4f}", flush=True)

    return refined_z, chosen_score, chosen_pixel, chosen_feature


@torch.no_grad()
def save_comparisons(G, targets, z, values):
    faces = torch.cat([G(z[i:i + 20]) for i in range(0, len(z), 20)])
    rows = []
    for i in range(0, min(40, len(z)), 10):
        rows.extend([targets[i:i + 10], faces[i:i + 10]])
    save_image((torch.cat(rows).cpu() + 1) / 2, "compare.png", nrow=10)
    worst = values.argsort()[:min(10, len(z))]
    save_image((torch.cat([targets[worst], faces[worst]]).cpu() + 1) / 2,
               "worst.png", nrow=len(worst))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="targets200.npy")
    p.add_argument("--generator", default="G.pt")
    p.add_argument("--classifier", default="classifier.pt")
    p.add_argument("--out", default="z.pt")
    p.add_argument("--resume_z", default=None,
                   help="Existing z.pt; skip initial search and stage-1 optimization")
    p.add_argument("--n_random", type=int, default=3000)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.035)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--feature_weight", type=float, default=0.20)
    p.add_argument("--coarse_weight", type=float, default=0.20)
    p.add_argument("--pixel_steps", type=int, default=80)
    p.add_argument("--pixel_lr", type=float, default=0.012)
    p.add_argument("--pixel_feature_weight", type=float, default=0.18)
    p.add_argument("--pixel_restarts", type=int, default=3)
    p.add_argument("--pixel_noise", type=float, default=0.06)
    p.add_argument("--hard_fraction", type=float, default=0.25)
    p.add_argument("--hard_steps", type=int, default=60)
    p.add_argument("--hard_noise", type=float, default=0.16)
    p.add_argument("--max_feature_drop", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    if a.topk < 1 or a.n_random < a.topk or a.steps < 0 or a.batch_size < 1:
        p.error("Require n_random >= topk >= 1, steps >= 0, batch_size >= 1")
    if not (0 <= a.feature_weight <= 1) or a.lr <= 0:
        p.error("Require 0 <= feature_weight <= 1 and lr > 0")
    if a.coarse_weight < 0:
        p.error("coarse_weight must be nonnegative")
    if not (0 <= a.hard_fraction <= 1) or a.hard_steps < 0 or a.hard_noise < 0:
        p.error("Invalid hard-target optimization arguments")
    if (a.pixel_steps < 0 or a.pixel_lr <= 0 or a.pixel_restarts < 1 or
            a.pixel_noise < 0 or a.max_feature_drop < 0 or
            not (0 <= a.pixel_feature_weight <= 1)):
        p.error("Invalid pixel refinement arguments")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.manual_seed(a.seed)
    started = time.time()
    G = load_generator(a.generator).to(device).eval()
    C = load_classifier(a.classifier).to(device).eval()
    for model in (G, C):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    raw = np.load(a.targets)
    if raw.ndim != 4 or raw.shape[1:] != (64, 64, 3):
        raise ValueError(f"Unexpected target shape: {raw.shape}")
    target = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(device)
    with torch.no_grad():
        ft = torch.cat([normalized_features(C, target[i:i + 50])
                        for i in range(0, len(target), 50)])
    rand_mse, rand_cos = reference_stats(G, C, target, ft)
    if a.resume_z:
        z = torch.load(a.resume_z, map_location="cpu", weights_only=True)
        if not isinstance(z, torch.Tensor) or tuple(z.shape) != (len(target), 128):
            raise ValueError("resume_z must have shape (N, 128)")
        z = z.to(device=device, dtype=torch.float32)
        if not torch.isfinite(z).all():
            raise ValueError("resume_z contains NaN/inf")
        if not ((z.norm(dim=1) >= 8.5) & (z.norm(dim=1) <= 14.1)).all():
            raise ValueError("resume_z violates z norm limits")
        values, pix, feat = evaluate_existing(G, C, target, ft, rand_mse, rand_cos, z)
        print(f"Loaded previous z: local={values.mean().item()*100:.2f}, "
              f"pixel={pix.mean().item():.4f}, feature={feat.mean().item():.4f}", flush=True)
    else:
        init, init_vals = random_start(G, C, target, ft, rand_mse, rand_cos,
                                      a.n_random, a.topk)
        print(f"Initial reference score {init_vals[:, 0].mean().item()*100:.2f}", flush=True)
        z, values, pix, feat = optimize(
            G, C, target, ft, rand_mse, rand_cos, init,
            a.steps, a.lr, a.batch_size, a.feature_weight,
            coarse_weight=a.coarse_weight
        )
        print(f"After stage 1: local={values.mean().item()*100:.2f}, "
              f"pixel={pix.mean().item():.4f}, feature={feat.mean().item():.4f}", flush=True)

    if a.pixel_steps > 0:
        old_total, old_pix, old_feat = values.clone(), pix.clone(), feat.clone()
        torch.save(z.detach().cpu().float(), "z_before_pixel.pt")
        z, values, pix, feat = refine_pixels(
            G, C, target, ft, rand_mse, rand_cos, z, values, pix, feat,
            steps=a.pixel_steps, lr=a.pixel_lr,
            feature_weight=a.pixel_feature_weight, batch_size=a.batch_size,
            restarts=a.pixel_restarts, noise=a.pixel_noise,
            max_feature_drop=a.max_feature_drop,
            coarse_weight=a.coarse_weight
        )
        print(f"Pixel: {old_pix.mean().item():.4f} -> {pix.mean().item():.4f}")
        print(f"Feature: {old_feat.mean().item():.4f} -> {feat.mean().item():.4f}")
        print(f"Local score: {old_total.mean().item()*100:.2f} -> {values.mean().item()*100:.2f}")
        print(f"Improved pixel for {(pix > old_pix + 1e-6).sum().item()}/{len(pix)} images")
    if a.hard_steps > 0 and a.hard_fraction > 0:
        hard_n = max(1, round(len(target) * a.hard_fraction))
        hard_idx = values.argsort()[:hard_n]
        before_hard = values[hard_idx].clone()
        hard_z, hard_s, hard_p, hard_f = refine_pixels(
            G, C, target[hard_idx], ft[hard_idx],
            rand_mse[hard_idx], rand_cos[hard_idx],
            z[hard_idx], values[hard_idx], pix[hard_idx], feat[hard_idx],
            steps=a.hard_steps, lr=0.012, feature_weight=0.26,
            batch_size=a.batch_size, restarts=3, noise=a.hard_noise,
            max_feature_drop=0.015, coarse_weight=a.coarse_weight,
            improve="score"
        )
        z[hard_idx] = hard_z
        values[hard_idx] = hard_s
        pix[hard_idx] = hard_p
        feat[hard_idx] = hard_f
        print(f"Hard-target local score: {before_hard.mean().item()*100:.2f} -> "
              f"{hard_s.mean().item()*100:.2f}", flush=True)

    result = z.detach().cpu().float()
    if result.shape != (len(target), 128) or not torch.isfinite(result).all():
        raise ValueError("Invalid submission tensor")
    norm = result.norm(dim=1)
    if not ((norm >= 8.5) & (norm <= 14.1)).all():
        raise ValueError("One or more z norms violate score.py")
    torch.save(result, a.out)
    save_comparisons(G, target, z, values)
    print(f"Saved {a.out}: {tuple(result.shape)}")
    print(f"Local reference {values.mean().item()*100:.2f}/100 "
          f"(pixel={pix.mean().item():.3f}, feature={feat.mean().item():.3f})")
    print(f"Elapsed: {time.time()-started:.1f} seconds")
    print(f"Verify with: python score.py --z {a.out}")


if __name__ == "__main__":
    main()
