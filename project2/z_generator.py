"""Optimize the 200 latent vectors, not the supplied GAN or classifier weights."""
import argparse
import time
from pathlib import Path
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


def refine_semantics(G, C, target, ft, rand_mse, rand_cos,
                     original_z, original_total, original_pixel, original_feature,
                     steps, lr, batch_size, restarts=3, attr_weight=0.20,
                     noise=0.12):
    """Conservative 20-attribute refinement with strict score safeguards.

    Attribute probabilities come from the supplied classifier's 20 outputs.
    Accept candidate only if normalized pixel/feature scores and public
    reference score remain within a narrow bound and attributes improve.
    This does NOT guarantee improved hidden-classifier score.
    """
    n, dim = original_z.shape
    result_z = original_z.clone()
    result_total = original_total.clone()
    result_pixel = original_pixel.clone()
    result_feature = original_feature.clone()
    gain_count = 0
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        count = end - start
        t0 = target[start:end]
        with torch.no_grad():
            t_attr = torch.sigmoid(C(t0))
            initial_attr = torch.sigmoid(C(G(original_z[start:end])))
            best_attr = (initial_attr - t_attr).square().mean(dim=1)
        best_z = result_z[start:end].clone()
        best_total = result_total[start:end].clone()
        best_pixel = result_pixel[start:end].clone()
        best_feature = result_feature[start:end].clone()

        init = original_z[start:end, None].repeat(1, restarts, 1)
        if restarts > 1 and noise > 0:
            init[:, 1:] += torch.randn_like(init[:, 1:]) * noise
        z = torch.nn.Parameter(init.reshape(count * restarts, dim).clone())
        project_norm_(z)
        optimizer = torch.optim.Adam([z], lr=lr)
        t = t0.repeat_interleave(restarts, 0)
        f = ft[start:end].repeat_interleave(restarts, 0)
        rm = rand_mse[start:end].repeat_interleave(restarts)
        rc = rand_cos[start:end].repeat_interleave(restarts)
        attr_target = t_attr.repeat_interleave(restarts, 0)

        for step in range(steps + 1):
            x = G(z)
            raw_features = C.features(x)
            fx = F.normalize(raw_features, dim=1)
            total, pixel, feature, mse, cosine = scores(
                x, t, fx, f, rm, rc)
            attr_loss = (
                torch.sigmoid(C.fc(raw_features)) - attr_target
            ).square().mean(dim=1)
            with torch.no_grad():
                totals = total.detach().reshape(count, restarts)
                pixels = pixel.detach().reshape(count, restarts)
                feats = feature.detach().reshape(count, restarts)
                attrs = attr_loss.detach().reshape(count, restarts)
                good = (
                    (totals >= best_total[:, None] + 1e-5)
                    & (pixels >= original_pixel[start:end, None] - 0.003)
                    & (feats >= original_feature[start:end, None] - 0.010)
                    & (attrs < best_attr[:, None] - 1e-4)
                )
                merit = (totals + 0.10 * (best_attr[:, None] - attrs))
                merit = merit.masked_fill(~good, -1.0)
                cand, idx = merit.max(dim=1)
                accepted = cand >= 0
                row = torch.arange(count, device=z.device)
                candidate_z = z.detach().reshape(count, restarts, dim)[row, idx]
                best_z[accepted] = candidate_z[accepted]
                best_total[accepted] = totals[row, idx][accepted]
                best_pixel[accepted] = pixels[row, idx][accepted]
                best_feature[accepted] = feats[row, idx][accepted]
                best_attr[accepted] = attrs[row, idx][accepted]

            if step == steps:
                break
            pixel_loss = mse / rm
            feature_loss = (1 - cosine) / (1 - rc).clamp_min(1e-6)
            coarse = (F.avg_pool2d(x, 4) - F.avg_pool2d(t, 4)).square().mean((1, 2, 3)) / rm
            loss = (0.65 * pixel_loss + 0.30 * feature_loss
                    + 0.10 * coarse + attr_weight * attr_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            project_norm_(z)

        result_z[start:end] = best_z
        result_total[start:end] = best_total
        result_pixel[start:end] = best_pixel
        result_feature[start:end] = best_feature
        changed = (best_total > original_total[start:end] + 1e-5).sum().item()
        gain_count += int(changed)
        print(f"Attribute refinement {end}/{n}: {changed}/{count} improved", flush=True)
    print(f"Accepted attribute refinements: {gain_count}/{n}", flush=True)
    return result_z, result_total, result_pixel, result_feature


def classifier_spatial_maps(C, x):
    """Three spatial representations from the SUPPLIED, unchanged classifier."""
    h = F.max_pool2d(F.relu(C.bn1(C.conv1(x))), 2)
    h = F.max_pool2d(F.relu(C.bn2(C.conv2(h))), 2)
    middle = h
    h = F.max_pool2d(F.relu(C.bn3(C.conv3(h))), 2)
    deep = h
    h = F.max_pool2d(F.relu(C.bn4(C.conv4(h))), 2)
    return (middle, deep, h)


def spatial_distance(maps, target_maps):
    """Mean location-aware cosine distance across intermediate feature grids."""
    distances = []
    for candidate, target in zip(maps, target_maps):
        a = F.normalize(candidate, dim=1, eps=1e-6)
        b = F.normalize(target, dim=1, eps=1e-6)
        distances.append((1 - (a * b).sum(dim=1)).mean(dim=(1, 2)))
    return sum(distances) / len(distances)


def refine_spatial_identity(G, C, target, ft, rand_mse, rand_cos,
                            base_z, base_score, base_pixel, base_feature,
                            steps, lr, batch_size, restarts=2, noise=0.05,
                            spatial_weight=0.12, attribute_weight=0.15):
    """Optional last-stage identity refinement.

    Only retain candidates that improve BOTH public reference total score
    and intermediate spatial agreement, while protecting pixel/feature scores.
    Using C's early maps can encourage pose/geometry agreement, but it is NOT
    a validation against the server's different, hidden classifier.
    """
    result_z = base_z.clone()
    result_score = base_score.clone()
    result_pixel = base_pixel.clone()
    result_feature = base_feature.clone()
    n, dim = base_z.shape
    accepted_count = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        count = end - start
        t0 = target[start:end]
        with torch.no_grad():
            target_maps = tuple(f.detach() for f in classifier_spatial_maps(C, t0))
            target_raw = F.adaptive_avg_pool2d(target_maps[-1], 1).flatten(1)
            target_attrs = torch.sigmoid(C.fc(target_raw))
            baseline_maps = classifier_spatial_maps(C, G(base_z[start:end]))
            best_spatial = spatial_distance(baseline_maps, target_maps)

        best_z = result_z[start:end].clone()
        best_score = result_score[start:end].clone()
        best_pixel = result_pixel[start:end].clone()
        best_feature = result_feature[start:end].clone()

        init = base_z[start:end, None, :].repeat(1, restarts, 1)
        if restarts > 1 and noise:
            init[:, 1:, :] += torch.randn_like(init[:, 1:, :]) * noise
        z = torch.nn.Parameter(init.reshape(count * restarts, dim).clone())
        project_norm_(z)
        optimizer = torch.optim.Adam([z], lr=lr)

        t = t0.repeat_interleave(restarts, 0)
        f = ft[start:end].repeat_interleave(restarts, 0)
        rm = rand_mse[start:end].repeat_interleave(restarts)
        rc = rand_cos[start:end].repeat_interleave(restarts)
        t_maps = tuple(u.repeat_interleave(restarts, 0) for u in target_maps)
        t_attrs = target_attrs.repeat_interleave(restarts, 0)
        rows = torch.arange(count, device=z.device)

        for iteration in range(steps + 1):
            x = G(z)
            maps = classifier_spatial_maps(C, x)
            raw_features = F.adaptive_avg_pool2d(maps[-1], 1).flatten(1)
            fx = F.normalize(raw_features, dim=1)
            total, pixel, feature, mse, cosine = scores(x, t, fx, f, rm, rc)
            spatial = spatial_distance(maps, t_maps)
            attr_loss = (torch.sigmoid(C.fc(raw_features)) - t_attrs).square().mean(1)

            with torch.no_grad():
                total_v = total.detach().reshape(count, restarts)
                pixel_v = pixel.detach().reshape(count, restarts)
                feature_v = feature.detach().reshape(count, restarts)
                spatial_v = spatial.detach().reshape(count, restarts)

                eligible = (
                    (total_v > best_score[:, None] + 2e-5)
                    & (pixel_v >= base_pixel[start:end, None] - 0.003)
                    & (feature_v >= base_feature[start:end, None] - 0.003)
                    & (spatial_v < best_spatial[:, None] - 0.0005)
                )
                # Prioritize score, with spatial agreement as a tie-breaker.
                merit = total_v + 0.03 * (best_spatial[:, None] - spatial_v)
                merit = merit.masked_fill(~eligible, -1.0)
                candidate_merit, indexes = merit.max(dim=1)
                accepted = candidate_merit >= 0
                latents = z.detach().reshape(count, restarts, dim)[rows, indexes]
                best_z[accepted] = latents[accepted]
                best_score[accepted] = total_v[rows, indexes][accepted]
                best_pixel[accepted] = pixel_v[rows, indexes][accepted]
                best_feature[accepted] = feature_v[rows, indexes][accepted]
                best_spatial[accepted] = spatial_v[rows, indexes][accepted]

            if iteration == steps:
                break
            pix_loss = mse / rm
            feat_loss = (1 - cosine) / (1 - rc).clamp_min(1e-6)
            coarse = (F.avg_pool2d(x, 4) - F.avg_pool2d(t, 4)).square().mean((1, 2, 3)) / rm
            loss = (
                0.65 * pix_loss + 0.30 * feat_loss + 0.10 * coarse
                + spatial_weight * spatial + attribute_weight * attr_loss
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            project_norm_(z)

        result_z[start:end] = best_z
        result_score[start:end] = best_score
        result_pixel[start:end] = best_pixel
        result_feature[start:end] = best_feature
        changed = int((best_score > base_score[start:end] + 2e-5).sum().item())
        accepted_count += changed
        print(f"Spatial identity {end}/{n}: {changed}/{count} changed", flush=True)

    print(f"Spatial improvements accepted: {accepted_count}/{n}", flush=True)
    return result_z, result_score, result_pixel, result_feature


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
    p.add_argument("--pixel_steps", type=int, default=100)
    p.add_argument("--pixel_lr", type=float, default=0.012)
    p.add_argument("--pixel_feature_weight", type=float, default=0.18)
    p.add_argument("--pixel_restarts", type=int, default=3)
    p.add_argument("--pixel_noise", type=float, default=0.06)
    p.add_argument("--hard_fraction", type=float, default=0.35)
    p.add_argument("--hard_steps", type=int, default=90)
    p.add_argument("--hard_noise", type=float, default=0.20)
    p.add_argument("--semantic_steps", type=int, default=55)
    p.add_argument("--semantic_fraction", type=float, default=0.35)
    p.add_argument("--semantic_restarts", type=int, default=3)
    p.add_argument("--semantic_lr", type=float, default=0.012)
    p.add_argument("--semantic_attr_weight", type=float, default=0.20)
    p.add_argument("--spatial_steps", type=int, default=50)
    p.add_argument("--spatial_fraction", type=float, default=0.65)
    p.add_argument("--spatial_lr", type=float, default=0.010)
    p.add_argument("--spatial_restarts", type=int, default=2)
    p.add_argument("--spatial_weight", type=float, default=0.12)
    p.add_argument("--spatial_attr_weight", type=float, default=0.15)
    p.add_argument("--max_feature_drop", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    # Search five seeds in one run; only ONE z.pt will be saved.
    p.add_argument("--seeds", default=None, help="Optional comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--extra_random", type=int, default=2200)
    p.add_argument("--extra_topk", type=int, default=4)
    p.add_argument("--extra_steps", type=int, default=95)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    if a.topk < 1 or a.n_random < a.topk or a.steps < 0 or a.batch_size < 1:
        p.error("Require n_random >= topk >= 1, steps >= 0, batch_size >= 1")
    if not (0 <= a.feature_weight <= 1) or a.lr <= 0:
        p.error("Require 0 <= feature_weight <= 1 and lr > 0")
    if a.coarse_weight < 0:
        p.error("coarse_weight must be nonnegative")
    if a.extra_random < a.extra_topk or a.extra_topk < 1 or a.extra_steps < 0:
        p.error("Require extra_random >= extra_topk >= 1 and extra_steps >= 0")
    try:
        seeds = ([int(part.strip()) for part in a.seeds.split(",")]
                 if a.seeds is not None else [a.seed + i for i in range(5)])
        if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be unique nonnegative integers")
    except ValueError:
        p.error("--seeds must contain unique nonnegative integers, e.g. 0,1,2")
    if (not 0 <= a.spatial_fraction <= 1 or a.spatial_steps < 0
            or a.spatial_lr <= 0 or a.spatial_restarts < 1
            or a.spatial_weight < 0 or a.spatial_attr_weight < 0):
        p.error("Invalid spatial refinement arguments")
    if (not 0 <= a.semantic_fraction <= 1 or a.semantic_steps < 0
            or a.semantic_restarts < 1 or a.semantic_lr <= 0
            or a.semantic_attr_weight < 0):
        p.error("Invalid semantic refinement arguments")
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
        # Each seed is independent, but the best vector is retained PER IMAGE.
        # Seed 0 takes the full V4 optimization path.
        best_z = best_score = best_pixel = best_feature = None
        source_seed = torch.full((len(target),), -1, dtype=torch.long, device=device)
        # If a previous z.pt exists, keep it as a candidate for every face.
        # No extra .pt file is written.
        if Path(a.out).is_file():
            try:
                old = torch.load(a.out, map_location="cpu", weights_only=True)
                if (not isinstance(old, torch.Tensor)
                        or tuple(old.shape) != (len(target), 128)
                        or not torch.isfinite(old).all()
                        or not ((old.norm(dim=1) >= 8.5)
                                & (old.norm(dim=1) <= 14.1)).all()):
                    raise ValueError("invalid shape, values or norm")
                old = old.to(device=device, dtype=torch.float32)
                best_score, best_pixel, best_feature = evaluate_existing(
                    G, C, target, ft, rand_mse, rand_cos, old)
                best_z = old.clone()
                print(f"Existing z.pt candidate: "
                      f"{best_score.mean().item()*100:.2f}/100", flush=True)
            except Exception as exc:
                print(f"Skipping invalid previous z.pt: {exc}", flush=True)
        for run_idx, seed in enumerate(seeds):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            full_run = run_idx == 0
            num_random = a.n_random if full_run else a.extra_random
            topk = a.topk if full_run else a.extra_topk
            steps = a.steps if full_run else a.extra_steps
            print(f"Seed {seed} ({run_idx + 1}/{len(seeds)}): "
                  f"random={num_random}, topk={topk}, Adam steps={steps}", flush=True)
            init, init_vals = random_start(
                G, C, target, ft, rand_mse, rand_cos, num_random, topk
            )
            zz, vv, pp, ff = optimize(
                G, C, target, ft, rand_mse, rand_cos, init,
                steps, a.lr, a.batch_size, a.feature_weight,
                coarse_weight=a.coarse_weight
            )
            if best_z is None:
                best_z, best_score, best_pixel, best_feature = (
                    zz.clone(), vv.clone(), pp.clone(), ff.clone()
                )
                source_seed.fill_(seed)
                print(f"Seed {seed}: initial full set local "
                      f"{best_score.mean().item() * 100:.2f}", flush=True)
            else:
                # Public classifier may not match the leaderboard. Do not
                # exchange large pixel/feature losses for tiny local gains.
                keep = (
                    (vv > best_score + 1e-6)
                    & (pp >= best_pixel - 0.002)
                    & (ff >= best_feature - 0.010)
                )
                count = int(keep.sum().item())
                best_z[keep] = zz[keep]
                best_score[keep] = vv[keep]
                best_pixel[keep] = pp[keep]
                best_feature[keep] = ff[keep]
                source_seed[keep] = seed
                print(f"Seed {seed}: improved {count}/{len(target)} targets; "
                      f"merged local={best_score.mean().item()*100:.2f}, "
                      f"pixel={best_pixel.mean().item():.4f}, "
                      f"feature={best_feature.mean().item():.4f}", flush=True)
        z, values, pix, feat = best_z, best_score, best_pixel, best_feature
        print("Selected seed counts:", {
            **{"existing": int((source_seed == -1).sum().item())},
            **{seed: int((source_seed == seed).sum().item()) for seed in seeds}
        }, flush=True)
        print(f"After multi-seed search: local={values.mean().item()*100:.2f}, "
              f"pixel={pix.mean().item():.4f}, feature={feat.mean().item():.4f}", flush=True)

    if a.pixel_steps > 0:
        old_total, old_pix, old_feat = values.clone(), pix.clone(), feat.clone()
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

    if a.semantic_steps > 0 and a.semantic_fraction > 0:
        # Focus additional attribute matching on weak local faces.
        attr_n = max(1, round(len(target) * a.semantic_fraction))
        sem_idx = values.argsort()[:attr_n]
        z_new, s_new, p_new, f_new = refine_semantics(
            G, C, target[sem_idx], ft[sem_idx],
            rand_mse[sem_idx], rand_cos[sem_idx],
            z[sem_idx], values[sem_idx], pix[sem_idx], feat[sem_idx],
            steps=a.semantic_steps, lr=a.semantic_lr,
            batch_size=a.batch_size, restarts=a.semantic_restarts,
            attr_weight=a.semantic_attr_weight, noise=0.12
        )
        z[sem_idx] = z_new
        values[sem_idx] = s_new
        pix[sem_idx] = p_new
        feat[sem_idx] = f_new
        print(f"After semantic stage: local={values.mean().item()*100:.2f}, "
              f"pixel={pix.mean().item():.4f}, "
              f"feature={feat.mean().item():.4f}", flush=True)

    if a.spatial_steps > 0 and a.spatial_fraction > 0:
        spatial_n = max(1, round(len(target) * a.spatial_fraction))
        # Score is public, so this is only a heuristic for difficult images.
        spatial_idx = values.argsort()[:spatial_n]
        old_total = values.mean().item() * 100
        old_pix = pix.mean().item()
        old_feat = feat.mean().item()
        zz, ss, pp, ff = refine_spatial_identity(
            G, C, target[spatial_idx], ft[spatial_idx],
            rand_mse[spatial_idx], rand_cos[spatial_idx],
            z[spatial_idx], values[spatial_idx], pix[spatial_idx], feat[spatial_idx],
            steps=a.spatial_steps, lr=a.spatial_lr,
            batch_size=a.batch_size, restarts=a.spatial_restarts,
            spatial_weight=a.spatial_weight,
            attribute_weight=a.spatial_attr_weight
        )
        z[spatial_idx] = zz
        values[spatial_idx] = ss
        pix[spatial_idx] = pp
        feat[spatial_idx] = ff
        print(f"Spatial stage: local {old_total:.2f} -> "
              f"{values.mean().item()*100:.2f}, "
              f"pixel {old_pix:.4f} -> {pix.mean().item():.4f}, "
              f"feature {old_feat:.4f} -> {feat.mean().item():.4f}", flush=True)

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
