"""Train from scratch, select by mean Macro-F1, and retain matching submission code."""
import argparse
import copy
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import PlantDataset, get_transforms, load_class_names, grouped_folds, fold_indices


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2 ** 32))


def macro_f1(conf):
    true_positive = conf.diag().float()
    precision = true_positive / conf.sum(0).clamp(min=1)
    recall = true_positive / conf.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return f1.mean().item(), f1.tolist()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    matrices = [torch.zeros(3, 3, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long)]
    for images, crops, healths in loader:
        outputs = model(images.to(device, non_blocking=True))
        for matrix, labels, logits, classes in zip(matrices, (crops, healths), outputs, (3, 2)):
            if logits.shape != (len(images), classes) or not torch.isfinite(logits).all():
                raise ValueError("Invalid output shape or non-finite validation logits")
            encoded = labels * classes + logits.argmax(1).cpu()
            matrix += torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
    result = {}
    for key, matrix in zip(("crop", "health"), matrices):
        f1, per_class = macro_f1(matrix)
        result.update({
            key + "_f1": f1, key + "_f1_per_class": per_class,
            key + "_accuracy": (matrix.diag().sum() / matrix.sum()).item(),
            key + "_confusion": matrix.tolist(),
        })
    result["score"] = (result["crop_f1"] + result["health_f1"]) / 2
    return result


def print_report(result, names):
    for key, labels in zip(("crop", "health"), names):
        print(f"{key}: Macro-F1 {result[key + '_f1']:.4f}")
        for label, value in zip(labels, result[key + "_f1_per_class"]):
            print(f"  {label}: {value:.4f}")
        print("  confusion (true rows, predicted columns):", result[key + "_confusion"])
    print(f"Mean Macro-F1: {result['score']:.4f}")


def class_weights(rows, key, classes, power, device):
    counts = torch.bincount(torch.tensor([int(row[key]) for row in rows]), minlength=classes).float()
    weights = counts.clamp(min=1).pow(-power)
    weights /= weights.mean()
    print(f"{key}: counts {counts.int().tolist()}, loss weights {weights.tolist()}")
    return weights.to(device)


def configure_regularization(model, dropout, drop_path):
    blocks = [module for module in model.modules() if hasattr(module, "drop_path")]
    for index, block in enumerate(blocks):
        block.drop_path = drop_path * (index + 1) / len(blocks)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout


def load_initial_weights(model, state):
    """Accept only a matching model or the immediately preceding deep architecture."""
    allowed = {"detail_projection.weight"}
    expected = model.state_dict()
    missing, unexpected = expected.keys() - state.keys(), state.keys() - expected.keys()
    if unexpected or missing - allowed:
        raise ValueError("Starting weights do not match this deep model; use your own deep-run best_model.pt")
    model.load_state_dict(state, strict=not missing)
    if missing:
        with torch.no_grad():
            model.detail_projection.weight.zero_()


def mix_batch(images, targets, alpha):
    fraction = torch.distributions.Beta(alpha, alpha).sample().item()
    fraction = max(fraction, 1 - fraction)
    order = torch.randperm(len(images), device=images.device)
    return (images * fraction + images[order] * (1 - fraction),
            tuple(target[order] for target in targets), fraction)


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    """Use only clean training images; leave dropout/stochastic depth disabled."""
    modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    momenta = [m.momentum for m in modules]
    model.eval()
    try:
        for module in modules:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        for images, *_ in loader:
            model(images.to(device, non_blocking=True))
    finally:
        for module, momentum in zip(modules, momenta):
            module.momentum = momentum
        model.eval()


def average_states(left, right, left_count):
    if left.keys() != right.keys():
        raise ValueError("Cannot average different architectures")
    return {key: value + (right[key].to(value) - value) / (left_count + 1)
            if value.is_floating_point() else value.clone() for key, value in left.items()}


def save_candidate(model, result, path, epoch, variant, smoke):
    atomic_save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
    if path.stat().st_size > 10_000_000:
        raise ValueError("Checkpoint exceeds 10 MB")
    write_json({"epoch": epoch, "variant": variant, "weights_bytes": path.stat().st_size,
                "smoke_run": smoke, **result}, path.parent / "best_metrics.json")


def clean_training_loader(args):
    dataset = PlantDataset(args.data_dir, "train", get_transforms(False))
    if args.fold_manifest:
        indices, _ = fold_indices(dataset, args.fold_manifest, args.fold_index)
        dataset = Subset(dataset, indices)
    if args.train_limit:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[
            :args.train_limit].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=args.device == "cuda")


def validation_loader(args, device):
    dataset = PlantDataset(args.data_dir, "train" if args.fold_manifest else "valid", get_transforms(False))
    if args.fold_manifest:
        _, indices = fold_indices(dataset, args.fold_manifest, args.fold_index)
        dataset = Subset(dataset, indices)
    if args.valid_limit:
        dataset = Subset(dataset, range(min(args.valid_limit, len(dataset))))
    return dataset, DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                               pin_memory=device.type == "cuda")


class ModelEMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.lerp_(current.detach(), 1 - decay)
        # Keep BN statistics current; validate EMA separately before selecting it.
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="student_data_128")
    parser.add_argument("--run_dir", default=None,
                        help="New runs default to a unique timestamped folder under runs/")
    parser.add_argument("--model", choices=("deep",), default="deep",
                        help="model.py contains the deep CNN; no separate model file is required")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="Defaults to 0.0002 with own starting weights, otherwise 0.001")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--class_weight_power", type=float, default=0.35,
                        help="0 disables class weighting; only train labels determine weights")
    parser.add_argument("--crop_class_weight_power", type=float, default=0.5,
                        help="Override class_weight_power for the crop task only")
    parser.add_argument("--health_class_weight_power", type=float, default=None,
                        help="Override class_weight_power for the health task only")
    parser.add_argument("--crop_loss_weight", type=float, default=1.25)
    parser.add_argument("--health_loss_weight", type=float, default=1.0)
    parser.add_argument("--augmentation", choices=("standard", "gentle", "detail", "finish", "robust"),
                        default="gentle")
    parser.add_argument("--init_weights", default=None,
                        help="Start a NEW run from your own checkpoint trained only on the provided data")
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--ema_decay", type=float, default=0.99, help="0 disables EMA")
    parser.add_argument("--mixup_alpha", type=float, default=0.15)
    parser.add_argument("--mixup_prob", type=float, default=0.35)
    parser.add_argument("--finish_fraction", type=float, default=0.2,
                        help="Last fraction uses pixel-preserving transforms and hard labels")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--drop_path", type=float, default=0.03)
    parser.add_argument("--swa_start", type=float, default=0.7,
                        help="Average late checkpoints within one run; 1 disables SWA")
    parser.add_argument("--no_bn_recalibration", action="store_true")
    parser.add_argument("--search", action="store_true",
                        help="Three training recipes then validation-gated single-model weight averaging")
    parser.add_argument("--search_seeds", type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument("--cv", action="store_true",
                        help="Select recipe/epoch on grouped train folds, then refit without valid selection")
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--cv_epochs", type=int, default=100)
    parser.add_argument("--fold_seed", type=int, default=2026)
    parser.add_argument("--fold_manifest", default=None)
    parser.add_argument("--fold_index", type=int, default=-1)
    parser.add_argument("--fixed_epochs", action="store_true",
                        help="Freeze final EMA weights before official valid is evaluated; no checkpoint search")
    parser.add_argument("--fixed_variant", choices=("ema", "swa", "raw"), default="ema",
                        help="Predeclared fixed-refit weights; never selected on official valid")
    parser.add_argument("--disable_detail", action="store_true",
                        help="Freeze detail projection at zero to reproduce the preceding deep architecture")
    parser.add_argument("--compare_runs", nargs="+", default=None,
                        help="Read-only evaluation using each saved run's model.py and weights")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--train_limit", type=int, default=0, help="Smoke checks only, 0 uses all data")
    parser.add_argument("--valid_limit", type=int, default=0, help="Smoke checks only, 0 uses all data")
    args = parser.parse_args()
    if args.run_dir is None:
        if args.resume or args.eval_only:
            # Preserve the legacy default; timestamped runs should specify their saved path.
            args.run_dir = "runs/refined_seed42"
        else:
            mode = "cv" if args.cv else "search" if args.search else "audit" if args.compare_runs else "single"
            base = Path("runs") / f"{mode}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
            selected, suffix = base, 1
            while selected.exists():
                selected = base.with_name(f"{base.name}_{suffix}")
                suffix += 1
            args.run_dir = str(selected)
    args.data_dir = str(Path(args.data_dir).resolve())
    if args.init_weights:
        args.init_weights = str(Path(args.init_weights).resolve())
    if args.fold_manifest:
        args.fold_manifest = str(Path(args.fold_manifest).resolve())
    if args.lr is None:
        args.lr = 2e-4 if args.init_weights else 1e-3
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0 or not 0 <= args.min_lr <= args.lr:
        parser.error("Require epochs >= 1, batch_size >= 2 and 0 <= min_lr <= lr with lr > 0")
    if args.warmup_epochs < 0 or args.weight_decay < 0 or args.class_weight_power < 0:
        parser.error("Warmup, decay and class_weight_power cannot be negative")
    if any(power is not None and power < 0 for power in
           (args.crop_class_weight_power, args.health_class_weight_power)):
        parser.error("Task-specific class weight powers cannot be negative")
    if args.crop_loss_weight <= 0 or args.health_loss_weight <= 0:
        parser.error("Task loss weights must be positive")
    if not 0 <= args.ema_decay < 1 or not 0 <= args.label_smoothing < 1:
        parser.error("EMA decay and label smoothing must be in [0, 1)")
    if min(args.num_workers, args.train_limit, args.valid_limit) < 0 or args.num_threads < 1:
        parser.error("Invalid worker, thread or subset counts")
    if args.mixup_alpha < 0 or not 0 <= args.mixup_prob <= 1:
        parser.error("Require mixup_alpha >= 0 and mixup_prob in [0, 1]")
    if not 0 <= args.finish_fraction < 1 or not 0 <= args.swa_start <= 1:
        parser.error("Invalid finish_fraction or swa_start")
    if not 0 <= args.dropout < 1 or not 0 <= args.drop_path < 1:
        parser.error("Dropout and stochastic depth must be in [0, 1)")
    if args.search and args.eval_only:
        parser.error("Evaluate a search result without --search")
    if sum((args.search, args.cv, bool(args.compare_runs))) > 1:
        parser.error("Choose one of search, cv or compare_runs")
    if args.cv_folds < 2 or args.cv_epochs < 1 or args.bootstrap < 1:
        parser.error("Require cv_folds >= 2, cv_epochs >= 1, bootstrap >= 1")
    if (args.cv or args.fold_manifest or args.fixed_epochs) and args.init_weights:
        parser.error("Independent folds/refit must start from scratch; full-train weights have seen fold labels")
    if args.cv and (args.eval_only or args.fold_manifest or args.fixed_epochs):
        parser.error("cv orchestrates its own folds and fixed refit")
    if args.fold_manifest and args.fold_index < 0:
        parser.error("fold_manifest needs a non-negative fold_index")
    if args.fixed_epochs and args.fixed_variant == "swa" and (args.swa_start == 1 or args.no_bn_recalibration):
        parser.error("Fixed SWA requires late weight averaging and BN recalibration")
    return args


def main():
    args = parse_args()
    if args.compare_runs:
        return compare_runs(args)
    if args.cv:
        return run_cv(args)
    if args.search:
        return run_search(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; enable a Colab GPU or select --device cpu")
    run_dir = Path(args.run_dir).resolve()
    source_dir = Path(__file__).resolve().parent
    model_source = source_dir / "model.py"
    spec = importlib.util.spec_from_file_location("selected_leaf_model", model_source)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_dir / "best_model.pt", run_dir / "last_checkpoint.pt"
    if not args.resume and not args.eval_only and any(
        (run_dir / name).exists() for name in ("config.json", "best_model.pt", "last_checkpoint.pt")
    ):
        raise FileExistsError("Run exists: choose a new --run_dir or pass --resume")
    source_paths = {"model.py": model_source, "dataset.py": source_dir / "dataset.py",
                    "train.py": source_dir / "train.py"}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}
    fold_hash = hashlib.sha256(Path(args.fold_manifest).read_bytes()).hexdigest() if args.fold_manifest else None
    if args.resume or args.eval_only:
        existing = json.loads((run_dir / "config.json").read_text())
        if existing["source_sha256"] != hashes:
            raise ValueError("Source changed since this run; use the saved run sources or a new run")
        if not args.eval_only and existing.get("fold_manifest_sha256") != fold_hash:
            raise ValueError("Fold assignment changed since this run")
    else:
        for name, path in source_paths.items():
            shutil.copy2(path, run_dir / name)
        initial_hash = (hashlib.sha256(Path(args.init_weights).read_bytes()).hexdigest()
                        if args.init_weights else None)
        write_json({"args": vars(args), "source_sha256": hashes,
                    "fold_manifest_sha256": fold_hash,
                    "init_weights_sha256": initial_hash,
                    "torch": torch.__version__, "device": str(device),
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
                   run_dir / "config.json")
    print(f"Device: {device}; model: {args.model}; run: {run_dir}", flush=True)
    if args.fold_manifest:
        print(f"Validation source: held-out TRAIN fold {args.fold_index}; official valid not loaded.", flush=True)
    if args.train_limit or args.valid_limit:
        print("SMOKE RUN: subset metrics are not full validation or submission performance.")
    names = load_class_names(args.data_dir)
    if tuple(map(len, names)) != (3, 2):
        raise ValueError("Expected 3 crop classes and 2 health classes")
    valid_set, valid_loader = ((None, None) if args.fixed_epochs and not args.eval_only
                               else validation_loader(args, device))
    model = model_module.MultiTaskCNN().to(device)
    configure_regularization(model, args.dropout, args.drop_path)
    if args.disable_detail:
        model.detail_projection.weight.requires_grad_(False)
        with torch.no_grad():
            model.detail_projection.weight.zero_()
    count = sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for b in model.buffers() if b.is_floating_point())
    print(f"Parameters + floating buffers: {count:,} / 2,000,000")
    if count > 2_000_000:
        raise ValueError("Model exceeds submission size limit")
    if args.eval_only:
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        result = evaluate(model, valid_loader, device)
        print_report(result, names)
        write_json(result, run_dir / "evaluation.json")
        return

    initial_result = None
    if args.init_weights and not args.resume:
        load_initial_weights(model, torch.load(args.init_weights, map_location=device, weights_only=True))
        if args.disable_detail:
            with torch.no_grad():
                model.detail_projection.weight.zero_()
        initial_result = evaluate(model, valid_loader, device)
        print(f"Own-data starting checkpoint: mean Macro-F1 {initial_result['score']:.4f}", flush=True)
        atomic_save({key: value.detach().cpu() for key, value in model.state_dict().items()}, best_path)
        if best_path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Starting checkpoint exceeds 10 MiB")
        write_json({"epoch": 0, "variant": "initial", "weights_bytes": best_path.stat().st_size,
                    "smoke_run": bool(args.train_limit or args.valid_limit), **initial_result},
                   run_dir / "best_metrics.json")

    train_dataset = PlantDataset(args.data_dir, "train", get_transforms(
        True, not args.no_augment, profile=args.augmentation))
    train_set = train_dataset
    rows = train_set.rows
    if args.fold_manifest:
        training_indices, _ = fold_indices(train_dataset, args.fold_manifest, args.fold_index)
        train_set = Subset(train_dataset, training_indices)
        rows = [rows[i] for i in training_indices]
    if args.train_limit:
        indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(args.seed))[
            :args.train_limit].tolist()
        rows = [rows[i] for i in indices]
        train_set = Subset(train_set, indices)
    if len(train_set) < 2:
        raise ValueError("Need at least two training images")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              worker_init_fn=seed_worker, generator=generator)
    powers = [args.class_weight_power if power is None else power for power in
              (args.crop_class_weight_power, args.health_class_weight_power)]
    criteria = [nn.CrossEntropyLoss(
        weight=class_weights(rows, key, classes, power, device),
        label_smoothing=args.label_smoothing,
    ) for (key, classes), power in zip((("crop_label", 3), ("health_label", 2)), powers)]
    print(f"Loss weights: crop {args.crop_loss_weight}, health {args.health_loss_weight}; "
          f"augmentation {args.augmentation}")
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)
    warmup = min(args.warmup_epochs, max(0, args.epochs - 1))

    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = min(1.0, (epoch - warmup) / max(1, args.epochs - warmup - 1))
        minimum = args.min_lr / args.lr
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay else None
    history, best_score, first_epoch = [], initial_result["score"] if initial_result else -1.0, 1
    swa_state, swa_count = None, 0
    if args.resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        ignored = {"resume", "eval_only", "run_dir", "device", "num_workers", "num_threads"}
        if any(saved["args"][key] != value for key, value in vars(args).items() if key not in ignored):
            raise ValueError("Resume requires the same training options, including total epochs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        if ema:
            ema.model.load_state_dict(saved["ema"])
            ema.steps = saved["ema_steps"]
        history, best_score = saved["history"], saved["best_score"]
        swa_state, swa_count = saved["swa_state"], saved["swa_count"]
        first_epoch = saved["epoch"] + 1
        random.setstate(saved["random_state"])
        torch.set_rng_state(saved["torch_rng"])
        generator.set_state(saved["loader_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"Resuming epoch {first_epoch}; best score {best_score:.4f}")

    validation_count = len(valid_set) if valid_set is not None else "withheld until weights freeze"
    print(f"Train {len(train_set)}; valid {validation_count}; AMP {amp}", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        start = time.perf_counter()
        finishing = args.finish_fraction > 0 and epoch > args.epochs * (1 - args.finish_fraction)
        profile = "finish" if finishing else args.augmentation
        train_dataset.transform = get_transforms(True, not args.no_augment, profile=profile)
        for criterion in criteria:
            criterion.label_smoothing = 0.0 if finishing else args.label_smoothing
        model.train()
        total_loss, seen, updates = 0.0, 0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, crops, healths in train_loader:
            images = images.to(device, non_blocking=True)
            crops, healths = crops.to(device, non_blocking=True), healths.to(device, non_blocking=True)
            mixed_targets, fraction = None, 1.0
            if (not finishing and not args.no_augment and args.mixup_alpha > 0
                    and random.random() < args.mixup_prob):
                images, mixed_targets, fraction = mix_batch(images, (crops, healths), args.mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                crop, health = model(images)
                loss = (args.crop_loss_weight * criteria[0](crop, crops)
                        + args.health_loss_weight * criteria[1](health, healths))
                if mixed_targets is not None:
                    loss = (fraction * loss + (1 - fraction) * (
                        args.crop_loss_weight * criteria[0](crop, mixed_targets[0])
                        + args.health_loss_weight * criteria[1](health, mixed_targets[1])))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not amp and not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradients")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                updates += 1
                if ema:
                    ema.update(model)
            total_loss += loss.item() * len(images)
            seen += len(images)
        if not updates:
            raise FloatingPointError("No optimizer updates completed")
        if args.swa_start < 1 and epoch > args.epochs * args.swa_start:
            current = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            swa_state = average_states(swa_state, current, swa_count) if swa_state else current
            swa_count += 1
        raw_result = evaluate(model, valid_loader, device) if not args.fixed_epochs else None
        ema_result = evaluate(ema.model, valid_loader, device) if ema and not args.fixed_epochs else None
        selected, result, variant = model, raw_result or {}, "raw"
        if ema_result and ema_result["score"] > raw_result["score"]:
            selected, result, variant = ema.model, ema_result, "ema"
        record = {"epoch": epoch, "train_loss": total_loss / seen, "lr": learning_rate,
                  "seconds": time.perf_counter() - start, "optimizer_updates": updates,
                  "raw_score": raw_result["score"] if raw_result else None,
                  "ema_score": ema_result["score"] if ema_result else None,
                  "selected": variant, "augmentation": profile, "swa_count": swa_count, **result}
        history.append(record)
        metrics_text = ("valid withheld; fixed refit" if args.fixed_epochs else
                        f"crop F1 {result['crop_f1']:.4f} health F1 {result['health_f1']:.4f} "
                        f"score {result['score']:.4f} ({variant})")
        print(f"[{epoch:02d}/{args.epochs}] loss {record['train_loss']:.4f} {metrics_text} "
              f"lr {learning_rate:.2g} {record['seconds']:.1f}s aug {profile}", flush=True)
        if not args.fixed_epochs and result["score"] > best_score:
            best_score = result["score"]
            save_candidate(selected, result, best_path, epoch, variant,
                           bool(args.train_limit or args.valid_limit))
            print(f"  Saved best_model.pt ({best_path.stat().st_size / 1024 ** 2:.2f} MiB)", flush=True)
        scheduler.step()
        write_json(history, run_dir / "history.json")
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["epoch", "train_loss", "lr", "seconds", "crop_f1", "health_f1", "score",
                      "raw_score", "ema_score", "selected", "optimizer_updates"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
        atomic_save({
            "epoch": epoch, "args": vars(args), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "ema": ema.model.state_dict() if ema else None,
            "swa_state": swa_state, "swa_count": swa_count,
            "ema_steps": ema.steps if ema else 0, "history": history, "best_score": best_score,
            "random_state": random.getstate(), "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, last_path)
    if args.fixed_epochs:
        if args.fixed_variant == "swa":
            if not swa_state:
                raise ValueError("No SWA samples were collected")
            model.load_state_dict(swa_state)
            frozen, frozen_variant = model, "swa"
        elif args.fixed_variant == "ema" and ema:
            frozen, frozen_variant = ema.model, "ema"
        else:
            frozen, frozen_variant = model, "raw"
        if not args.no_bn_recalibration:
            recalibrate_bn(frozen, clean_training_loader(args), device)
        # Freeze the checkpoint before using any official validation predictions or scores.
        atomic_save({key: value.detach().cpu() for key, value in frozen.state_dict().items()}, best_path)
        digest = hashlib.sha256(best_path.read_bytes()).hexdigest()
        write_json({"epochs": args.epochs, "variant": frozen_variant,
                    "weights_sha256_before_validation": digest,
                    "validation_source": "train_fold" if args.fold_manifest else "official_valid",
                    "official_valid_used_for_selection": False}, run_dir / "frozen_selection.json")
        _, valid_loader = validation_loader(args, device)
        final_result = evaluate(frozen, valid_loader, device)
        if digest != hashlib.sha256(best_path.read_bytes()).hexdigest():
            raise RuntimeError("Frozen checkpoint changed during validation")
        if best_path.stat().st_size > 10_000_000:
            raise ValueError("Checkpoint exceeds 10 MB")
        write_json({"epoch": args.epochs, "variant": "fixed_" + frozen_variant,
                    "weights_bytes": best_path.stat().st_size,
                    "smoke_run": bool(args.train_limit or args.valid_limit), **final_result},
                   run_dir / "best_metrics.json")
        print_report(final_result, names)
        write_json(final_result, run_dir / "evaluation.json")
        return
    # Candidate selection can only improve the validation score, including the starting model.
    candidates = {"best_bn": torch.load(best_path, map_location="cpu", weights_only=True),
                  "last_bn": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
    if ema:
        candidates["ema_bn"] = {key: value.detach().cpu().clone() for key, value in ema.model.state_dict().items()}
    if swa_state:
        candidates["swa_bn"] = swa_state
    model.load_state_dict(candidates["best_bn"])
    # A resumed, completed run may already contain a better post-training candidate.
    best_score = max(best_score, evaluate(model, valid_loader, device)["score"])
    if not args.no_bn_recalibration:
        clean_loader = clean_training_loader(args)
        # DataLoader consumes RNG even without shuffling. Preserve training RNG for reproducibility.
        rng = torch.get_rng_state()
        for variant, state in candidates.items():
            model.load_state_dict(state)
            recalibrate_bn(model, clean_loader, device)
            result = evaluate(model, valid_loader, device)
            print(f"Final candidate {variant}: mean Macro-F1 {result['score']:.4f}", flush=True)
            if result["score"] > best_score:
                best_score = result["score"]
                save_candidate(model, result, best_path, args.epochs, variant,
                               bool(args.train_limit or args.valid_limit))
        torch.set_rng_state(rng)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_result = evaluate(model, valid_loader, device)
    print_report(final_result, names)
    write_json(final_result, run_dir / "evaluation.json")


def prediction_metrics(labels, predictions):
    result = {}
    for task, classes, column in (("crop", 3, 0), ("health", 2, 1)):
        matrix = torch.bincount(labels[:, column] * classes + predictions[:, column],
                                minlength=classes * classes).reshape(classes, classes)
        score, per_class = macro_f1(matrix)
        result.update({task + "_f1": score, task + "_f1_per_class": per_class,
                       task + "_confusion": matrix.tolist()})
    result["score"] = (result["crop_f1"] + result["health_f1"]) / 2
    return result


def paired_bootstrap(labels, reference, candidate, repeats=1000, seed=2026):
    """Paired, joint-label-stratified resampling; uncertainty on valid, not a test guarantee."""
    generator = torch.Generator().manual_seed(seed)
    joint = labels[:, 0] * 2 + labels[:, 1]
    groups = [(joint == group).nonzero().flatten() for group in joint.unique().tolist()]
    differences = []
    for _ in range(repeats):
        indices = torch.cat([group[torch.randint(len(group), (len(group),), generator=generator)]
                             for group in groups])
        differences.append(prediction_metrics(labels[indices], candidate[indices])["score"]
                           - prediction_metrics(labels[indices], reference[indices])["score"])
    bounds = torch.tensor(differences).quantile(torch.tensor([0.025, 0.975])).tolist()
    return {"delta": prediction_metrics(labels, candidate)["score"]
                     - prediction_metrics(labels, reference)["score"],
            "ci95": bounds, "clear_valid_improvement": bounds[0] > 0, "repeats": repeats}


@torch.no_grad()
def compare_runs(args):
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    dataset = PlantDataset(args.data_dir, "valid", get_transforms(False))
    rows = dataset.rows
    if args.valid_limit:
        rows = rows[:args.valid_limit]
        dataset = Subset(dataset, range(len(rows)))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    output = Path(args.run_dir).resolve()
    if output in {Path(path).resolve() for path in args.compare_runs}:
        raise ValueError("Comparison output must be separate from the trained model folders")
    output.mkdir(parents=True, exist_ok=True)
    reports, reference = [], None
    for index, directory in enumerate(args.compare_runs):
        directory = Path(directory).resolve()
        model_path, weights = directory / "model.py", directory / "best_model.pt"
        before = hashlib.sha256(weights.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(f"comparison_model_{index}", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.MultiTaskCNN().to(device).eval()
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        batches, truths, probabilities = [], [], [[], []]
        for images, crops, healths in loader:
            logits = model(images.to(device))
            if any(out.shape != (len(images), classes) or not torch.isfinite(out).all()
                   for out, classes in zip(logits, (3, 2))):
                raise ValueError("Invalid saved model output")
            batches.append(torch.stack([out.argmax(1).cpu() for out in logits], dim=1))
            truths.append(torch.stack((crops, healths), dim=1))
            for target, out in zip(probabilities, logits):
                target.append(out.softmax(1).cpu())
        labels, predictions = torch.cat(truths), torch.cat(batches)
        probabilities = [torch.cat(parts) for parts in probabilities]
        result = prediction_metrics(labels, predictions)
        result.update({"run": str(directory), "weights_sha256": before,
                       "smoke_run": bool(args.valid_limit), "groups": {}})
        for group in (labels[:, 0] * 2 + labels[:, 1]).unique().tolist():
            mask = labels[:, 0] * 2 + labels[:, 1] == group
            result["groups"][str(group)] = {"count": int(mask.sum()),
                "crop_correct": int((predictions[mask, 0] == labels[mask, 0]).sum()),
                "health_correct": int((predictions[mask, 1] == labels[mask, 1]).sum())}
        for column, key in enumerate(("crop", "health")):
            confidence, guess = probabilities[column].max(1)
            wrong = guess != labels[:, column]
            result[key + "_nll"] = -probabilities[column][torch.arange(len(labels)), labels[:, column]].clamp_min(1e-8).log().mean().item()
            result[key + "_wrong_confidence"] = confidence[wrong].mean().item() if wrong.any() else None
        if reference is None:
            reference = predictions
        else:
            result["paired_vs_first"] = paired_bootstrap(labels, reference, predictions, args.bootstrap)
        error_path = output / f"errors_{index}_{directory.name}.csv"
        with error_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "crop_true", "health_true", "crop_pred", "health_pred"])
            for row, truth, prediction in zip(rows, labels.tolist(), predictions.tolist()):
                if truth != prediction:
                    writer.writerow([row["id"], *truth, *prediction])
        if before != hashlib.sha256(weights.read_bytes()).hexdigest():
            raise RuntimeError("Comparison modified a checkpoint")
        print(f"\nSaved model: {directory}")
        print_report(result, load_class_names(args.data_dir))
        if "paired_vs_first" in result:
            print("Valid difference and 95% paired interval:", result["paired_vs_first"])
        reports.append(result)
        del model
    write_json(reports, output / "comparison.json")


def cli_command(options, source_dir):
    command = [sys.executable, "-u", str(source_dir / "train.py")]
    for key, value in options.items():
        if value is None or value is False:
            continue
        if value is True:
            command.append("--" + key)
        elif isinstance(value, list):
            command.extend(["--" + key, *map(str, value)])
        else:
            command.extend(["--" + key, str(value)])
    return command


def run_cv(args):
    source_dir = Path(__file__).resolve().parent
    root = Path(args.run_dir).resolve()
    hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
              for name in ("model.py", "dataset.py", "train.py")}
    config_path = root / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        ignored = {"resume", "device", "num_workers", "num_threads"}
        if not args.resume or saved["source_sha256"] != hashes or any(
            saved["args"][key] != value for key, value in vars(args).items() if key not in ignored
        ):
            raise ValueError("CV run exists; resume with the same code/options or choose another folder")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("CV output folder is not empty; choose another folder")
        root.mkdir(parents=True, exist_ok=True)
        for name in hashes:
            shutil.copy2(source_dir / name, root / name)
        write_json({"args": vars(args), "source_sha256": hashes}, config_path)
        pool = PlantDataset(args.data_dir, "train", get_transforms(False))
        if args.train_limit:
            indices = torch.randperm(len(pool), generator=torch.Generator().manual_seed(args.fold_seed))[
                :args.train_limit].tolist()
            pool.rows = [pool.rows[i] for i in indices]
        manifest = grouped_folds(pool, args.cv_folds, args.fold_seed)
        manifest["train_csv_sha256"] = hashlib.sha256((Path(args.data_dir) / "train.csv").read_bytes()).hexdigest()
        write_json(manifest, root / "folds.json")
    manifest_path = root / "folds.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["train_csv_sha256"] != hashlib.sha256((Path(args.data_dir) / "train.csv").read_bytes()).hexdigest():
        raise ValueError("Training CSV changed after the CV split was frozen")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config = json.loads(config_path.read_text())
    if config.get("fold_manifest_sha256", manifest_hash) != manifest_hash:
        raise ValueError("CV fold assignments changed after they were frozen")
    config["fold_manifest_sha256"] = manifest_hash
    write_json(config, config_path)
    print("Train-only joint-stratified fold counts:", manifest["joint_counts"], flush=True)
    print("Official valid is excluded from recipe/epoch selection.", flush=True)
    baseline = {"disable_detail": True, "augmentation": "gentle", "mixup_alpha": 0.0,
                "mixup_prob": 0.0, "finish_fraction": 0.0, "drop_path": 0.0,
                "label_smoothing": 0.02, "dropout": 0.2, "weight_decay": 3e-4,
                "swa_start": 1.0, "no_bn_recalibration": True, "ema_decay": 0.99, "fixed_variant": "ema"}
    recipes = [
        {"name": "previous_deep", "options": baseline},
        {"name": "robust_deep", "options": {**baseline, "augmentation": "robust",
             "mixup_alpha": 0.1, "mixup_prob": 0.25, "drop_path": 0.02,
             "no_bn_recalibration": False}},
        {"name": "multiscale", "options": {**baseline, "disable_detail": False,
             "mixup_alpha": 0.1, "mixup_prob": 0.25, "drop_path": 0.02,
             "swa_start": 0.75, "no_bn_recalibration": False, "fixed_variant": "swa"}},
    ]
    summaries = []
    for recipe in recipes:
        metrics = []
        for fold in range(args.cv_folds):
            child = root / recipe["name"] / f"fold{fold}"
            options = {**vars(args), **recipe["options"], "cv": False, "search": False,
                       "compare_runs": None, "fixed_epochs": True, "eval_only": False,
                       "run_dir": str(child), "fold_manifest": str(manifest_path), "fold_index": fold,
                       "seed": args.seed + fold, "epochs": args.cv_epochs,
                       "resume": (child / "last_checkpoint.pt").exists()}
            print(f"\nCV {recipe['name']}; inner fold {fold + 1}/{args.cv_folds}", flush=True)
            subprocess.run(cli_command(options, source_dir), cwd=source_dir, check=True)
            metrics.append(json.loads((child / "best_metrics.json").read_text()))
        scores = [metric["score"] for metric in metrics]
        summary = {**recipe, "folds": metrics, "mean": statistics.mean(scores),
                   "std": statistics.stdev(scores), "se": statistics.stdev(scores) / math.sqrt(len(scores))}
        summaries.append(summary)
        write_json(summaries, root / "cv_results.json")
    winner = max(summaries, key=lambda row: row["mean"])
    # Predeclared one-standard-error rule favors the simpler earlier recipe when close.
    chosen = next(row for row in summaries if row["mean"] >= winner["mean"] - winner["se"])
    median_epoch = statistics.median(row["epoch"] for row in chosen["folds"])
    # Keep the same per-image epoch budget rather than extending training after seeing valid.
    final_epochs = max(1, min(args.epochs, round(median_epoch)))
    decision = {"chosen_recipe": chosen["name"], "options": chosen["options"],
                "mean_inner_f1": chosen["mean"], "std_inner_f1": chosen["std"],
                "rule": "one_standard_error_in_declared_recipe_order", "refit_epochs": final_epochs,
                "official_valid_used_for_selection": False, "public_used_for_selection": False,
                "smoke_run": bool(args.train_limit or args.valid_limit)}
    write_json(decision, root / "cv_selection.json")
    print("\nFrozen train-only decision:", decision, flush=True)
    final = root / "final_fit"
    options = {**vars(args), **chosen["options"], "cv": False, "search": False, "compare_runs": None,
               "fixed_epochs": True, "fold_manifest": None, "fold_index": -1, "epochs": final_epochs,
               "run_dir": str(final), "eval_only": False, "resume": (final / "last_checkpoint.pt").exists()}
    subprocess.run(cli_command(options, source_dir), cwd=source_dir, check=True)
    for name in ("model.py", "best_model.pt", "best_metrics.json", "evaluation.json", "frozen_selection.json"):
        shutil.copy2(final / name, root / name)
    print("\nFinal model frozen and evaluated once on official valid:", root, flush=True)


def run_search(args):
    """Train three recipes and export one model; never concatenate an ensemble."""
    source_dir = Path(__file__).resolve().parent
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run_dir = Path(args.run_dir).resolve()
    source_paths = {name: source_dir / name for name in ("model.py", "dataset.py", "train.py")}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}
    ignored = {"resume", "device", "num_workers", "num_threads"}
    if (run_dir / "config.json").exists():
        if not args.resume:
            raise FileExistsError("Search exists: use another --run_dir or repeat the command with --resume")
        existing = json.loads((run_dir / "config.json").read_text())
        if existing["source_sha256"] != hashes or any(
            existing["args"][key] != value for key, value in vars(args).items() if key not in ignored
        ):
            raise ValueError("Search resume requires the same sources and training settings")
        if args.init_weights and existing["init_weights_sha256"] != hashlib.sha256(
            Path(args.init_weights).read_bytes()
        ).hexdigest():
            raise ValueError("The search starting checkpoint changed; use the original checkpoint")
    else:
        if any((run_dir / name).exists() for name in ("best_model.pt", "last_checkpoint.pt")):
            raise FileExistsError("Run already contains weights; choose a new --run_dir")
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, source in source_paths.items():
            shutil.copy2(source, run_dir / name)
        write_json({"args": vars(args), "source_sha256": hashes,
                    "init_weights_sha256": hashlib.sha256(Path(args.init_weights).read_bytes()).hexdigest()
                    if args.init_weights else None}, run_dir / "config.json")
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a Colab GPU")
    spec = importlib.util.spec_from_file_location("search_leaf_model", source_paths["model.py"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MultiTaskCNN().to(device)
    if args.init_weights:
        # Strict compatibility check before spending time on any child training.
        load_initial_weights(model, torch.load(args.init_weights, map_location=device, weights_only=True))
    validation = PlantDataset(args.data_dir, "valid", get_transforms(False))
    if args.valid_limit:
        validation = Subset(validation, range(min(args.valid_limit, len(validation))))
    valid_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.num_workers)
    recipes = [
        {"augmentation": "gentle", "lr": args.lr, "weight_decay": 1e-4,
         "mixup_alpha": 0.15, "mixup_prob": 0.35, "label_smoothing": 0.01,
         "dropout": 0.2, "drop_path": 0.02},
        {"augmentation": "detail", "lr": args.lr * 0.5, "weight_decay": 3e-4,
         "mixup_alpha": 0.0, "mixup_prob": 0.0, "label_smoothing": 0.01,
         "crop_class_weight_power": 0.35, "dropout": 0.1, "drop_path": 0.0},
        {"augmentation": "standard", "lr": args.lr * 1.5, "weight_decay": 3e-4,
         "mixup_alpha": 0.2, "mixup_prob": 0.5, "label_smoothing": 0.02,
         "dropout": 0.2, "drop_path": 0.05},
    ]
    candidates = []
    for index, (seed, recipe) in enumerate(zip(args.search_seeds, recipes), 1):
        child = run_dir / f"trial{index}_seed{seed}"
        options = {**vars(args), **recipe, "seed": seed, "run_dir": str(child),
                   "search": False, "eval_only": False, "resume": (child / "last_checkpoint.pt").exists()}
        command = [sys.executable, "-u", str(source_dir / "train.py")]
        for key, value in options.items():
            if key in ("search", "search_seeds") or value is None:
                continue
            if isinstance(value, bool):
                if value:
                    command.append("--" + key)
            else:
                command.extend(["--" + key, str(value)])
        print(f"\nSearch trial {index}/3: seed {seed}; {recipe}", flush=True)
        subprocess.run(command, cwd=source_dir, check=True)
        metrics = json.loads((child / "best_metrics.json").read_text())
        candidates.append({"name": child.name, "weights": str(child / "best_model.pt"),
                           "score": metrics["score"]})
    # Include the untouched starting checkpoint as a candidate too.
    if args.init_weights:
        load_initial_weights(model, torch.load(args.init_weights, map_location=device, weights_only=True))
        result = evaluate(model, valid_loader, device)
        candidates.append({"name": "initial", "weights": str(Path(args.init_weights).resolve()),
                           "score": result["score"]})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    load_initial_weights(model, torch.load(best["weights"], map_location=device, weights_only=True))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    result = evaluate(model, valid_loader, device)
    best_score = result["score"]
    smoke = bool(args.train_limit or args.valid_limit)
    save_candidate(model, result, run_dir / "best_model.pt", 0, best["name"], smoke)
    soup_records, accepted, count = [], [best["name"]], 1
    soup_state = best_state
    # Independently initialized networks have unaligned channels. Average only shared-init runs.
    if args.init_weights and not args.no_bn_recalibration:
        clean_loader = clean_training_loader(args)
        for candidate in candidates[1:]:
            load_initial_weights(model, torch.load(candidate["weights"], map_location=device, weights_only=True))
            right = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            proposed = average_states(soup_state, right, count)
            model.load_state_dict(proposed)
            recalibrate_bn(model, clean_loader, device)
            score = evaluate(model, valid_loader, device)
            improved = score["score"] > best_score
            soup_records.append({"added": candidate["name"], "score": score["score"], "accepted": improved})
            print(f"Weight average + {candidate['name']}: {score['score']:.4f}; accepted {improved}", flush=True)
            if improved:
                count += 1
                accepted.append(candidate["name"])
                soup_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_score = score["score"]
                save_candidate(model, score, run_dir / "best_model.pt", 0, "weight_average", smoke)
    write_json({"candidates": candidates, "weight_average": soup_records,
                "accepted": accepted, "smoke_run": smoke}, run_dir / "search_results.json")
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    result = evaluate(model, valid_loader, device)
    print("\nFinal single-model validation result:")
    print_report(result, load_class_names(args.data_dir))
    write_json(result, run_dir / "evaluation.json")


if __name__ == "__main__":
    main()
