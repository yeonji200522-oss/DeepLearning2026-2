"""Train from scratch, select by mean Macro-F1, and retain matching submission code."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import PlantDataset, get_transforms, load_class_names


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
    parser.add_argument("--run_dir", default="runs/improved_seed42")
    parser.add_argument("--model", choices=("standard", "deep"), default="standard",
                        help="deep adds residual blocks and larger heads; exported model.py matches weights")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--class_weight_power", type=float, default=0.35,
                        help="0 disables class weighting; only train labels determine weights")
    parser.add_argument("--crop_class_weight_power", type=float, default=None,
                        help="Override class_weight_power for the crop task only")
    parser.add_argument("--health_class_weight_power", type=float, default=None,
                        help="Override class_weight_power for the health task only")
    parser.add_argument("--crop_loss_weight", type=float, default=1.0)
    parser.add_argument("--health_loss_weight", type=float, default=1.0)
    parser.add_argument("--augmentation", choices=("standard", "gentle"), default="standard")
    parser.add_argument("--init_weights", default=None,
                        help="Start a NEW run from your own checkpoint trained only on the provided data")
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--ema_decay", type=float, default=0.99, help="0 disables EMA")
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
    return args


def main():
    args = parse_args()
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
    model_source = source_dir / ("model_deep.py" if args.model == "deep" else "model.py")
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
    if args.resume or args.eval_only:
        existing = json.loads((run_dir / "config.json").read_text())
        if existing["source_sha256"] != hashes:
            raise ValueError("Source changed since this run; use the saved run sources or a new run")
    else:
        for name, path in source_paths.items():
            shutil.copy2(path, run_dir / name)
        if args.model == "deep":
            # Retain this name too so the saved trainer can resume --model deep.
            shutil.copy2(model_source, run_dir / "model_deep.py")
        initial_hash = (hashlib.sha256(Path(args.init_weights).read_bytes()).hexdigest()
                        if args.init_weights else None)
        write_json({"args": vars(args), "source_sha256": hashes,
                    "init_weights_sha256": initial_hash,
                    "torch": torch.__version__, "device": str(device),
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
                   run_dir / "config.json")
    print(f"Device: {device}; model: {args.model}; run: {run_dir}", flush=True)
    if args.train_limit or args.valid_limit:
        print("SMOKE RUN: subset metrics are not full validation or submission performance.")
    names = load_class_names(args.data_dir)
    if tuple(map(len, names)) != (3, 2):
        raise ValueError("Expected 3 crop classes and 2 health classes")
    valid_set = PlantDataset(args.data_dir, "valid", get_transforms(False))
    if args.valid_limit:
        valid_set = Subset(valid_set, range(min(args.valid_limit, len(valid_set))))
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda")
    model = model_module.MultiTaskCNN().to(device)
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
        model.load_state_dict(torch.load(args.init_weights, map_location=device, weights_only=True))
        initial_result = evaluate(model, valid_loader, device)
        print(f"Own-data starting checkpoint: mean Macro-F1 {initial_result['score']:.4f}", flush=True)
        atomic_save({key: value.detach().cpu() for key, value in model.state_dict().items()}, best_path)
        if best_path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Starting checkpoint exceeds 10 MiB")
        write_json({"epoch": 0, "variant": "initial", "weights_bytes": best_path.stat().st_size,
                    "smoke_run": bool(args.train_limit or args.valid_limit), **initial_result},
                   run_dir / "best_metrics.json")

    train_set = PlantDataset(args.data_dir, "train", get_transforms(
        True, not args.no_augment, profile=args.augmentation))
    rows = train_set.rows
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
        first_epoch = saved["epoch"] + 1
        random.setstate(saved["random_state"])
        torch.set_rng_state(saved["torch_rng"])
        generator.set_state(saved["loader_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"Resuming epoch {first_epoch}; best score {best_score:.4f}")

    print(f"Train {len(train_set)}; valid {len(valid_set)}; AMP {amp}", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss, seen, updates = 0.0, 0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, crops, healths in train_loader:
            images = images.to(device, non_blocking=True)
            crops, healths = crops.to(device, non_blocking=True), healths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                crop, health = model(images)
                loss = (args.crop_loss_weight * criteria[0](crop, crops)
                        + args.health_loss_weight * criteria[1](health, healths))
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
        raw_result = evaluate(model, valid_loader, device)
        ema_result = evaluate(ema.model, valid_loader, device) if ema else None
        selected, result, variant = model, raw_result, "raw"
        if ema_result and ema_result["score"] > raw_result["score"]:
            selected, result, variant = ema.model, ema_result, "ema"
        record = {"epoch": epoch, "train_loss": total_loss / seen, "lr": learning_rate,
                  "seconds": time.perf_counter() - start, "optimizer_updates": updates,
                  "raw_score": raw_result["score"],
                  "ema_score": ema_result["score"] if ema_result else None,
                  "selected": variant, **result}
        history.append(record)
        print(f"[{epoch:02d}/{args.epochs}] loss {record['train_loss']:.4f} "
              f"crop F1 {result['crop_f1']:.4f} health F1 {result['health_f1']:.4f} "
              f"score {result['score']:.4f} ({variant}) lr {learning_rate:.2g} "
              f"{record['seconds']:.1f}s", flush=True)
        if result["score"] > best_score:
            best_score = result["score"]
            atomic_save({key: value.detach().cpu() for key, value in selected.state_dict().items()}, best_path)
            if best_path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("Checkpoint exceeds 10 MiB")
            write_json({"epoch": epoch, "variant": variant, "weights_bytes": best_path.stat().st_size,
                        "smoke_run": bool(args.train_limit or args.valid_limit), **result},
                       run_dir / "best_metrics.json")
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
            "ema_steps": ema.steps if ema else 0, "history": history, "best_score": best_score,
            "random_state": random.getstate(), "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, last_path)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_result = evaluate(model, valid_loader, device)
    print_report(final_result, names)
    write_json(final_result, run_dir / "evaluation.json")


if __name__ == "__main__":
    main()
