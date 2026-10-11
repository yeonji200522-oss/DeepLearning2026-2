"""Evaluate a retained model on unaugmented train and valid; export misclassified images."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import macro_f1


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def inspect_split(model, dataset, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == "cuda", shuffle=False)
    crop_conf = torch.zeros(3, 3, dtype=torch.long)
    health_conf = torch.zeros(2, 2, dtype=torch.long)
    groups = {}
    errors, offset = [], 0
    for images, crops, healths in loader:
        crop_logits, health_logits = model(images.to(device, non_blocking=True))
        if crop_logits.shape != (len(images), 3) or health_logits.shape != (len(images), 2):
            raise ValueError("Invalid output shape")
        if not torch.isfinite(crop_logits).all() or not torch.isfinite(health_logits).all():
            raise ValueError("Non-finite logits")
        crop_prob, health_prob = crop_logits.softmax(1).cpu(), health_logits.softmax(1).cpu()
        crop_pred, health_pred = crop_prob.argmax(1), health_prob.argmax(1)
        crop_conf += torch.bincount(crops * 3 + crop_pred, minlength=9).reshape(3, 3)
        health_conf += torch.bincount(healths * 2 + health_pred, minlength=4).reshape(2, 2)
        for i in range(len(images)):
            crop, health = int(crops[i]), int(healths[i])
            predicted_crop, predicted_health = int(crop_pred[i]), int(health_pred[i])
            key = f"crop{crop}_health{health}"
            group = groups.setdefault(key, {"count": 0, "crop_correct": 0, "health_correct": 0})
            group["count"] += 1
            group["crop_correct"] += int(crop == predicted_crop)
            group["health_correct"] += int(health == predicted_health)
            if crop != predicted_crop or health != predicted_health:
                row = dataset.rows[offset + i]
                errors.append({
                    "id": row["id"], "image_path": str(Path(dataset.img_dir) / row["id"]),
                    "crop_true": crop, "crop_pred": predicted_crop,
                    "health_true": health, "health_pred": predicted_health,
                    "crop_pred_probability": float(crop_prob[i, predicted_crop]),
                    "health_pred_probability": float(health_prob[i, predicted_health]),
                })
        offset += len(images)
    crop_f1, crop_per_class = macro_f1(crop_conf)
    health_f1, health_per_class = macro_f1(health_conf)
    return {
        "count": offset, "crop_f1": crop_f1, "health_f1": health_f1,
        "score": (crop_f1 + health_f1) / 2,
        "crop_f1_per_class": crop_per_class, "health_f1_per_class": health_per_class,
        "crop_confusion": crop_conf.tolist(), "health_confusion": health_conf.tolist(),
        "groups": groups,
    }, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="runs/improved_seed42")
    parser.add_argument("--data_dir", default="student_data_128")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid batch size or worker count")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    run_dir = Path(args.run_dir).resolve()
    # Saved run sources keep older checkpoints usable after training-code changes.
    model_module = load_module("retained_model", run_dir / "model.py")
    data_module = load_module("retained_dataset", run_dir / "dataset.py")
    model = model_module.MultiTaskCNN().to(device).eval()
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    report = {"device": str(device), "preprocessing": "official evaluation; no augmentation"}
    for split in ("train", "valid"):
        dataset = data_module.PlantDataset(args.data_dir, split, data_module.get_transforms(False))
        result, errors = inspect_split(model, dataset, device, args.batch_size, args.num_workers)
        report[split] = result
        print(f"{split}: {result['count']} images; crop F1 {result['crop_f1']:.4f}; "
              f"health F1 {result['health_f1']:.4f}; mean {result['score']:.4f}", flush=True)
        print("  crop F1 per class:", result["crop_f1_per_class"])
        print("  crop confusion:", result["crop_confusion"])
        print("  class combinations:", result["groups"])
        with (run_dir / f"{split}_errors.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["id", "image_path", "crop_true", "crop_pred", "health_true", "health_pred",
                      "crop_pred_probability", "health_pred_probability"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(errors)
    report["train_minus_valid_score"] = report["train"]["score"] - report["valid"]["score"]
    (run_dir / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Train - valid Macro-F1:", round(report["train_minus_valid_score"], 4))
    print("Saved diagnostic.json and train_errors.csv / valid_errors.csv; checkpoint unchanged")


if __name__ == "__main__":
    main()
