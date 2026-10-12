"""Official CSV format, with training-only augmentation and fixed grader transforms."""
import csv
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def load_class_names(data_dir):
    with (Path(data_dir) / "class_mapping.json").open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    return tuple(
        [mapping[key][str(i)] for i in range(len(mapping[key]))]
        for key in ("crop_label", "health_label")
    )


class PlantDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.img_dir = Path(data_dir) / split / "images"
        self.transform = transform
        with (Path(data_dir) / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Empty split: {split}")
        for row in self.rows:
            if not 0 <= int(row["crop_label"]) < 3 or not 0 <= int(row["health_label"]) < 2:
                raise ValueError(f"Invalid labels: {row}")
            if not (self.img_dir / row["id"]).is_file():
                raise FileNotFoundError(self.img_dir / row["id"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.img_dir / row["id"]) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["crop_label"]), int(row["health_label"])


def grouped_folds(dataset, folds=3, seed=2026):
    """Stratify crop/health pairs and keep identical RGB images in one fold."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    groups = {}
    for index, row in enumerate(dataset.rows):
        with Image.open(dataset.img_dir / row["id"]) as image:
            digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        groups.setdefault(digest, []).append(index)
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: -len(item[1]))
    counts = [[0] * 6 for _ in range(folds)]
    totals = [0] * folds
    entries = []
    for digest, indices in items:
        labels = [int(dataset.rows[i]["crop_label"]) * 2 + int(dataset.rows[i]["health_label"])
                  for i in indices]
        fold = min(range(folds), key=lambda f: (sum(counts[f][label] for label in labels), totals[f], f))
        for index, label in zip(indices, labels):
            counts[fold][label] += 1
            totals[fold] += 1
            entries.append({**dataset.rows[index], "fold": fold, "pixel_sha256": digest})
    if not all(totals):
        raise ValueError("Not enough distinct images for the requested folds")
    return {"folds": folds, "seed": seed, "rows": entries, "joint_counts": counts,
            "identical_image_groups": sum(len(indices) > 1 for indices in groups.values())}


def fold_indices(dataset, manifest_path, fold_index):
    manifest = json.loads(Path(manifest_path).read_text())
    if not 0 <= fold_index < manifest["folds"]:
        raise ValueError("Invalid fold index")
    lookup = {row["id"]: (index, row) for index, row in enumerate(dataset.rows)}
    train, valid, seen = [], [], set()
    for entry in manifest["rows"]:
        if entry["id"] in seen:
            raise ValueError("Repeated id in fold manifest")
        seen.add(entry["id"])
        index, row = lookup[entry["id"]]
        if any(row[key] != entry[key] for key in ("crop_label", "health_label")):
            raise ValueError("Dataset labels differ from the frozen fold manifest")
        (valid if entry["fold"] == fold_index else train).append(index)
    if len(train) < 2 or not valid:
        raise ValueError("Empty or insufficient fold data")
    return train, valid


def get_transforms(train, augment=True, profile="standard"):
    normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
    if train and augment:
        if profile not in ("standard", "gentle", "detail", "finish", "robust", "leaf"):
            raise ValueError(f"Unknown augmentation profile: {profile}")
        if profile == "leaf":
            # Retain the full leaf; perturb orientation and appearance without large crops.
            return transforms.Compose([
                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.RandomApply([transforms.RandomAffine(
                    10, translate=(0.025, 0.025), scale=(0.97, 1.03),
                    interpolation=InterpolationMode.BILINEAR, fill=(128, 128, 128))], p=0.3),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.5))], p=0.1),
                transforms.ToTensor(), normalize,
            ])
        if profile == "robust":
            return transforms.Compose([
                transforms.RandomResizedCrop(128, scale=(0.9, 1.0), ratio=(0.9, 1.1),
                                             interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomAffine(25, translate=(0.04, 0.04), scale=(0.95, 1.05),
                                        interpolation=InterpolationMode.BILINEAR, fill=(128, 128, 128)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.6))], p=0.15),
                transforms.ToTensor(), normalize,
            ])
        if profile in ("detail", "finish"):
            # Preserve all leaf pixels and lesions; quarter turns need no interpolation.
            operations = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                          transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5)]
            if profile == "detail":
                operations.append(transforms.ColorJitter(brightness=0.05, contrast=0.05,
                                                        saturation=0.04))
            return transforms.Compose([*operations, transforms.ToTensor(), normalize])
        gentle = profile == "gentle"
        return transforms.Compose([
            transforms.RandomResizedCrop(
                128, scale=(0.9, 1.0) if gentle else (0.8, 1.0), ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15 if gentle else 25, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(
                brightness=0.10 if gentle else 0.15, contrast=0.10 if gentle else 0.15,
                saturation=0.08 if gentle else 0.12, hue=0.01 if gentle else 0.02,
            ),
            transforms.ToTensor(),
            normalize,
        ])
    # Do not change this: the official grader uses this exact preprocessing.
    return transforms.Compose([transforms.ToTensor(), normalize])
