"""Official CSV format, with training-only augmentation and fixed grader transforms."""
import csv
import json
from pathlib import Path

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


def get_transforms(train, augment=True):
    normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                128, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ])
    # Do not change this: the official grader uses this exact preprocessing.
    return transforms.Compose([transforms.ToTensor(), normalize])
