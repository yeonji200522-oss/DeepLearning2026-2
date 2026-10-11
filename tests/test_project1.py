"""Contract and checkpoint tests; set PLANT_DATA_DIR for real-image integration tests."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project1"
sys.path.insert(0, str(PROJECT))
from model import MultiTaskCNN
from dataset import get_transforms, PlantDataset
from train import macro_f1, evaluate

spec = importlib.util.spec_from_file_location("official_checker", PROJECT / "check_submission.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class SubmissionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_and_custom_heads_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        model = MultiTaskCNN().eval()
        total = sum(p.numel() for p in model.parameters()) + sum(
            b.numel() for b in model.buffers() if b.is_floating_point())
        self.assertLessEqual(total, checker.MAX_PARAMS)
        self.assertEqual(checker.check_imports(str(PROJECT / "model.py")), [])
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            expected = model(images)
            custom = MultiTaskCNN(5, 4).eval()(images)
        self.assertEqual([tuple(x.shape) for x in expected], [(2, 3), (2, 2)])
        self.assertEqual([tuple(x.shape) for x in custom], [(2, 5), (2, 4)])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            torch.save(model.state_dict(), checkpoint)
            self.assertLessEqual(checkpoint.stat().st_size, checker.MAX_WEIGHT_BYTES)
            restored = MultiTaskCNN().eval()
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            with torch.no_grad():
                actual = restored(images)
            for before, after in zip(expected, actual):
                self.assertTrue(torch.isfinite(after).all())
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_eval_transform_matches_grader_and_train_preserves_shape(self):
        values = torch.arange(128 * 128 * 3).remainder(256).byte().numpy().tobytes()
        image = Image.frombytes("RGB", (128, 128), values)
        torch.testing.assert_close(get_transforms(False)(image), checker.EVAL_TRANSFORM(image), rtol=0, atol=0)
        for profile in ("standard", "gentle"):
            torch.testing.assert_close(get_transforms(False, profile=profile)(image),
                                       checker.EVAL_TRANSFORM(image), rtol=0, atol=0)
            for seed in range(5):
                torch.manual_seed(seed)
                augmented = get_transforms(True, profile=profile)(image)
                self.assertEqual(tuple(augmented.shape), (3, 128, 128))
                self.assertTrue(torch.isfinite(augmented).all())
                self.assertGreaterEqual(augmented.min().item(), -1)
                self.assertLessEqual(augmented.max().item(), 1)

    def test_macro_f1_counts_all_classes_like_grader(self):
        pairs = [(0, 0), (0, 1), (1, 1), (1, 1), (1, 0)]
        confusion = torch.tensor([[1, 1, 0], [1, 2, 0], [0, 0, 0]])
        score, per_class = macro_f1(confusion)
        self.assertAlmostEqual(score, checker.macro_f1(pairs, 3), places=6)
        self.assertAlmostEqual(score, (0.5 + 2 / 3 + 0) / 3, places=6)
        self.assertEqual(per_class[2], 0)


@unittest.skipUnless(os.environ.get("PLANT_DATA_DIR"), "Set PLANT_DATA_DIR to run real-data tests")
class TrainingIntegrationTests(unittest.TestCase):
    def test_diagnostic_covers_images_and_matches_validation(self):
        from diagnose import inspect_split
        from torch.utils.data import DataLoader
        torch.set_num_threads(2)
        dataset = PlantDataset(os.environ["PLANT_DATA_DIR"], "valid", get_transforms(False))
        dataset.rows = dataset.rows[:16]
        model = MultiTaskCNN().eval()
        device = torch.device("cpu")
        result, errors = inspect_split(model, dataset, device, batch_size=8, workers=0)
        reference = evaluate(model, DataLoader(dataset, batch_size=8), device)
        self.assertEqual(result["count"], 16)
        self.assertEqual(sum(group["count"] for group in result["groups"].values()), 16)
        self.assertEqual(result["crop_confusion"], reference["crop_confusion"])
        self.assertEqual(result["health_confusion"], reference["health_confusion"])
        self.assertEqual(result["score"], reference["score"])
        for row in errors:
            self.assertTrue(Path(row["image_path"]).is_file())
            self.assertTrue(row["crop_true"] != row["crop_pred"] or row["health_true"] != row["health_pred"])

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        data = Path(os.environ["PLANT_DATA_DIR"]).resolve()
        with tempfile.TemporaryDirectory(prefix="leaf-integration-") as temporary:
            root = Path(temporary)
            common = ["--data_dir", str(data), "--epochs", "3", "--batch_size", "8",
                      "--train_limit", "32", "--valid_limit", "24", "--num_workers", "0",
                      "--device", "cpu", "--seed", "19", "--augmentation", "gentle",
                      "--crop_class_weight_power", "0.5", "--crop_loss_weight", "1.25"]
            continuous, resumed = root / "continuous", root / "resumed"

            def execute(arguments, expected=0):
                result = subprocess.run(arguments, cwd=PROJECT, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, expected, result.stdout)

            execute([sys.executable, "train.py", "--run_dir", str(continuous), *common])
            # Simulate a disconnect immediately after a complete epoch checkpoint.
            wrapper = (
                "import train,sys; original=train.atomic_save\n"
                "def save_and_disconnect(value,path):\n"
                " original(value,path)\n"
                " if path.name == 'last_checkpoint.pt': raise SystemExit(75)\n"
                "train.atomic_save=save_and_disconnect; train.main()\n"
            )
            execute([sys.executable, "-c", wrapper, "--run_dir", str(resumed), *common], expected=75)
            saved = torch.load(resumed / "last_checkpoint.pt", weights_only=True)
            self.assertEqual(saved["epoch"], 1)
            execute([sys.executable, "train.py", "--run_dir", str(resumed), *common, "--resume"])
            expected = torch.load(continuous / "last_checkpoint.pt", weights_only=True)
            actual = torch.load(resumed / "last_checkpoint.pt", weights_only=True)
            self.assertEqual(actual["epoch"], 3)
            self.assertEqual(len(actual["history"]), 3)
            for group in ("model", "ema"):
                for key in expected[group]:
                    torch.testing.assert_close(expected[group][key], actual[group][key], rtol=0, atol=0)
            self.assertEqual(expected["best_score"], actual["best_score"])
            self.assertEqual(expected["scheduler"], actual["scheduler"])
            for directory in (continuous, resumed):
                metrics = json.loads((directory / "best_metrics.json").read_text())
                self.assertTrue(metrics["smoke_run"])
                self.assertGreater(actual["history"][-1]["optimizer_updates"], 0)
                self.assertEqual((directory / "model.py").read_bytes(), (PROJECT / "model.py").read_bytes())
            # A second fresh invocation must refuse to overwrite a retained run.
            execute([sys.executable, "train.py", "--run_dir", str(resumed), *common], expected=1)
            # Fine-tuning must retain its starting checkpoint if later epochs are worse.
            refined = root / "refined"
            parent_metrics = json.loads((continuous / "best_metrics.json").read_text())
            execute([sys.executable, "train.py", "--run_dir", str(refined), *common,
                     "--epochs", "1", "--lr", "0.0001",
                     "--init_weights", str(continuous / "best_model.pt")])
            refined_metrics = json.loads((refined / "best_metrics.json").read_text())
            self.assertGreaterEqual(refined_metrics["score"], parent_metrics["score"])
            self.assertIsNotNone(json.loads((refined / "config.json").read_text())["init_weights_sha256"])


if __name__ == "__main__":
    unittest.main()
