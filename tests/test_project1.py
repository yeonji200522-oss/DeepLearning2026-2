"""Contract and checkpoint tests; set PLANT_DATA_DIR for real-image integration tests."""
import importlib.util
import csv
import hashlib
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
from model import MultiTaskCNN, MBConv
from dataset import get_transforms, PlantDataset, grouped_folds, fold_indices
from train import (macro_f1, evaluate, load_initial_weights, mix_batch, average_states,
                   recalibrate_bn, configure_regularization)
from train import paired_bootstrap, sam_perturb, sam_restore, suspend_bn_updates
from train import find_resumable_run, run_lock, update_early_stop
from train import update_adaptive_stage

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
        for profile in ("standard", "gentle", "detail", "finish", "robust", "leaf"):
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

    def test_early_stopping_waits_for_final_phase_and_resets_on_improvement(self):
        state = {"best": None, "bad_epochs": 0, "stopped": False}
        for score in (0.9, 0.901, 0.9, 0.9):
            self.assertFalse(update_early_stop(state, score, False, 2, 0.0002))
        self.assertEqual(state["bad_epochs"], 0)
        self.assertFalse(update_early_stop(state, 0.9011, True, 2, 0.0002))
        self.assertTrue(update_early_stop(state, 0.9011, True, 2, 0.0002))
        self.assertFalse(update_early_stop(state, 0.902, True, 2, 0.0002))
        self.assertEqual(state["bad_epochs"], 0)

    def test_adaptive_schedule_lowers_lr_then_polishes_before_stopping(self):
        from types import SimpleNamespace
        args = SimpleNamespace(min_delta=0.0001, warmup_epochs=1, lr_patience=2,
                               lr_factor=0.5, min_lr=0.0005, polish_min_epochs=3, patience=2)
        state = {"phase": "learn", "lr": 0.001, "best": None, "bad_epochs": 0,
                 "reductions": 0, "polish_epochs": 0}
        self.assertIsNone(update_adaptive_stage(state, 0.9, 1, args))
        self.assertIsNone(update_adaptive_stage(state, 0.9, 2, args))
        self.assertEqual(update_adaptive_stage(state, 0.9, 3, args), "lower_lr")
        self.assertEqual(state["lr"], 0.0005)
        self.assertIsNone(update_adaptive_stage(state, 0.901, 4, args))
        self.assertIsNone(update_adaptive_stage(state, 0.901, 5, args))
        self.assertEqual(update_adaptive_stage(state, 0.901, 6, args), "start_polish")
        self.assertIsNone(update_adaptive_stage(state, 0.901, 7, args))
        self.assertIsNone(update_adaptive_stage(state, 0.901, 8, args))
        self.assertEqual(update_adaptive_stage(state, 0.901, 9, args), "stop")
        self.assertEqual(state["phase"], "done")

    def test_mbconv_backbone_receives_finite_gradients_from_both_tasks(self):
        torch.manual_seed(23)
        model = MultiTaskCNN().train()
        configure_regularization(model, dropout=0.2, drop_path=0.03)
        crop, health = model(torch.randn(2, 3, 128, 128))
        loss = (torch.nn.functional.cross_entropy(crop, torch.tensor([0, 2])) +
                torch.nn.functional.cross_entropy(health, torch.tensor([0, 1])))
        loss.backward()
        # Catch detached branches, unstable deep activations and disconnected heads.
        for block in model.features:
            if isinstance(block, MBConv):
                for weight in (block.depthwise[0].weight, block.project[0].weight,
                               block.attention.layers[1].weight):
                    self.assertIsNotNone(weight.grad)
                    self.assertTrue(torch.isfinite(weight.grad).all())
                    self.assertGreater(weight.grad.abs().sum().item(), 0)
        for layer in (model.crop_head, model.health_head, model.detail_projection,
                      model.crop_layout[-1]):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_crop_spatial_route_changes_crop_without_changing_health(self):
        torch.manual_seed(23)
        model = MultiTaskCNN().eval()
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            before = model(images)
            model.crop_layout[-1].weight.fill_(0.05)
            after = model(images)
        self.assertFalse(torch.allclose(before[0], after[0]))
        torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
        model.zero_grad(set_to_none=True)
        model(images)[1].square().sum().backward()
        self.assertTrue(all(p.grad is None for p in model.crop_layout.parameters()))

    def test_compatible_weights_preserve_predictions_and_reject_incomplete_or_old_architecture(self):
        old = MultiTaskCNN().eval()
        state = {key: value for key, value in old.state_dict().items()
                 if key != "detail_projection.weight"}
        restored = MultiTaskCNN().eval()
        load_initial_weights(restored, state)
        configure_regularization(restored, dropout=0.1, drop_path=0.2)
        images = torch.randn(2, 3, 128, 128)
        # Check the same backbone without a detail branch.
        with torch.no_grad():
            x = old.features(old.stem(images))
            features = torch.cat((old.avg_pool(x).flatten(1), old.max_pool(x).flatten(1)), dim=1)
            expected = (old.crop_head(old.crop_features(features)),
                        old.health_head(old.health_features(features)))
            actual = restored(images)
        for reference, result in zip(expected, actual):
            torch.testing.assert_close(reference, result, rtol=0, atol=0)
        self.assertEqual(restored.detail_projection.weight.count_nonzero().item(), 0)
        broken = dict(state)
        broken.pop("stem.0.weight")
        with self.assertRaises(ValueError):
            load_initial_weights(restored, broken)
        historical = dict(state)
        # The previous residual architecture used a 32-channel stem, not 24.
        historical["stem.0.weight"] = torch.randn(32, 3, 3, 3)
        with self.assertRaises(ValueError):
            load_initial_weights(restored, historical)

    def test_sam_gradient_scaling_restore_and_second_pass_bn_statistics(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        original = model.weight.detach().clone()
        model.weight.grad = torch.tensor([[3.0, 4.0]]) * 128
        saved = sam_perturb(model, 0.2, loss_scale=128)
        torch.testing.assert_close(model.weight, original + torch.tensor([[0.12, 0.16]]))
        torch.testing.assert_close((model.weight - original).norm(), torch.tensor(0.2))
        sam_restore(saved)
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
        model.weight.grad.fill_(float("inf"))
        self.assertIsNone(sam_perturb(model, 0.2))
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)

        bn = torch.nn.BatchNorm2d(3).train()
        images = torch.randn(4, 3, 8, 8)
        bn(images)
        state = {key: value.clone() for key, value in bn.state_dict().items()}
        reference = bn(images)
        bn.load_state_dict(state)
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            with suspend_bn_updates(bn):
                actual = bn(images)
                torch.testing.assert_close(actual, reference)
                raise RuntimeError("simulated")
        self.assertTrue(bn.track_running_stats)
        for key, value in state.items():
            torch.testing.assert_close(bn.state_dict()[key], value, rtol=0, atol=0)

    def test_sam_two_pass_update_with_grad_scaler(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=128)
        images = torch.ones(1, 1)
        first = model(images).square().mean()
        scaler.scale(first).backward()
        originals = sam_perturb(model, 0.1, scaler.get_scale())
        optimizer.zero_grad(set_to_none=True)
        try:
            scaler.scale(model(images).square().mean()).backward()
        finally:
            sam_restore(originals)
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        # Gradient at perturbed weight 1.1 is 2.2; apply it at original weight 1.0.
        torch.testing.assert_close(model.weight, torch.tensor([[0.78]]))

    def test_mixup_pairs_both_labels_and_weight_average_preserves_integer_buffers(self):
        torch.manual_seed(17)
        targets = torch.arange(8)
        images = targets.float().reshape(8, 1, 1, 1).expand(-1, 3, 8, 8)
        mixed, labels, fraction = mix_batch(images, (targets, targets + 100), 0.2)
        self.assertGreaterEqual(fraction, 0.5)
        torch.testing.assert_close(mixed[:, 0, 0, 0], fraction * targets + (1 - fraction) * labels[0])
        self.assertTrue(torch.equal(labels[0] + 100, labels[1]))
        left = {"weight": torch.tensor([1.0, 3.0]), "counter": torch.tensor(11)}
        right = {"weight": torch.tensor([5.0, 7.0]), "counter": torch.tensor(20)}
        actual = average_states(left, right, 3)
        torch.testing.assert_close(actual["weight"], torch.tensor([2.0, 4.0]))
        self.assertEqual(actual["counter"].item(), 11)
        torch.testing.assert_close(left["weight"], torch.tensor([1.0, 3.0]))

    def test_batch_norm_recalibration_uses_clean_images_and_disables_dropout(self):
        from torch.utils.data import DataLoader, TensorDataset
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Dropout(0.9))
        images = torch.arange(8).float().reshape(8, 1, 1, 1).expand(-1, 3, 4, 4)
        loader = DataLoader(TensorDataset(images, torch.zeros(8)), batch_size=4)
        original = model[0].weight.detach().clone()
        recalibrate_bn(model, loader, torch.device("cpu"))
        torch.testing.assert_close(model[0].running_mean, torch.full((3,), 3.5))
        torch.testing.assert_close(model[0].weight, original)
        self.assertEqual(model[0].num_batches_tracked.item(), 2)
        self.assertEqual(model[0].momentum, 0.1)
        self.assertFalse(any(module.training for module in model.modules()))

    def test_identical_images_stay_together_and_folds_are_disjoint(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = []
            for index in range(13):
                group = index if index < 12 else 0
                Image.new("RGB", (128, 128), (group * 17, 40, 80)).save(path / f"{index}.png")
                rows.append({"id": f"{index}.png", "crop_label": str(group % 3),
                             "health_label": str(group % 2)})
            data = SimpleNamespace(rows=rows, img_dir=path)
            result = grouped_folds(data, 3, 2026)
            assignments = {row["id"]: row["fold"] for row in result["rows"]}
            self.assertEqual(assignments["0.png"], assignments["12.png"])
            self.assertEqual(result["identical_image_groups"], 1)
            self.assertEqual(result, grouped_folds(data, 3, 2026))
            manifest = path / "folds.json"
            manifest.write_text(json.dumps(result))
            validation_union = set()
            for fold in range(3):
                train, valid = fold_indices(data, manifest, fold)
                self.assertFalse(set(train) & set(valid))
                self.assertEqual(set(train) | set(valid), set(range(13)))
                self.assertFalse(validation_union & set(valid))
                validation_union.update(valid)

    def test_paired_bootstrap_identical_models_zero_and_clear_improvement(self):
        labels = torch.tensor([[crop, health] for crop in range(3) for health in range(2) for _ in range(12)])
        self.assertEqual(paired_bootstrap(labels, labels, labels, 40)["ci95"], [0.0, 0.0])
        wrong = labels.clone()
        wrong[::3, 0] = (wrong[::3, 0] + 1) % 3
        result = paired_bootstrap(labels, wrong, labels, 100)
        self.assertGreater(result["delta"], 0)
        self.assertTrue(result["clear_valid_improvement"])
        self.assertEqual(result, paired_bootstrap(labels, wrong, labels, 100))

    def test_deep_model_standalone_submission_contract(self):
        self.assertEqual(checker.check_imports(str(PROJECT / "model.py")), [])
        model = MultiTaskCNN().eval()
        total = sum(p.numel() for p in model.parameters()) + sum(
            b.numel() for b in model.buffers() if b.is_floating_point())
        self.assertLessEqual(total, checker.MAX_PARAMS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.py").write_bytes((PROJECT / "model.py").read_bytes())
            spec = importlib.util.spec_from_file_location("isolated_deep_submission", path / "model.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            torch.save(model.state_dict(), path / "best_model.pt")
            self.assertLessEqual((path / "best_model.pt").stat().st_size, checker.MAX_WEIGHT_BYTES)
            restored = module.MultiTaskCNN().eval()
            restored.load_state_dict(torch.load(path / "best_model.pt", weights_only=True))
            with torch.no_grad():
                images = torch.randn(2, 3, 128, 128)
                expected = model(images)
                actual = restored(images)
            self.assertEqual([tuple(x.shape) for x in actual], [(2, 3), (2, 2)])
            for before, after in zip(expected, actual):
                torch.testing.assert_close(before, after, rtol=0, atol=0)


@unittest.skipUnless(os.environ.get("PLANT_DATA_DIR"), "Set PLANT_DATA_DIR to run real-data tests")
class TrainingIntegrationTests(unittest.TestCase):
    def tiny_data(self, root, include_valid=True, alter_valid=False):
        source = Path(os.environ["PLANT_DATA_DIR"]).resolve()
        root.mkdir()
        (root / "class_mapping.json").write_bytes((source / "class_mapping.json").read_bytes())
        for split, limit in (("train", 24), ("valid", 8)):
            if split == "valid" and not include_valid:
                continue
            (root / split).mkdir()
            (root / split / "images").symlink_to(source / split / "images", target_is_directory=True)
            with (source / f"{split}.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))[:limit]
            if split == "valid" and alter_valid:
                for row in rows:
                    row["crop_label"] = str((int(row["crop_label"]) + 1) % 3)
                    row["health_label"] = str(1 - int(row["health_label"]))
            with (root / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "crop_label", "health_label"])
                writer.writeheader()
                writer.writerows(rows)

    def execute(self, arguments):
        result = subprocess.run([sys.executable, "train.py", *arguments], cwd=PROJECT,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def test_default_run_auto_resumes_after_disconnect_and_rejects_live_duplicate(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(prefix="leaf-auto-resume-") as temporary:
            root = Path(temporary)
            work = root / "project"
            work.mkdir()
            self.tiny_data(root / "data")
            for name in ("model.py", "dataset.py", "train.py"):
                (work / name).write_bytes((PROJECT / name).read_bytes())
            options = ["--data_dir", str(root / "data"), "--epochs", "3", "--batch_size", "8",
                       "--num_workers", "0", "--device", "cpu"]
            wrapper = ("import train; original=train.atomic_save\n"
                       "def disconnect(value,path):\n"
                       " original(value,path)\n"
                       " if path.name == 'last_checkpoint.pt': raise SystemExit(75)\n"
                       "train.atomic_save=disconnect; train.main()\n")
            interrupted = subprocess.run([sys.executable, "-c", wrapper, *options], cwd=work,
                                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(interrupted.returncode, 75, interrupted.stdout)
            run = next((work / "runs").glob("single_*"))
            config = json.loads((run / "config.json").read_text())
            args = SimpleNamespace(**config["args"])
            with run_lock(run):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    find_resumable_run(args, work, work / "runs")
            resumed = subprocess.run([sys.executable, "train.py", *options], cwd=work,
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            self.assertIn("Auto-resuming", resumed.stdout)
            self.assertEqual(len(list((work / "runs").glob("single_*"))), 1)
            checkpoint = torch.load(run / "last_checkpoint.pt", weights_only=True)
            self.assertEqual([row["epoch"] for row in checkpoint["history"]], [1, 2, 3])
            self.assertGreater(checkpoint["swa_count"], 0)
            self.assertEqual(checkpoint["args"]["sam_rho"], 0.03)
            self.assertIsNotNone(checkpoint["history"][0]["raw_score"])
            self.assertTrue((run / "completed.json").is_file())
            self.assertIsNone(find_resumable_run(args, work, work / "runs"))

    def test_early_stopped_run_saves_checkpoint_and_resume_does_not_train_more(self):
        with tempfile.TemporaryDirectory(prefix="leaf-early-stop-") as temporary:
            root = Path(temporary)
            self.tiny_data(root / "data")
            run = root / "run"
            options = ["--data_dir", str(root / "data"), "--run_dir", str(run), "--epochs", "10",
                       "--batch_size", "8", "--num_workers", "0", "--device", "cpu",
                       "--finish_fraction", "0.5", "--patience", "1", "--min_delta", "1.0"]
            result = self.execute(options)
            self.assertIn("Early stopping after epoch 6", result.stdout)
            checkpoint = torch.load(run / "last_checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 6)
            self.assertTrue(checkpoint["early_stop"]["stopped"])
            before = torch.load(run / "best_model.pt", weights_only=True)
            self.execute(options + ["--resume"])
            self.assertEqual(torch.load(run / "last_checkpoint.pt", weights_only=True)["epoch"], 6)
            after = torch.load(run / "best_model.pt", weights_only=True)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_adaptive_polish_transition_and_disconnect_resume_match(self):
        with tempfile.TemporaryDirectory(prefix="leaf-adaptive-") as temporary:
            root = Path(temporary)
            self.tiny_data(root / "data")
            options = ["--data_dir", str(root / "data"), "--epochs", "0", "--schedule", "adaptive",
                       "--batch_size", "8", "--num_workers", "0", "--device", "cpu",
                       "--warmup_epochs", "0", "--lr", "0.001", "--min_lr", "0.0005",
                       "--lr_patience", "1", "--polish_min_epochs", "2", "--patience", "1",
                       "--no_bn_recalibration"]
            # Synthetic plateau scores test the controller, not model accuracy; real images train the CNN.
            wrapper = ("import train\n"
                       "def plateau(*unused):\n"
                       " return {'score':0.5,'crop_f1':0.5,'health_f1':0.5,"
                       "'crop_f1_per_class':[0.5]*3,'health_f1_per_class':[0.5]*2,"
                       "'crop_confusion':[[0]*3 for _ in range(3)],'health_confusion':[[0]*2 for _ in range(2)]}\n"
                       "train.evaluate=plateau\n")
            disconnect = ("original=train.atomic_save\n"
                          "def interrupt(value,path):\n"
                          " original(value,path)\n"
                          " if path.name=='last_checkpoint.pt' and value['epoch']==3: raise SystemExit(75)\n"
                          "train.atomic_save=interrupt\n")
            runs = [root / "continuous", root / "resumed"]
            def execute(run, pause=False, resume=False):
                code = wrapper + (disconnect if pause else "") + "train.main()\n"
                result = subprocess.run([sys.executable, "-c", code, *options, "--run_dir", str(run),
                                         *(["--resume"] if resume else [])], cwd=PROJECT,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 75 if pause else 0, result.stdout)
            execute(runs[0])
            execute(runs[1], pause=True)
            halfway = torch.load(runs[1] / "last_checkpoint.pt", weights_only=True)
            self.assertEqual(halfway["adaptive_state"]["phase"], "polish")
            self.assertEqual(halfway["optimizer"]["state"], {})
            execute(runs[1], resume=True)
            records = [torch.load(run / "last_checkpoint.pt", weights_only=True) for run in runs]
            for record in records:
                self.assertEqual(record["epoch"], 5)
                self.assertEqual(record["adaptive_state"]["phase"], "done")
                self.assertTrue(record["early_stop"]["stopped"])
                self.assertEqual(record["history"][2]["transition"], "start_polish")
                self.assertEqual(record["history"][3]["augmentation"], "finish")
                self.assertEqual(record["history"][3]["sam_rho"], 0.01)
            for group in ("model", "ema"):
                for key in records[0][group]:
                    torch.testing.assert_close(records[0][group][key], records[1][group][key], rtol=0, atol=0)
            before = torch.load(runs[1] / "best_model.pt", weights_only=True)
            execute(runs[1], resume=True)
            after = torch.load(runs[1] / "best_model.pt", weights_only=True)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_internal_fold_training_does_not_need_official_valid(self):
        with tempfile.TemporaryDirectory(prefix="leaf-fold-") as directory:
            root = Path(directory)
            data, run = root / "data", root / "run"
            self.tiny_data(data, include_valid=False)
            manifest = root / "folds.json"
            manifest.write_text(json.dumps(grouped_folds(PlantDataset(data, "train"), 2, 2026)))
            options = ["--data_dir", str(data), "--run_dir", str(run), "--fold_manifest", str(manifest),
                       "--fold_index", "0", "--fixed_epochs", "--epochs", "1", "--batch_size", "4",
                       "--num_workers", "0", "--device", "cpu"]
            self.execute(options)
            frozen = json.loads((run / "frozen_selection.json").read_text())
            self.assertEqual(frozen["validation_source"], "train_fold")
            self.assertFalse((data / "valid.csv").exists())
            before = torch.load(run / "best_model.pt", weights_only=True)
            self.execute(options + ["--resume"])
            after = torch.load(run / "best_model.pt", weights_only=True)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_official_validation_labels_cannot_change_fixed_refit_weights(self):
        with tempfile.TemporaryDirectory(prefix="leaf-frozen-") as directory:
            root = Path(directory)
            runs = []
            for index in range(2):
                data, run = root / f"data{index}", root / f"run{index}"
                self.tiny_data(data, alter_valid=bool(index))
                self.execute(["--data_dir", str(data), "--run_dir", str(run), "--fixed_epochs",
                              "--epochs", "2", "--batch_size", "4", "--num_workers", "0", "--device", "cpu"])
                manifest = json.loads((run / "frozen_selection.json").read_text())
                self.assertFalse(manifest["official_valid_used_for_selection"])
                self.assertEqual(manifest["weights_sha256_before_validation"],
                                 hashlib.sha256((run / "best_model.pt").read_bytes()).hexdigest())
                history = json.loads((run / "history.json").read_text())
                self.assertTrue(all(row["raw_score"] is None and row["ema_score"] is None for row in history))
                runs.append(run)
            before, after = [torch.load(run / "best_model.pt", weights_only=True) for run in runs]
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
            # Read-only comparison uses saved model sources; identical models have exactly zero delta.
            self.execute(["--data_dir", str(root / "data0"), "--run_dir", str(root / "audit"),
                          "--compare_runs", *map(str, runs), "--bootstrap", "30", "--num_workers", "0"])
            report = json.loads((root / "audit/comparison.json").read_text())
            self.assertEqual(report[1]["paired_vs_first"]["ci95"], [0.0, 0.0])
            for run in runs:
                self.assertTrue((root / "audit" / f"errors_{runs.index(run)}_{run.name}.csv").is_file())

    def test_cross_validation_selects_from_train_and_exports_fixed_model(self):
        with tempfile.TemporaryDirectory(prefix="leaf-cv-") as directory:
            root = Path(directory)
            data, run = root / "data", root / "cv"
            self.tiny_data(data)
            options = ["--cv", "--data_dir", str(data), "--run_dir", str(run), "--cv_folds", "2",
                       "--cv_epochs", "1", "--epochs", "2", "--batch_size", "4", "--num_workers", "0",
                       "--device", "cpu"]
            self.execute(options)
            decision = json.loads((run / "cv_selection.json").read_text())
            self.assertFalse(decision["official_valid_used_for_selection"])
            self.assertFalse(decision["public_used_for_selection"])
            self.assertEqual(decision["refit_epochs"], 1)
            results = json.loads((run / "cv_results.json").read_text())
            self.assertEqual(len(results), 3)
            for recipe in results:
                self.assertEqual(len(recipe["folds"]), 2)
                for fold in range(2):
                    child = run / recipe["name"] / f"fold{fold}"
                    config = json.loads((child / "config.json").read_text())
                    self.assertIsNone(config["args"]["init_weights"])
                    self.assertTrue(config["args"]["fixed_epochs"])
                    frozen = json.loads((child / "frozen_selection.json").read_text())
                    self.assertEqual(frozen["variant"], "swa" if recipe["name"] == "multiscale" else "ema")
            model = MultiTaskCNN()
            model.load_state_dict(torch.load(run / "best_model.pt", weights_only=True))
            frozen = json.loads((run / "frozen_selection.json").read_text())
            self.assertEqual(frozen["validation_source"], "official_valid")
            before = torch.load(run / "best_model.pt", weights_only=True)
            self.execute(options + ["--resume"])
            after = torch.load(run / "best_model.pt", weights_only=True)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_search_exports_one_model_retains_starting_score_and_resumes(self):
        from torch.utils.data import DataLoader, Subset
        data = Path(os.environ["PLANT_DATA_DIR"]).resolve()
        with tempfile.TemporaryDirectory(prefix="leaf-search-") as directory:
            root = Path(directory)
            parent = root / "parent"
            search = root / "search"
            common = ["--data_dir", str(data), "--epochs", "2", "--batch_size", "4",
                      "--train_limit", "8", "--valid_limit", "8", "--num_workers", "0",
                      "--device", "cpu", "--mixup_prob", "1", "--finish_fraction", "0.5",
                      "--swa_start", "0.5"]

            def execute(extra):
                result = subprocess.run([sys.executable, "train.py", *common, *extra], cwd=PROJECT,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 0, result.stdout)

            execute(["--run_dir", str(parent)])
            # Simulate the preceding architecture's plain state_dict from a real-data smoke run.
            weights = torch.load(parent / "best_model.pt", weights_only=True)
            weights.pop("detail_projection.weight")
            initial = root / "old_deep.pt"
            torch.save(weights, initial)
            arguments = ["--search", "--run_dir", str(search), "--init_weights", str(initial)]
            execute(arguments)
            results = json.loads((search / "search_results.json").read_text())
            self.assertEqual(len(results["candidates"]), 4)
            self.assertTrue(results["smoke_run"])
            metrics = json.loads((search / "best_metrics.json").read_text())
            self.assertGreaterEqual(metrics["score"], max(row["score"] for row in results["candidates"]))
            saved = torch.load(search / "best_model.pt", weights_only=True)
            model = MultiTaskCNN().eval()
            model.load_state_dict(saved)
            dataset = Subset(PlantDataset(data, "valid", get_transforms(False)), range(8))
            actual = evaluate(model, DataLoader(dataset, batch_size=4), torch.device("cpu"))
            self.assertEqual(actual["score"], metrics["score"])
            for child in search.glob("trial*"):
                history = json.loads((child / "history.json").read_text())
                self.assertEqual(history[-1]["augmentation"], "finish")
                checkpoint = torch.load(child / "last_checkpoint.pt", weights_only=True)
                self.assertGreater(checkpoint["swa_count"], 0)
                self.assertFalse((child / "model_deep.py").exists())
            execute(arguments + ["--resume"])
            final = torch.load(search / "best_model.pt", weights_only=True)
            for key in saved:
                torch.testing.assert_close(saved[key], final[key], rtol=0, atol=0)

    def test_three_files_train_and_resume_without_extra_model_file(self):
        with tempfile.TemporaryDirectory(prefix="leaf-deep-") as directory:
            run = Path(directory) / "run"
            work = Path(directory) / "project"
            work.mkdir()
            for name in ("model.py", "dataset.py", "train.py"):
                (work / name).write_bytes((PROJECT / name).read_bytes())
            data = Path(os.environ["PLANT_DATA_DIR"]).resolve()
            arguments = ["--data_dir", str(data), "--run_dir", str(run),
                         "--epochs", "1", "--batch_size", "4", "--train_limit", "8",
                         "--valid_limit", "8", "--num_workers", "0", "--device", "cpu"]
            result = subprocess.run([sys.executable, "train.py", *arguments], cwd=work,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual((run / "model.py").read_bytes(), (PROJECT / "model.py").read_bytes())
            self.assertFalse((run / "model_deep.py").exists())
            restored = MultiTaskCNN()
            restored.load_state_dict(torch.load(run / "best_model.pt", weights_only=True))
            # Both training and retained run startup need only the three source files.
            resumed = subprocess.run([sys.executable, "train.py", *arguments, "--resume"], cwd=run,
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(resumed.returncode, 0, resumed.stdout)

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
