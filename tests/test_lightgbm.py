"""LightGBM interface, preprocessing, and persistence checks."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import Booster

from src.data import FEATURES, build_dataset
from src.models.lightgbm import fit_predict


ROOT = Path(__file__).resolve().parents[1]


class LightGBMTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 150
        raw = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n)})
        for prefix in ("", "nasdaq_"):
            close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
            raw[prefix + "open"] = close * 0.995
            raw[prefix + "high"] = close * 1.01
            raw[prefix + "low"] = close * 0.99
            raw[prefix + "close"] = close
            raw[prefix + "volume"] = rng.integers(1000, 5000, n)
        config = {"window": 12, "start": "2020-01-01",
                  "train_end": str(raw.date.iloc[99].date()),
                  "validation_end": str(raw.date.iloc[-1].date())}
        self.dataset = build_dataset(raw, config)
        self.settings = {"n_estimators": 40, "learning_rate": 0.05,
                         "num_leaves": 7, "min_child_samples": 5,
                         "early_stopping_rounds": 5, "n_jobs": 1, "verbosity": -1}

    def test_interface_and_lag_order(self):
        settings = copy.deepcopy(self.settings)
        model, prediction, details = fit_predict(self.dataset, settings, seed=42)
        self.assertEqual(settings, self.settings)
        self.assertEqual(prediction.shape, self.dataset.validation.y.shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(details["n_features"], 12 * 8)
        self.assertEqual(details["feature_names"][:8], [f"lag_11_{f}" for f in FEATURES])
        self.assertEqual(details["feature_names"][-8:], [f"lag_0_{f}" for f in FEATURES])
        best = details["best_iteration"]
        self.assertGreaterEqual(best, 1)
        history = details["history"]["validation"]["rmse"]
        self.assertAlmostEqual(history[best - 1], min(history))
        self.assertEqual(model.n_features_, 96)
        json.dumps(details, allow_nan=False)

    def test_prediction_scale_and_saved_booster(self):
        model, prediction, details = fit_predict(self.dataset, self.settings, seed=42)
        validation = self.dataset.transform(self.dataset.validation.X).reshape(
            len(self.dataset.validation.y), -1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lightgbm.txt"
            model.booster_.save_model(str(path), num_iteration=details["best_iteration"])
            restored = Booster(model_file=str(path))
            actual = (restored.predict(validation) * self.dataset.target_scale
                      + self.dataset.target_mean)
            np.testing.assert_allclose(actual, prediction, rtol=1e-9, atol=1e-12)
        rmse = float(np.sqrt(np.mean((prediction - self.dataset.validation.y) ** 2)))
        scaled = details["history"]["validation"]["rmse"][details["best_iteration"] - 1]
        self.assertAlmostEqual(rmse, scaled * self.dataset.target_scale, places=7)

    def test_reproducible_seed(self):
        _, first, _ = fit_predict(self.dataset, self.settings, seed=42)
        _, second, _ = fit_predict(self.dataset, self.settings, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_without_early_stopping_records_actual_tree_count(self):
        settings = dict(self.settings, early_stopping_rounds=0, n_estimators=4)
        model, _, details = fit_predict(self.dataset, settings, seed=42)
        self.assertEqual(details["best_iteration"], model.booster_.current_iteration())
        self.assertEqual(details["best_iteration"], 4)

    def test_common_experiment_rules_unchanged(self):
        common = json.loads((ROOT / "configs/aapl.json").read_text())
        ours = json.loads((ROOT / "configs/aapl_lightgbm.json").read_text())
        for key in ("ticker", "data_path", "start", "train_end", "validation_end",
                    "test_end", "holdout_end", "window", "seed"):
            self.assertEqual(common[key], ours[key])

    def test_help_does_not_import_other_model_runtimes(self):
        code = (
            "import sys; from scripts.run_experiment import main; "
            "sys.argv=['main.py','--help']; "
            "\ntry: main()\nexcept SystemExit as e: assert e.code == 0\n"
            "assert 'torch' not in sys.modules; assert 'xgboost' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lightgbm", result.stdout)


if __name__ == "__main__":
    unittest.main()
