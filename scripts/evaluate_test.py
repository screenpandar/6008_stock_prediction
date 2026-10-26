"""Final test using saved five-model checkpoints; never fit or tune a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.run_experiment import package_versions, save_json
from src.data import build_evaluation_samples
from src.evaluate import comparison_table, evaluate, plot_comparison
from src.inference import FrozenModels, MODEL_FILES, validate_preprocessing


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_path(root, name):
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError("Use a directory name under artifacts, not a full path")
    return Path(root) / "artifacts" / name


def load_source(folder):
    """Fail on incomplete or mismatched snapshots before accessing test targets."""
    required = ["config.json", "manifest.json", "preprocessing.json", "training.json",
                "direction_reference.json", "predictions.csv", *MODEL_FILES.values()]
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{folder.name}: missing {', '.join(missing)}")
    source = {key: read_json(folder / f"{key}.json") for key in
              ("config", "manifest", "preprocessing", "training", "direction_reference")}
    manifest, config = source["manifest"], source["config"]
    if (manifest.get("status") != "complete" or manifest.get("evaluation_split") != "validation"
            or manifest.get("test_evaluated", False) or manifest.get("holdout_evaluated", False)):
        raise ValueError("Source must be a completed validation experiment")
    if not set(MODEL_FILES).issubset(manifest.get("models_requested", [])):
        raise ValueError("Source must contain all five trained models")
    if any(name not in source["training"] or name not in config for name in MODEL_FILES):
        raise ValueError("Source training/configuration records are incomplete")
    if not config.get("ticker") or manifest.get("seed") != config.get("seed"):
        raise ValueError("Source ticker or seed records are invalid")
    if not isinstance(source["direction_reference"].get("train_majority_up"), bool):
        raise ValueError("Source is missing the training-fitted majority direction")
    validate_preprocessing(config, source["preprocessing"])
    source["file_sha256"] = {name: sha256(folder / name) for name in required}
    return source


def replay_validation(raw, source, folder, models, batch_size):
    """Check restored predictions on the already-used validation split only."""
    samples, _ = build_evaluation_samples(raw, source["config"], split="validation")
    saved = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
    for column in samples.dates:
        np.testing.assert_array_equal(samples.dates[column], pd.to_datetime(saved[column]),
                                       err_msg=f"Validation {column} changed")
    np.testing.assert_allclose(samples.y, saved.actual_return, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(saved.actual_direction, (samples.y > 0).astype(int))
    p = source["preprocessing"]
    baselines = {"zero_return": np.zeros(len(samples.y)),
                 "historical_mean": np.full(len(samples.y), p["target_mean"]),
                 "last_return": samples.last_return}
    predictions, _ = models.predict(samples.X, batch_size)
    maximum_errors = {}
    for name, values in {**baselines, **predictions}.items():
        # CPU/GPU and batch-size changes can cause small floating-point differences.
        np.testing.assert_allclose(values, saved[name], rtol=1e-5, atol=1e-7,
                                   err_msg=f"{name}: restored validation predictions do not match")
        maximum_errors[name] = float(np.max(np.abs(values - saved[name].to_numpy())))
    return {"status": "passed", "samples": len(samples.y), "rtol": 1e-5, "atol": 1e-7,
            "max_absolute_return_difference": maximum_errors}


def run_test(source_name, run_name=None, *, device="auto", batch_size=128,
             check_only=False, root=ROOT):
    root = Path(root)
    folder = artifact_path(root, source_name)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if check_only and run_name is not None:
        raise ValueError("--check-only does not create results; omit --run-name")
    output = None
    if not check_only:
        name = run_name or f"{source_name}_test_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}"
        output = artifact_path(root, name)
        if output.exists():
            raise FileExistsError(f"Output already exists; choose a new --run-name: {output}")
    source = load_source(folder)
    config, preprocessing = source["config"], source["preprocessing"]
    data_path = root / config["data_path"]
    data_hash = sha256(data_path)
    if data_hash != source["manifest"].get("data_sha256"):
        raise ValueError("Raw data differs from the saved experiment; restore the matching CSV")
    raw = pd.read_csv(data_path)
    models = FrozenModels(folder, config, preprocessing, source["training"], device)
    replay = replay_validation(raw, source, folder, models, batch_size)
    print(f"{config['ticker']}: five saved models restored; validation replay passed.", flush=True)
    if check_only:
        print("Check only: no test samples evaluated, no output directory created.", flush=True)
        return None
    samples, audit = build_evaluation_samples(raw, config, split="test")
    output.mkdir(parents=True, exist_ok=False)
    import torch
    manifest = {
        "status": "running", "evaluation_split": "test", "source_run": source_name,
        "source_created_utc": source["manifest"].get("created_utc"),
        "source_artifact_sha256": source["file_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "executable": sys.executable, "seed": config["seed"], "models_requested": list(MODEL_FILES),
        "data_sha256": data_hash, "retrained": False, "preprocessing_fitted": False,
        "test_evaluated": False, "holdout_evaluated": False,
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in
                          [root / "main.py", *sorted((root / "src").rglob("*.py")),
                           *sorted((root / "scripts").rglob("*.py"))]},
        "packages": package_versions(("torch", "numpy", "pandas", "scikit-learn", "matplotlib",
                                      "xgboost", "lightgbm")),
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if models.device == "cuda" else None,
        "inference_device": models.device, "inference_batch_size": batch_size,
    }
    save_json(output / "manifest.json", manifest)
    try:
        for key in ("config", "preprocessing"):
            save_json(output / f"{key}.json", source[key])
        save_json(output / "source_manifest.json", source["manifest"])
        save_json(output / "validation_replay.json", replay)
        save_json(output / "data_audit.json", audit)
        predictions, inference = models.predict(samples.X, batch_size)
        predictions = {
            "zero_return": np.zeros(len(samples.y)),
            "historical_mean": np.full(len(samples.y), preprocessing["target_mean"]),
            "last_return": samples.last_return.copy(), **predictions,
        }
        metrics = {name: evaluate(samples.y, values) for name, values in predictions.items()}
        manifest["test_evaluated"] = True
        rows = samples.dates.copy()
        rows["actual_return"], rows["actual_direction"] = samples.y, (samples.y > 0).astype(int)
        for name, values in predictions.items():
            rows[name] = values
        rows.to_csv(output / "predictions.csv", index=False)
        majority = source["direction_reference"]["train_majority_up"]
        save_json(output / "direction_reference.json", {
            "train_majority_up": majority, "test_accuracy": float(np.mean((samples.y > 0) == majority)),
        })
        save_json(output / "metrics.json", metrics)
        save_json(output / "inference.json", {"source_run": source_name, "retrained": False,
                                               "models": inference})
        table = comparison_table(metrics)
        table.to_csv(output / "metrics.csv")
        fig = plot_comparison(metrics, split="Test")
        fig.savefig(output / "test_comparison.png", dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        manifest["status"] = "complete"
        print(f"Test: {audit['target_first']} through {audit['target_last']} ({len(samples.y)} samples)")
        print(table.to_string())
        print(f"Results: {output}")
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(output / "manifest.json", manifest)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True,
                        help="Saved five-model validation directory under artifacts/")
    parser.add_argument("--run-name", help="New output directory under artifacts/ (never overwritten)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Neural inference device; tree models use CPU")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size only")
    parser.add_argument("--check-only", action="store_true",
                        help="Restore and replay validation only; do not evaluate test or write outputs")
    args = parser.parse_args(argv)
    run_test(args.source_run, args.run_name, device=args.device, batch_size=args.batch_size,
             check_only=args.check_only)


if __name__ == "__main__":
    main()
