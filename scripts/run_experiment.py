"""Validation-only experiment entry / 仅训练与验证的实验入口。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
from datetime import datetime, timezone

# Set before importing torch / 必须在初始化 CUDA 前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.baselines import predict_baselines
from src.data import FEATURES, load_dataset
from src.evaluate import comparison_table, evaluate, plot_comparison


def save_json(path, value):
    """Strict JSON prevents hidden NaN metrics / 严格 JSON 防止无效数值悄悄进入结果。"""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def package_versions(names):
    """Record optional dependencies without requiring every model runtime."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run(config, output, models):
    """Run a reproducible validation experiment / 执行可追溯的验证集实验。"""
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch = None
    if "patchtst" in models:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PatchTST requires torch; run LightGBM alone or install torch") from exc
        torch.manual_seed(seed)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dataset = load_dataset(config, ROOT)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    save_json(output / "data_audit.json", dataset.audit)
    save_json(output / "preprocessing.json", {
        "features": FEATURES, "window": config["window"],
        "feature_mean": dataset.feature_mean.tolist(), "feature_scale": dataset.feature_scale.tolist(),
        "target_mean": dataset.target_mean, "target_scale": dataset.target_scale,
    })
    print(json.dumps(dataset.audit, indent=2), flush=True)
    manifest = {
        "status": "running", "evaluation_split": "validation",
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "executable": sys.executable, "seed": seed, "models_requested": models,
        "data_sha256": hashlib.sha256((ROOT / config["data_path"]).read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / "main.py", *sorted((ROOT / "src").rglob("*.py")),
                                    *sorted((ROOT / "scripts").rglob("*.py"))]},
        "packages": package_versions(
            ("torch", "numpy", "pandas", "scikit-learn", "matplotlib", "xgboost", "lightgbm")),
        "cuda_build": torch.version.cuda if torch is not None else None,
        "gpu": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None,
        "test_evaluated": False, "holdout_evaluated": False,
    }
    save_json(output / "manifest.json", manifest)
    try:
        predictions = predict_baselines(dataset.train, dataset.validation)
        training = {}
        if "xgboost" in models:
            from src.models.xgboost import fit_predict
            model, predictions["xgboost"], training["xgboost"] = fit_predict(
                dataset, config["xgboost"], seed)
            # Persist only the selected tree prefix / 保存最佳树前缀，避免重载时使用过拟合尾部。
            model.get_booster()[:model.best_iteration + 1].save_model(output / "xgboost.ubj")
            print("XGBoost finished", flush=True)
        if "lightgbm" in models:
            from src.models.lightgbm import fit_predict
            model, predictions["lightgbm"], training["lightgbm"] = fit_predict(
                dataset, config["lightgbm"], seed)
            model.booster_.save_model(str(output / "lightgbm.txt"),
                                      num_iteration=training["lightgbm"]["best_iteration"])
            print("LightGBM finished", flush=True)
        if "patchtst" in models:
            from src.models.patchtst import fit_predict
            model, predictions["patchtst"], training["patchtst"] = fit_predict(
                dataset, config["patchtst"], seed)
            torch.save({"model_kwargs": training["patchtst"]["model_kwargs"],
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       output / "patchtst.pt")
        metrics = {name: evaluate(dataset.validation.y, p) for name, p in predictions.items()}
        # Majority label is fitted on train; do not invent a return magnitude for it.
        majority_up = bool((dataset.train.y > 0).mean() > 0.5)
        direction_reference = {
            "train_majority_up": majority_up,
            "validation_accuracy": float(np.mean((dataset.validation.y > 0) == majority_up)),
        }
        rows = dataset.validation.dates.copy()
        rows["actual_return"] = dataset.validation.y
        rows["actual_direction"] = (dataset.validation.y > 0).astype(int)
        for name, values in predictions.items():
            rows[name] = values
        rows.to_csv(output / "predictions.csv", index=False)
        save_json(output / "metrics.json", metrics)
        save_json(output / "direction_reference.json", direction_reference)
        save_json(output / "training.json", training)
        table = comparison_table(metrics)
        table.to_csv(output / "metrics.csv")
        fig = plot_comparison(metrics)
        fig.savefig(output / "validation_comparison.png", dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        manifest["status"] = "complete"
        print(table.to_string(), flush=True)
        print(f"Results: {output}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(output / "manifest.json", manifest)
    return output


def main():
    """CLI entry, never reads test labels / 命令行入口不构造测试期标签。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/aapl.json",
                        help="JSON configuration (default: configs/aapl.json)")
    parser.add_argument("--models", nargs="+", choices=["baselines", "xgboost", "lightgbm", "patchtst"],
                        default=["xgboost", "patchtst"],
                        help="Models to train; baselines are always included")
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%S_%fZ"),
                        help="New folder name under artifacts/; must not already exist")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("run-name must contain only letters, digits, underscores and hyphens")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    run(config, ROOT / "artifacts" / args.run_name, args.models)


if __name__ == "__main__":
    main()
