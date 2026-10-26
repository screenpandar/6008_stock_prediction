"""Restore selected checkpoints and apply saved preprocessing; no training."""
from pathlib import Path
import time

import numpy as np

from src.data import selected_features


MODEL_FILES = {
    "xgboost": "xgboost.ubj", "lightgbm": "lightgbm.txt",
    "patchtst": "patchtst.pt", "lstm": "lstm.pt", "gru": "gru.pt",
}


def validate_preprocessing(config, preprocessing):
    features = selected_features(config)
    if (preprocessing["features"] != features
            or preprocessing["window"] != config["window"]):
        raise ValueError("Saved feature order/window disagrees with source configuration")
    mean = np.asarray(preprocessing["feature_mean"], dtype=np.float64)
    scale = np.asarray(preprocessing["feature_scale"], dtype=np.float64)
    if (mean.shape != (len(features),) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not (scale > 0).all()):
        raise ValueError("Invalid saved feature scaling")
    if (not np.isfinite(preprocessing["target_mean"])
            or not np.isfinite(preprocessing["target_scale"])
            or preprocessing["target_scale"] <= 0):
        raise ValueError("Invalid saved target scaling")


class FrozenModels:
    """Five saved models, with their selected tree prefixes / neural weights."""

    def __init__(self, folder, config, preprocessing, training, device="auto"):
        import torch
        import xgboost as xgb
        from lightgbm import Booster
        from src.models.patchtst import PatchTSTRegressor
        from src.models.lstm import LSTMRegressor
        from src.models.gru import GRURegressor

        validate_preprocessing(config, preprocessing)
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.preprocessing = preprocessing
        self.window, self.channels = config["window"], len(preprocessing["features"])
        folder = Path(folder)
        self.models = {}
        tree = xgb.Booster()
        tree.load_model(folder / MODEL_FILES["xgboost"])
        tree.set_param({"device": "cpu", "nthread": 4})
        if tree.num_boosted_rounds() != training["xgboost"]["best_iteration"] + 1:
            raise ValueError("XGBoost checkpoint is not the selected tree prefix")
        self.models["xgboost"] = tree
        tree = Booster(model_file=str(folder / MODEL_FILES["lightgbm"]))
        if tree.current_iteration() != training["lightgbm"]["best_iteration"]:
            raise ValueError("LightGBM checkpoint is not the selected tree prefix")
        self.models["lightgbm"] = tree
        if (self.models["xgboost"].num_features() != self.window * self.channels
                or tree.num_feature() != self.window * self.channels):
            raise ValueError("Saved tree input shape disagrees with preprocessing")
        factories = {"patchtst": PatchTSTRegressor, "lstm": LSTMRegressor, "gru": GRURegressor}
        for name, factory in factories.items():
            checkpoint = torch.load(folder / MODEL_FILES[name], map_location="cpu", weights_only=True)
            kwargs = checkpoint["model_kwargs"]
            if kwargs != training[name]["model_kwargs"]:
                raise ValueError(f"{name}: checkpoint architecture disagrees with training.json")
            if name == "lstm":
                if kwargs["input_size"] != self.channels or checkpoint["sequence_length"] != self.window:
                    raise ValueError("LSTM checkpoint input shape disagrees with preprocessing")
            elif kwargs["channels"] != self.channels or kwargs["window"] != self.window:
                raise ValueError(f"{name}: checkpoint input shape disagrees with preprocessing")
            model = factory(**kwargs)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(self.device).eval()
            model.requires_grad_(False)
            self.models[name] = model

    def predict(self, X, batch_size=128):
        """Apply training-time scales, predict, and invert target scaling."""
        import torch
        import xgboost as xgb

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3 or X.shape[1:] != (self.window, self.channels) or not len(X):
            raise ValueError("Expected nonempty [samples, saved window, saved channels]")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        p = self.preprocessing
        scaled = ((X - np.asarray(p["feature_mean"])) / np.asarray(p["feature_scale"])).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Non-finite standardized inputs")
        flat = scaled.reshape(len(X), -1)
        predictions, details = {}, {}
        for name, model in self.models.items():
            started = time.perf_counter()
            if name == "xgboost":
                values = model.predict(xgb.DMatrix(flat))
            elif name == "lightgbm":
                values = model.predict(flat, num_iteration=model.current_iteration(), num_threads=4)
            else:
                chunks = []
                with torch.inference_mode():
                    for i in range(0, len(scaled), batch_size):
                        inputs = torch.from_numpy(scaled[i:i + batch_size]).to(self.device)
                        chunks.append(model(inputs).cpu().numpy())
                values = np.concatenate(chunks)
            values = np.asarray(values, dtype=np.float64) * p["target_scale"] + p["target_mean"]
            if values.shape != (len(X),) or not np.isfinite(values).all():
                raise ValueError(f"{name}: invalid predictions")
            predictions[name] = values
            details[name] = {"seconds": time.perf_counter() - started,
                             "device": "cpu" if name in ("xgboost", "lightgbm") else self.device}
        return predictions, details
