"""LightGBM adapter for the shared next-day return task / LightGBM 回归适配器。"""
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from src.data import FEATURES


def _feature_names(window, features):
    """Name flattened values by relative day and shared feature."""
    return [
        f"lag_{window - 1 - day}_{feature}"
        for day in range(window)
        for feature in features
    ]


def fit_predict(dataset, settings, seed):
    """Fit on train and select the tree prefix using validation RMSE.

    LightGBM receives the same standardized 60-day windows as every other
    model, flattened in chronological order. Predictions are returned in the
    original decimal-return scale required by ``src.evaluate``.
    """
    started = time.perf_counter()
    if (dataset.train.X.ndim != 3 or dataset.validation.X.ndim != 3
            or dataset.train.X.shape[1:] != dataset.validation.X.shape[1:]
            or dataset.train.X.shape[2] != len(FEATURES)):
        raise ValueError("Expected aligned [samples, window, 8] feature windows")
    window = dataset.train.X.shape[1]
    channels = dataset.train.X.shape[2]
    train = dataset.transform(dataset.train.X).reshape(len(dataset.train.y), -1)
    validation = dataset.transform(dataset.validation.X).reshape(len(dataset.validation.y), -1)
    train_y = (dataset.train.y - dataset.target_mean) / dataset.target_scale
    validation_y = (dataset.validation.y - dataset.target_mean) / dataset.target_scale

    model_settings = dict(settings)
    early_stopping_rounds = int(model_settings.pop("early_stopping_rounds", 0))
    verbosity = int(model_settings.pop("verbosity", -1))
    if early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds must be nonnegative")
    model = LGBMRegressor(
        objective="regression",
        metric="rmse",
        random_state=seed,
        verbosity=verbosity,
        deterministic=True,
        force_col_wise=True,
        **model_settings,
    )
    callbacks = [log_evaluation(0)]
    if early_stopping_rounds > 0:
        callbacks.append(early_stopping(early_stopping_rounds, first_metric_only=True,
                                       verbose=False))
    names = _feature_names(window, FEATURES)
    train_frame = pd.DataFrame(train, columns=names)
    validation_frame = pd.DataFrame(validation, columns=names)
    model.fit(
        train_frame,
        train_y,
        eval_set=[(train_frame, train_y), (validation_frame, validation_y)],
        eval_names=["train", "validation"],
        feature_name=names,
        callbacks=callbacks,
    )
    best_iteration = int(model.best_iteration_ or model.booster_.current_iteration())
    prediction = model.predict(validation_frame, num_iteration=best_iteration)
    prediction = prediction.astype(np.float64) * dataset.target_scale + dataset.target_mean
    if not np.isfinite(prediction).all():
        raise FloatingPointError("LightGBM returned non-finite predictions")

    history = model.evals_result_
    details = {
        "seconds": time.perf_counter() - started,
        "device": "cpu",
        "best_iteration": best_iteration,
        "iterations_run": len(history["validation"]["rmse"]),
        "n_features": int(train.shape[1]),
        "window": int(window),
        "channels": int(channels),
        "feature_names": names,
        "target_mean": dataset.target_mean,
        "target_scale": dataset.target_scale,
        "history_units": "standardized target RMSE; multiply by target_scale for decimal return RMSE",
        "history": {split: {metric: [float(value) for value in values]
                            for metric, values in metrics.items()}
                    for split, metrics in history.items()},
        "parameters": model.get_params(),
    }
    return model, prediction, details
