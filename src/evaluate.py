"""Shared metrics and plots / 统一评价与绘图。"""
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support,
                             r2_score)


def evaluate(y, prediction):
    """Returns in decimal units; direction threshold=0 / 收益为小数，方向阈值为零。"""
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.ndim != 1 or prediction.shape != y.shape or len(y) < 2:
        raise ValueError("Need aligned one-dimensional arrays of length >=2")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics cannot accept missing or infinite returns")
    true_up, predicted_up = y > 0, prediction > 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_up, predicted_up, average="binary", zero_division=0)
    sse = float(np.sum((y - prediction) ** 2))
    zero_sse = float(np.sum(y ** 2))
    return {
        "n": len(y), "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)) if np.var(y) > 0 else None,
        "r2_vs_zero": 1 - sse / zero_sse if zero_sse > 0 else None,
        "direction_accuracy": float(np.mean(true_up == predicted_up)),
        "up_precision": float(precision), "up_recall": float(recall), "up_f1": float(f1),
        "actual_up_rate": float(true_up.mean()), "predicted_up_rate": float(predicted_up.mean()),
        "flat_count": int((y == 0).sum()),
        "confusion_matrix": confusion_matrix(true_up, predicted_up, labels=[False, True]).tolist(),
    }


def comparison_table(metrics):
    """Build a compact comparison / 汇总为便于比较的表格。"""
    columns = ["n", "mae", "rmse", "r2", "r2_vs_zero", "direction_accuracy", "up_f1"]
    return pd.DataFrame(metrics).T[columns].rename_axis("model")


def plot_comparison(metrics, split="Validation"):
    """Return a figure for both CLI and notebook / 命令行和 Notebook 共用图表。"""
    import matplotlib.pyplot as plt
    table = comparison_table(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    names = [n.replace("_", "\n") for n in table.index]
    axes[0].bar(names, table.rmse.astype(float) * 100)
    axes[0].set_ylabel("Daily return RMSE (percentage points)")
    axes[0].set_title(f"{split} error (lower is better)")
    axes[1].bar(names, table.direction_accuracy.astype(float) * 100)
    axes[1].set_ylabel("Direction accuracy (%)")
    axes[1].set_title(f"{split} direction (threshold = 0)")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    return fig
