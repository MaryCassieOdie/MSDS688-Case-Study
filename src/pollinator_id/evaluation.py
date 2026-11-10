"""Metrics and plots for comparing backbones."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, top_k_accuracy_score)

from . import config
from .iucn import THREATENED


def summarize(name: str, y_true, probs, params: int, train_seconds: float, ms_per_image: float,
              class_names=config.CLASS_NAMES, iucn_codes: dict | None = None) -> dict:
    y_pred = probs.argmax(1)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "top3_accuracy": top_k_accuracy_score(y_true, probs, k=3, labels=range(len(class_names))),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "params_millions": params / 1e6,
        "train_minutes": train_seconds / 60,
        "ms_per_image": ms_per_image,
    }
    if iucn_codes:
        # Recall on threatened species is what matters most for conservation: a
        # missed rusty-patched bumble bee is costlier than a missed honey bee.
        idx = [i for i, n in enumerate(class_names) if iucn_codes.get(n) in THREATENED]
        mask = np.isin(y_true, idx)
        row["threatened_recall"] = float((y_pred[mask] == y_true[mask]).mean()) if mask.any() else np.nan
        # Threatened-vs-not as a binary decision (right status even if wrong species).
        pred_thr, true_thr = np.isin(y_pred, idx), mask
        row["threatened_flag_recall"] = float(pred_thr[true_thr].mean()) if true_thr.any() else np.nan
        row["threatened_flag_precision"] = float(true_thr[pred_thr].mean()) if pred_thr.any() else np.nan
    return row


def per_class_report(y_true, probs, class_names=config.CLASS_NAMES) -> pd.DataFrame:
    rep = classification_report(y_true, probs.argmax(1), labels=range(len(class_names)),
                                target_names=class_names, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T


def plot_confusion(y_true, probs, title: str, class_names=config.CLASS_NAMES, ax=None):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, probs.argmax(1), labels=range(len(class_names)), normalize="true")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    short = [f"{n.split()[0][0]}. {n.split()[1]}" for n in class_names]
    ax.set_xticks(range(len(short)), short, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(short)), short, fontsize=8)
    for i in range(len(short)):
        for j in range(len(short)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_histories(histories: dict[str, dict], metric: str = "accuracy"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for name, h in histories.items():
        line, = axes[0].plot(h[metric], label=f"{name} train")
        axes[0].plot(h[f"val_{metric}"], "--", color=line.get_color(), label=f"{name} val")
        axes[1].plot(h["val_loss"], color=line.get_color(), label=name)
        axes[0].axvline(h["fine_tune_start_epoch"] - 0.5, color=line.get_color(), alpha=0.25)
    axes[0].set_title(f"{metric} (vertical line = fine-tuning starts)")
    axes[1].set_title("validation loss")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig
