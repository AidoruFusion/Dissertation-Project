"""
adversarial_plots.py
--------------------
Saves confusion-matrix and ROC-curve figures for adversarial training
experiments. Produces consistently-named PNG files ready to drop into
the dissertation.

Output naming pattern:
    cm_{dataset}_{model}_{condition}.png      (confusion matrices)
    roc_{dataset}_{model}.png                 (combined ROC overlay)

Usage:
    See the bottom of ethereum_rf_adv_model.py / ethereum_mlp_adv_model.py
    for the few extra lines needed at the end of main() to call this.
"""

from __future__ import annotations
import os
from typing import Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; no GUI required
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: str,
    class_names=("Non-fraud", "Fraud"),
) -> None:
    """Save a 2x2 confusion matrix as a PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    # Annotate each cell with its count.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ---------------------------------------------------------------------------
# Combined ROC overlay
# ---------------------------------------------------------------------------

def save_roc_comparison(
    runs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    out_path: str,
) -> None:
    """
    Plot multiple ROC curves on a single axis.

    runs : dict mapping condition_label -> (y_true, y_score)
    """
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for label, (y_true, y_score) in runs.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", linewidth=1.8)
        except ValueError:
            # Single-class predictions can break roc_curve; skip gracefully.
            continue

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ---------------------------------------------------------------------------
# Helper: pull a probability/score out of any sklearn model
# ---------------------------------------------------------------------------

def get_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)
