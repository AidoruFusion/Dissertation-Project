"""
adversarial_plots.py
--------------------
Confusion-matrix and ROC-curve figure helpers, styled to match the
supervisor-preferred format from Yeboah-Ofori et al. (2022).

Each confusion matrix figure shows:
    - True/Predicted axes
    - Cell counts annotated inside
    - Classification report (precision, recall, f1-score, support per class,
      plus accuracy / macro avg / weighted avg)
    - ROC AUC line at the bottom

Output naming:
    cm_{dataset}_{model}_{condition}.png   (confusion + report)
    roc_{dataset}_{model}.png              (multi-condition ROC overlay)
"""

from __future__ import annotations
import os
from typing import Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Confusion matrix + classification report (supervisor style)
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: str,
    class_names=("Non-fraud", "Fraud"),
    roc_auc_value: float | None = None,
    model_label: str | None = None,
) -> None:
    """
    Save a confusion matrix WITH classification report below.

    Parameters
    ----------
    y_true       : ground-truth labels.
    y_pred       : predicted labels.
    title        : figure title.
    out_path     : path to save PNG.
    class_names  : 2-tuple of (negative_label, positive_label).
    roc_auc_value: optional ROC AUC value to print at the bottom. If None,
                   the AUC line is suppressed (e.g. when y_pred is hard
                   labels rather than scores).
    model_label  : optional name shown next to the AUC line, e.g.
                   "Random Forest". Defaults to title if not supplied.
    """
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(
        y_true, y_pred,
        target_names=list(class_names),
        digits=2, zero_division=0,
    )

    # Figure layout: heatmap on top, report+AUC text below.
    fig = plt.figure(figsize=(6.4, 7.6))
    gs = gridspec.GridSpec(
        nrows=2, ncols=1,
        height_ratios=[3.4, 2.6],
        hspace=0.35,
    )

    # --- top: heatmap ---
    ax_cm = fig.add_subplot(gs[0])
    im = ax_cm.imshow(cm, cmap="rocket_r" if "rocket_r" in plt.colormaps() else "Blues")
    ax_cm.set_title(title, fontsize=11, pad=12)

    # X-axis labels on top, like the paper.
    ax_cm.xaxis.set_label_position("top")
    ax_cm.xaxis.tick_top()
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["0", "1"])
    ax_cm.set_yticklabels(["0", "1"])
    ax_cm.set_xlabel("Predicted", fontsize=10, labelpad=8)
    ax_cm.set_ylabel("True", fontsize=10)

    # Annotate cell counts.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # --- bottom: classification report and AUC line ---
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")

    text_block = report_text
    if roc_auc_value is not None:
        suffix = f"ROC AUC: {roc_auc_value:.2f}"
        if model_label:
            suffix += f"   [{model_label}]"
        text_block = text_block.rstrip() + "\n\n" + suffix

    ax_txt.text(
        0.0, 1.0, text_block,
        family="monospace", fontsize=10,
        va="top", ha="left",
        transform=ax_txt.transAxes,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ---------------------------------------------------------------------------
# Multi-condition ROC overlay (unchanged from previous version)
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
            continue

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
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


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute ROC AUC if possible; return None if degenerate (e.g. one class)."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None
