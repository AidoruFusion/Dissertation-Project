

from __future__ import annotations

import warnings
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.linear_model import LogisticRegression

# ART wrappers and attacks. Install with:
#   pip install adversarial-robustness-toolbox
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    HopSkipJump,
    ZooAttack,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Adversarial example generation
# ---------------------------------------------------------------------------

def wrap_classifier(model: BaseEstimator, n_features: int) -> SklearnClassifier:
    """Wrap a fitted sklearn classifier so that ART can attack it."""
    return SklearnClassifier(model=model, clip_values=None)


def generate_adversarial_examples(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "pgd",
    eps: float = 0.1,
    eps_step: float = 0.02,
    max_iter: int = 40,
    surrogate: Optional[BaseEstimator] = None,
) -> np.ndarray:
    """
    Generate adversarial examples.

    Parameters
    ----------
    model     : the fitted target classifier (e.g. RF, MLP).
    X, y      : samples and true labels to be perturbed.
    method    : one of {'fgsm','pgd','hopskipjump','zoo','transfer'}.
                - fgsm / pgd: white-box gradient attacks, suitable for MLP.
                - hopskipjump / zoo: black-box query attacks, suitable for RF.
                - transfer: craft on a logistic-regression surrogate and
                  transfer to the target; works for any classifier.
    eps       : L-inf budget (features are assumed standardised, so ~0.05-0.2
                is a reasonable range).
    eps_step  : step size per PGD iteration.
    max_iter  : iteration budget for iterative attacks.
    surrogate : optional pre-fitted surrogate for 'transfer' method.

    Returns
    -------
    X_adv : np.ndarray of the same shape as X.
    """
    method = method.lower()
    n_features = X.shape[1]

    if method == "fgsm":
        clf = wrap_classifier(model, n_features)
        attack = FastGradientMethod(estimator=clf, eps=eps, norm=np.inf)
        return attack.generate(x=X.astype(np.float32))

    if method == "pgd":
        clf = wrap_classifier(model, n_features)
        attack = ProjectedGradientDescent(
            estimator=clf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            norm=np.inf,
            num_random_init=1,
        )
        return attack.generate(x=X.astype(np.float32))

    if method == "hopskipjump":
        clf = wrap_classifier(model, n_features)
        attack = HopSkipJump(
            classifier=clf,
            targeted=False,
            norm=np.inf,
            max_iter=max_iter,
            max_eval=1000,
            init_eval=100,
            init_size=100,
        )
        return attack.generate(x=X.astype(np.float32))

    if method == "zoo":
        clf = wrap_classifier(model, n_features)
        attack = ZooAttack(
            classifier=clf,
            confidence=0.0,
            targeted=False,
            learning_rate=1e-2,
            max_iter=max_iter,
            binary_search_steps=10,
            initial_const=1e-3,
            nb_parallel=5,
        )
        return attack.generate(x=X.astype(np.float32))

    if method == "transfer":
        if surrogate is None:
            # Train a quick LR surrogate on the same samples.
            surrogate = LogisticRegression(max_iter=1000, n_jobs=-1)
            surrogate.fit(X, y)
        sur_clf = wrap_classifier(surrogate, n_features)
        attack = ProjectedGradientDescent(
            estimator=sur_clf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            norm=np.inf,
            num_random_init=1,
        )
        return attack.generate(x=X.astype(np.float32))

    raise ValueError(f"Unknown attack method: {method!r}")


# ---------------------------------------------------------------------------
# 2. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    label: str = "",
    pos_label: int = 1,
) -> Dict[str, float]:
    """Return the standard metric dict used throughout the dissertation."""
    y_pred = model.predict(X)
    # Probabilistic score for ROC-AUC, if available.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    else:
        y_score = y_pred
    try:
        roc = roc_auc_score(y, y_score)
    except ValueError:
        roc = float("nan")

    metrics = {
        "label":     label,
        "accuracy":  accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label=pos_label, zero_division=0),
        "recall":    recall_score(y, y_pred, pos_label=pos_label, zero_division=0),
        "f1":        f1_score(y, y_pred, pos_label=pos_label, zero_division=0),
        "roc_auc":   roc,
    }

    print(f"\n--- {label} ---")
    for k, v in metrics.items():
        if k != "label":
            print(f"  {k:<10s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  Confusion matrix:")
    print(confusion_matrix(y, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# 3. Core adversarial training loop
# ---------------------------------------------------------------------------

def adversarial_training(
    baseline_model: BaseEstimator,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    attack_method: str = "pgd",
    eps: float = 0.1,
    max_iter: int = 20,
    aug_fraction: float = 0.5,
    random_state: int = 42,
) -> Tuple[BaseEstimator, pd.DataFrame]:
    """
    Run a full baseline-vs-hardened experiment and return the hardened
    model plus a tidy results DataFrame.

    Procedure:
      1. Evaluate the baseline on the clean test set.
      2. Craft adversarial examples against the baseline on (a subset of)
         the training set and (all of) the test set.
      3. Evaluate the baseline on the baseline-crafted adversarial test set.
      4. Retrain a fresh copy of the classifier on [X_train; adversarial X_train].
      5. Re-craft adversarial examples against the *hardened* model on the
         test set (this avoids the evaluation pitfall flagged by Carlini
         et al., 2019).
      6. Evaluate the hardened model on clean test + hardened-crafted adv test.
    """
    from sklearn.base import clone

    rng = np.random.default_rng(random_state)

    # --- 1. Baseline on clean ---
    m_clean_baseline = evaluate(baseline_model, X_test, y_test,
                                "Baseline (clean)")

    # --- 2. Craft adversarial training pool and adversarial test set ---
    # Sample a fraction of the training set to attack (keeps runtime sane
    # on large datasets like EMBER).
    n_train = X_train.shape[0]
    n_aug = int(aug_fraction * n_train)
    idx = rng.choice(n_train, size=n_aug, replace=False)
    print(f"\n[adv] crafting {n_aug} adversarial training samples with "
          f"{attack_method}, eps={eps} ...")
    X_train_adv = generate_adversarial_examples(
        baseline_model,
        X_train[idx], y_train[idx],
        method=attack_method, eps=eps, max_iter=max_iter,
    )
    print(f"[adv] crafting adversarial test set against baseline ...")
    X_test_adv_baseline = generate_adversarial_examples(
        baseline_model,
        X_test, y_test,
        method=attack_method, eps=eps, max_iter=max_iter,
    )

    # --- 3. Baseline under attack ---
    m_attack_baseline = evaluate(baseline_model, X_test_adv_baseline, y_test,
                                 "Baseline under attack")

    # --- 4. Retrain on augmented data ---
    hardened = clone(baseline_model)
    # Some sklearn models need a fixed random_state; re-apply if present.
    if hasattr(hardened, "random_state"):
        hardened.random_state = random_state

    X_aug = np.vstack([X_train, X_train_adv])
    y_aug = np.concatenate([y_train, y_train[idx]])
    print(f"\n[adv] retraining hardened model on "
          f"{X_aug.shape[0]} samples ({n_aug} adversarial) ...")
    hardened.fit(X_aug, y_aug)

    # --- 5. Re-craft adversarial test against the hardened model ---
    print(f"[adv] crafting adversarial test set against hardened model ...")
    X_test_adv_hardened = generate_adversarial_examples(
        hardened,
        X_test, y_test,
        method=attack_method, eps=eps, max_iter=max_iter,
    )

    # --- 6. Hardened evaluation ---
    m_clean_hardened  = evaluate(hardened, X_test, y_test,
                                 "Hardened (clean)")
    m_attack_hardened = evaluate(hardened, X_test_adv_hardened, y_test,
                                 "Hardened under attack")

    results = pd.DataFrame([
        m_clean_baseline, m_attack_baseline,
        m_clean_hardened, m_attack_hardened,
    ])
    return hardened, results


# ---------------------------------------------------------------------------
# 4. Pretty-print summary for copy-pasting into the dissertation
# ---------------------------------------------------------------------------

def summarise(results: pd.DataFrame, dataset_name: str, model_name: str) -> None:
    print(f"\n================ SUMMARY: {model_name} on {dataset_name} ================")
    cols = ["label", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print(results[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.3f}"))
    print("=" * (40 + len(model_name) + len(dataset_name)))
