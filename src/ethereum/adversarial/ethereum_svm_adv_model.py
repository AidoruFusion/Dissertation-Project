

import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adversarial_utils import evaluate, generate_adversarial_examples, summarise
from adversarial_plots import (
    save_confusion_matrix,
    save_roc_comparison,
    get_scores,
    safe_auc,
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "transaction_dataset.csv")
FIG_DIR = os.path.join(BASE_DIR, "..", "..", "..", "figures", "Ethereum")

TARGET_COL = "FLAG"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Use transfer attack for SVM so it stays consistent with your RF setup.
ATTACK_METHOD = "transfer"
EPS = 0.05
MAX_ITER = 30
AUG_FRACTION = 0.50

SVM_PARAMS = dict(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

DATASET_NAME = "ethereum"
MODEL_NAME = "svm"
PRETTY_DS = "Ethereum fraud"
PRETTY_MDL = "SVM"
CLASS_NAMES = ("Non-fraud", "Fraud")


def load_ethereum():
    df = pd.read_csv(DATA_PATH)

    df = df.drop(
        columns=[
            "Unnamed: 0",
            "Index",
            "Address",
            " ERC20 most sent token type",
            " ERC20_most_rec_token_type",
        ],
        errors="ignore",
    )

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].astype(int).values

    X = X.fillna(0).values.astype(np.float32)

    return X, y


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading Ethereum fraud dataset ...")
    X, y = load_ethereum()
    print(f"Shape: X={X.shape}, positives={int(y.sum())}/{len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # -----------------------------------------------------------------------
    # 1. Train baseline SVM
    # -----------------------------------------------------------------------
    print("\nTraining baseline SVM ...")
    baseline = SVC(**SVM_PARAMS)
    baseline.fit(X_train, y_train)

    m_clean_baseline = evaluate(
        baseline,
        X_test,
        y_test,
        "Baseline (clean)",
    )

    # -----------------------------------------------------------------------
    # 2. Generate adversarial training samples and baseline attack test set
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    n_aug = int(AUG_FRACTION * X_train.shape[0])
    idx = rng.choice(X_train.shape[0], size=n_aug, replace=False)

    print(f"\n[adv] crafting {n_aug} adversarial training samples ...")
    X_train_adv = generate_adversarial_examples(
        baseline,
        X_train[idx],
        y_train[idx],
        method=ATTACK_METHOD,
        eps=EPS,
        max_iter=MAX_ITER,
    )

    print("[adv] crafting adversarial test set against baseline ...")
    X_test_adv_baseline = generate_adversarial_examples(
        baseline,
        X_test,
        y_test,
        method=ATTACK_METHOD,
        eps=EPS,
        max_iter=MAX_ITER,
    )

    m_attack_baseline = evaluate(
        baseline,
        X_test_adv_baseline,
        y_test,
        "Baseline under attack",
    )

    # -----------------------------------------------------------------------
    # 3. Retrain hardened SVM
    # -----------------------------------------------------------------------
    hardened = clone(baseline)

    X_aug = np.vstack([X_train, X_train_adv])
    y_aug = np.concatenate([y_train, y_train[idx]])

    print(f"\n[adv] retraining hardened model on {X_aug.shape[0]} samples ...")
    hardened.fit(X_aug, y_aug)

    # -----------------------------------------------------------------------
    # 4. Re-attack hardened model
    # -----------------------------------------------------------------------
    print("[adv] crafting adversarial test set against hardened model ...")
    X_test_adv_hardened = generate_adversarial_examples(
        hardened,
        X_test,
        y_test,
        method=ATTACK_METHOD,
        eps=EPS,
        max_iter=MAX_ITER,
    )

    m_clean_hardened = evaluate(
        hardened,
        X_test,
        y_test,
        "Hardened (clean)",
    )

    m_attack_hardened = evaluate(
        hardened,
        X_test_adv_hardened,
        y_test,
        "Hardened under attack",
    )

    # -----------------------------------------------------------------------
    # 5. Save results CSV
    # -----------------------------------------------------------------------
    results = pd.DataFrame(
        [
            m_clean_baseline,
            m_attack_baseline,
            m_clean_hardened,
            m_attack_hardened,
        ]
    )

    _results_dir = os.path.join(BASE_DIR, "..", "..", "..", "results", DATASET_NAME)
    os.makedirs(_results_dir, exist_ok=True)
    out_csv = os.path.join(_results_dir, f"results_{DATASET_NAME}_{MODEL_NAME}_adv_{ATTACK_METHOD}.csv")
    results.to_csv(out_csv, index=False)

    summarise(results, PRETTY_DS, PRETTY_MDL + " (hardened)")
    print(f"\nResults written to {out_csv}")

    # -----------------------------------------------------------------------
    # 6. Save four confusion matrices
    # -----------------------------------------------------------------------
    print("\nGenerating confusion matrices ...")

    save_confusion_matrix(
        y_test,
        baseline.predict(X_test),
        f"{PRETTY_MDL}: Baseline clean ({PRETTY_DS})",
        os.path.join(FIG_DIR, f"cm_{DATASET_NAME}_{MODEL_NAME}_baseline_clean.png"),
        class_names=CLASS_NAMES,
        roc_auc_value=safe_auc(y_test, get_scores(baseline, X_test)),
        model_label=PRETTY_MDL,
    )

    save_confusion_matrix(
        y_test,
        baseline.predict(X_test_adv_baseline),
        f"{PRETTY_MDL}: Baseline under attack ({PRETTY_DS})",
        os.path.join(FIG_DIR, f"cm_{DATASET_NAME}_{MODEL_NAME}_baseline_attack.png"),
        class_names=CLASS_NAMES,
        roc_auc_value=safe_auc(y_test, get_scores(baseline, X_test_adv_baseline)),
        model_label=PRETTY_MDL,
    )

    save_confusion_matrix(
        y_test,
        hardened.predict(X_test),
        f"{PRETTY_MDL}: Hardened clean ({PRETTY_DS})",
        os.path.join(FIG_DIR, f"cm_{DATASET_NAME}_{MODEL_NAME}_hardened_clean.png"),
        class_names=CLASS_NAMES,
        roc_auc_value=safe_auc(y_test, get_scores(hardened, X_test)),
        model_label=PRETTY_MDL,
    )

    save_confusion_matrix(
        y_test,
        hardened.predict(X_test_adv_hardened),
        f"{PRETTY_MDL}: Hardened under attack ({PRETTY_DS})",
        os.path.join(FIG_DIR, f"cm_{DATASET_NAME}_{MODEL_NAME}_hardened_attack.png"),
        class_names=CLASS_NAMES,
        roc_auc_value=safe_auc(y_test, get_scores(hardened, X_test_adv_hardened)),
        model_label=PRETTY_MDL,
    )

    # -----------------------------------------------------------------------
    # 7. Save ROC comparison
    # -----------------------------------------------------------------------
    save_roc_comparison(
        runs={
            "Baseline clean": (y_test, get_scores(baseline, X_test)),
            "Baseline under attack": (
                y_test,
                get_scores(baseline, X_test_adv_baseline),
            ),
            "Hardened clean": (y_test, get_scores(hardened, X_test)),
            "Hardened under attack": (
                y_test,
                get_scores(hardened, X_test_adv_hardened),
            ),
        },
        title=f"ROC: {PRETTY_MDL} on {PRETTY_DS}",
        out_path=os.path.join(FIG_DIR, f"roc_{DATASET_NAME}_{MODEL_NAME}.png"),
    )

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()