

import os
import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adversarial_utils import evaluate, generate_adversarial_examples, summarise
from adversarial_plots import save_confusion_matrix, save_roc_comparison, get_scores


# ---------------------------------------------------------------------------
# CONFIG -- aligned with phishing_mlp_model.py baseline
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "..", "data", "Training Dataset.arff")
FIG_DIR      = os.path.join(BASE_DIR, "..", "figures")
RANDOM_STATE = 42
TEST_SIZE    = 0.20

ATTACK_METHOD = "transfer"
EPS           = 0.10
MAX_ITER      = 30
AUG_FRACTION  = 0.50

MLP_PARAMS = dict(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="constant",
    max_iter=300,
    early_stopping=True,
    random_state=RANDOM_STATE,
)

DATASET_NAME = "phishing"
MODEL_NAME   = "mlp"
PRETTY_DS    = "Phishing"
PRETTY_MDL   = "MLP"
CLASS_NAMES  = ("Benign", "Phishing")


def load_phishing():
    """Matches phishing_mlp_model.py loading exactly."""
    data, _ = arff.loadarff(DATA_PATH)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    df = df.apply(pd.to_numeric, errors="coerce")
    if "Result" in df.columns:
        df["label"] = df["Result"].astype(int)
        df = df.drop(columns=["Result"])
    else:
        df.rename(columns={df.columns[-1]: "label"}, inplace=True)
    df["label"] = df["label"].apply(lambda x: 1 if x == -1 else 0)

    X = df.drop("label", axis=1).fillna(0).values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading Phishing dataset ...")
    X, y = load_phishing()
    print(f"Shape: X={X.shape}, positives={int(y.sum())}/{len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    print("\nTraining baseline MLP ...")
    baseline = MLPClassifier(**MLP_PARAMS)
    baseline.fit(X_train, y_train)
    m_clean_baseline = evaluate(baseline, X_test, y_test, "Baseline (clean)")

    rng = np.random.default_rng(RANDOM_STATE)
    n_aug = int(AUG_FRACTION * X_train.shape[0])
    idx = rng.choice(X_train.shape[0], size=n_aug, replace=False)

    print(f"\n[adv] crafting {n_aug} adversarial training samples ...")
    X_train_adv = generate_adversarial_examples(
        baseline, X_train[idx], y_train[idx],
        method=ATTACK_METHOD, eps=EPS, max_iter=MAX_ITER,
    )
    print("[adv] crafting adversarial test set against baseline ...")
    X_test_adv_baseline = generate_adversarial_examples(
        baseline, X_test, y_test,
        method=ATTACK_METHOD, eps=EPS, max_iter=MAX_ITER,
    )
    m_attack_baseline = evaluate(baseline, X_test_adv_baseline, y_test,
                                 "Baseline under attack")

    hardened = clone(baseline)
    if hasattr(hardened, "random_state"):
        hardened.random_state = RANDOM_STATE
    X_aug = np.vstack([X_train, X_train_adv])
    y_aug = np.concatenate([y_train, y_train[idx]])
    print(f"\n[adv] retraining hardened model on {X_aug.shape[0]} samples ...")
    hardened.fit(X_aug, y_aug)

    print("[adv] crafting adversarial test set against hardened model ...")
    X_test_adv_hardened = generate_adversarial_examples(
        hardened, X_test, y_test,
        method=ATTACK_METHOD, eps=EPS, max_iter=MAX_ITER,
    )
    m_clean_hardened  = evaluate(hardened, X_test, y_test, "Hardened (clean)")
    m_attack_hardened = evaluate(hardened, X_test_adv_hardened, y_test,
                                 "Hardened under attack")

    results = pd.DataFrame([
        m_clean_baseline, m_attack_baseline,
        m_clean_hardened, m_attack_hardened,
    ])
    out_csv = f"results_{DATASET_NAME}_{MODEL_NAME}_adv_{ATTACK_METHOD}.csv"
    results.to_csv(out_csv, index=False)
    summarise(results, PRETTY_DS, PRETTY_MDL + " (hardened)")
    print(f"\nResults written to {out_csv}")

    print("\nGenerating figures ...")
    save_confusion_matrix(
        y_test, baseline.predict(X_test_adv_baseline),
        f"{PRETTY_MDL}: Baseline under attack ({PRETTY_DS})",
        os.path.join(FIG_DIR, f"cm_{DATASET_NAME}_{MODEL_NAME}_baseline_attack.png"),
        class_names=CLASS_NAMES,
    )
    save_confusion_matrix(
        y_test, hardened.predict(X_test_adv_hardened),
        f"{PRETTY_MDL}: Hardened under attack ({PRETTY_DS})",
        os.path.join(FIG_DIR, f"cm_{DATASET_NAME}_{MODEL_NAME}_hardened_attack.png"),
        class_names=CLASS_NAMES,
    )
    save_confusion_matrix(
        y_test, baseline.predict(X_test),
        f"{PRETTY_MDL}: Baseline clean ({PRETTY_DS})",
        os.path.join(FIG_DIR, f"cm_{DATASET_NAME}_{MODEL_NAME}_baseline_clean.png"),
        class_names=CLASS_NAMES,
    )
    save_confusion_matrix(
        y_test, hardened.predict(X_test),
        f"{PRETTY_MDL}: Hardened clean ({PRETTY_DS})",
        os.path.join(FIG_DIR, f"cm_{DATASET_NAME}_{MODEL_NAME}_hardened_clean.png"),
        class_names=CLASS_NAMES,
    )

    save_roc_comparison(
        runs={
            "Baseline (clean)":       (y_test, get_scores(baseline, X_test)),
            "Baseline under attack":  (y_test, get_scores(baseline, X_test_adv_baseline)),
            "Hardened (clean)":       (y_test, get_scores(hardened, X_test)),
            "Hardened under attack":  (y_test, get_scores(hardened, X_test_adv_hardened)),
        },
        title=f"ROC: {PRETTY_MDL} on {PRETTY_DS}",
        out_path=os.path.join(FIG_DIR, f"roc_{DATASET_NAME}_{MODEL_NAME}.png"),
    )

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
