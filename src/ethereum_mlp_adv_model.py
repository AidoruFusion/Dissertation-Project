"""
ethereum_mlp_adv_model.py
-------------------------
Adversarial training of the lightweight MLP on the Ethereum fraud
detection dataset, using PGD per Madry et al. (2018).
"""

import os
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adversarial_utils import adversarial_training, summarise


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "..", "data", "transaction_dataset.csv")
TARGET_COL   = "FLAG"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

ATTACK_METHOD = "pgd"
EPS           = 0.10
MAX_ITER      = 40
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


def load_ethereum():
    """Matches the loading pipeline in ethereum_rf_model.py exactly."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[
        "Unnamed: 0",
        "Index",
        "Address",
        " ERC20 most sent token type",
        " ERC20_most_rec_token_type",
    ], errors="ignore")
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].astype(int).values
    X = X.fillna(0).values.astype(np.float32)
    return X, y

def main():
    print("Loading Ethereum fraud dataset ...")
    X, y = load_ethereum()
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

    hardened, results = adversarial_training(
        baseline_model=baseline,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        attack_method=ATTACK_METHOD,
        eps=EPS,
        max_iter=MAX_ITER,
        aug_fraction=AUG_FRACTION,
        random_state=RANDOM_STATE,
    )

    out_csv = f"results_ethereum_mlp_adv_{ATTACK_METHOD}.csv"
    results.to_csv(out_csv, index=False)
    summarise(results, "Ethereum fraud", "MLP (hardened)")
    print(f"\nResults written to {out_csv}")


if __name__ == "__main__":
    main()
