
import os
import argparse
import warnings
import numpy as np
import pandas as pd

from scipy.io import arff

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adversarial_utils import generate_adversarial_examples


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FIG_SUBDIR = {
    "ethereum": "Ethereum",
    "phishing": "Phishing",
    "malware": "Malware",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20

ATTACK_METHOD = "transfer"

# Dataset-specific attack settings
ATTACK_CONFIG = {
    "ethereum": {"eps": 0.05, "max_iter": 30, "aug_fraction": 0.50},
    "phishing": {"eps": 0.10, "max_iter": 30, "aug_fraction": 0.50},
    "malware": {"eps": 0.05, "max_iter": 30, "aug_fraction": 0.30},
}


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
def load_ethereum():
    data_path = os.path.join(BASE_DIR, "..", "data", "transaction_dataset.csv")
    df = pd.read_csv(data_path)

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

    X = df.drop("FLAG", axis=1).fillna(0).values.astype(np.float32)
    y = df["FLAG"].astype(int).values

    return X, y


def load_phishing():
    data_path = os.path.join(BASE_DIR, "..", "data", "Training Dataset.arff")

    data, _ = arff.loadarff(data_path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    df = df.apply(pd.to_numeric, errors="coerce")

    if "Result" in df.columns:
        df["label"] = df["Result"].astype(int)
        df = df.drop(columns=["Result"])
    else:
        df.rename(columns={df.columns[-1]: "label"}, inplace=True)

    # UCI phishing: -1 = phishing, 1 = benign
    df["label"] = df["label"].apply(lambda x: 1 if x == -1 else 0)

    X = df.drop("label", axis=1).fillna(0).values.astype(np.float32)
    y = df["label"].values.astype(int)

    return X, y


def load_malware():
    data_path = os.path.join(
        BASE_DIR,
        "..",
        "data",
        "ember",
        "train_ember_2018_v2_features.parquet",
    )

    print("Loading EMBER subset ...")
    df = pd.read_parquet(data_path).iloc[:100000]
    df = df.sample(n=10000, random_state=RANDOM_STATE)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Remove unknown class
    mask = y != -1
    X = X[mask]
    y = y[mask]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X.values.astype(np.float32), y.values.astype(int)


def load_dataset(dataset_name):
    if dataset_name == "ethereum":
        return load_ethereum(), "Ethereum fraud"
    if dataset_name == "phishing":
        return load_phishing(), "Phishing"
    if dataset_name == "malware":
        return load_malware(), "Malware"

    raise ValueError("Dataset must be one of: ethereum, phishing, malware")


# ---------------------------------------------------------------------------
# MODEL BUILDERS
# ---------------------------------------------------------------------------
def get_models(dataset_name):
    """
    Returns the five classifiers.

    For malware, SVM uses SGDClassifier with hinge loss because EMBER is high-dimensional
    and full SVC can be too slow.
    """

    if dataset_name == "malware":
        svm_model = SGDClassifier(
            loss="hinge",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        svm_model = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    models = {
        "SVM": svm_model,
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="constant",
            max_iter=300,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    return models


# ---------------------------------------------------------------------------
# MAJORITY VOTING CLASSIFIER
# ---------------------------------------------------------------------------
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple hard majority voting classifier.

    For ROC curves, predict_proba returns the proportion of models voting for class 1.
    This makes it usable for ROC-AUC even when one model only has decision_function.
    """

    def __init__(self, estimators):
        self.estimators = estimators
        self.fitted_estimators_ = None

    def fit(self, X, y):
        self.fitted_estimators_ = []

        for name, model in self.estimators:
            m = clone(model)
            m.fit(X, y)
            self.fitted_estimators_.append((name, m))

        return self

    def predict(self, X):
        votes = []

        for _, model in self.fitted_estimators_:
            votes.append(model.predict(X))

        votes = np.vstack(votes)
        positive_votes = votes.mean(axis=0)

        return (positive_votes >= 0.5).astype(int)

    def predict_proba(self, X):
        votes = []

        for _, model in self.fitted_estimators_:
            votes.append(model.predict(X))

        votes = np.vstack(votes)
        positive_vote_fraction = votes.mean(axis=0)

        return np.vstack(
            [1.0 - positive_vote_fraction, positive_vote_fraction]
        ).T


# ---------------------------------------------------------------------------
# SCORING HELPERS
# ---------------------------------------------------------------------------
def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict(X).astype(float)


def add_roc_curve(ax, y_true, y_score, label):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=1.7, label=f"{label} (AUC={auc:.3f})")
    except ValueError:
        print(f"[warning] Could not compute ROC for {label}")


# ---------------------------------------------------------------------------
# TRAIN / ATTACK / HARDEN PIPELINE
# ---------------------------------------------------------------------------
def evaluate_model_four_conditions(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    eps,
    max_iter,
    aug_fraction,
):
    print("\n" + "-" * 80)
    print(f"Training and evaluating: {model_name}")
    print("-" * 80)

    baseline = clone(model)
    baseline.fit(X_train, y_train)

    baseline_clean_scores = get_scores(baseline, X_test)

    print(f"[{model_name}] generating baseline attack test set ...")
    X_test_adv_baseline = generate_adversarial_examples(
        baseline,
        X_test,
        y_test,
        method=ATTACK_METHOD,
        eps=eps,
        max_iter=max_iter,
    )

    baseline_attack_scores = get_scores(baseline, X_test_adv_baseline)

    rng = np.random.default_rng(RANDOM_STATE)
    n_aug = int(aug_fraction * X_train.shape[0])
    idx = rng.choice(X_train.shape[0], size=n_aug, replace=False)

    print(f"[{model_name}] generating adversarial training samples ...")
    X_train_adv = generate_adversarial_examples(
        baseline,
        X_train[idx],
        y_train[idx],
        method=ATTACK_METHOD,
        eps=eps,
        max_iter=max_iter,
    )

    X_aug = np.vstack([X_train, X_train_adv])
    y_aug = np.concatenate([y_train, y_train[idx]])

    hardened = clone(model)
    hardened.fit(X_aug, y_aug)

    hardened_clean_scores = get_scores(hardened, X_test)

    print(f"[{model_name}] generating hardened attack test set ...")
    X_test_adv_hardened = generate_adversarial_examples(
        hardened,
        X_test,
        y_test,
        method=ATTACK_METHOD,
        eps=eps,
        max_iter=max_iter,
    )

    hardened_attack_scores = get_scores(hardened, X_test_adv_hardened)

    return {
        "Baseline clean": baseline_clean_scores,
        "Baseline under attack": baseline_attack_scores,
        "Hardened clean": hardened_clean_scores,
        "Hardened under attack": hardened_attack_scores,
    }


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_all_roc(results, y_test, title, out_path):
    conditions = [
        "Baseline clean",
        "Baseline under attack",
        "Hardened clean",
        "Hardened under attack",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    for ax, condition in zip(axes, conditions):
        for model_name, model_results in results.items():
            y_score = model_results[condition]
            add_roc_curve(ax, y_test, y_score, model_name)

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(condition)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ethereum", "phishing", "malware"],
        help="Dataset to run: ethereum, phishing, or malware",
    )

    args = parser.parse_args()
    dataset_name = args.dataset

    FIG_DIR = os.path.join(BASE_DIR, "..", "figures", DATASET_FIG_SUBDIR[dataset_name])
    os.makedirs(FIG_DIR, exist_ok=True)

    (X, y), pretty_dataset_name = load_dataset(dataset_name)

    print(f"\nDataset: {pretty_dataset_name}")
    print(f"X shape: {X.shape}")
    print(f"Positive class count: {int(y.sum())}/{len(y)}")

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

    cfg = ATTACK_CONFIG[dataset_name]
    eps = cfg["eps"]
    max_iter = cfg["max_iter"]
    aug_fraction = cfg["aug_fraction"]

    models = get_models(dataset_name)

    # -----------------------------------------------------------------------
    # 1. Run without Majority Voting
    # -----------------------------------------------------------------------
    results_no_mv = {}

    for model_name, model in models.items():
        results_no_mv[model_name] = evaluate_model_four_conditions(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            eps,
            max_iter,
            aug_fraction,
        )

    out_no_mv = os.path.join(
        FIG_DIR,
        f"roc_{dataset_name}_all_classifiers_no_majority_voting.png",
    )

    plot_all_roc(
        results_no_mv,
        y_test,
        f"ROC comparison on {pretty_dataset_name} without Majority Voting",
        out_no_mv,
    )

    # -----------------------------------------------------------------------
    # 2. Run with Majority Voting
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Adding Majority Voting classifier")
    print("=" * 80)

    majority_model = MajorityVoteClassifier(
        estimators=[(name, model) for name, model in models.items()]
    )

    results_with_mv = dict(results_no_mv)

    results_with_mv["Majority Voting"] = evaluate_model_four_conditions(
        "Majority Voting",
        majority_model,
        X_train,
        X_test,
        y_train,
        y_test,
        eps,
        max_iter,
        aug_fraction,
    )

    out_with_mv = os.path.join(
        FIG_DIR,
        f"roc_{dataset_name}_all_classifiers_with_majority_voting.png",
    )

    plot_all_roc(
        results_with_mv,
        y_test,
        f"ROC comparison on {pretty_dataset_name} with Majority Voting",
        out_with_mv,
    )

    print("\nFinished.")
    print(f"Saved without Majority Voting: {out_no_mv}")
    print(f"Saved with Majority Voting:    {out_with_mv}")


if __name__ == "__main__":
    main()