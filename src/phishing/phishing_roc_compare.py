import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "..", "data", "Training Dataset.arff")

# Load ARFF
data, meta = arff.loadarff(data_path)
df = pd.DataFrame(data)

# Decode byte columns if present
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

# Convert everything to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Rename label column
if "Result" in df.columns:
    df["label"] = df["Result"].astype(int)
    df = df.drop(columns=["Result"])
else:
    df.rename(columns={df.columns[-1]: "label"}, inplace=True)

# Convert -1 to 1 (malicious), 1 to 0 (benign)
df["label"] = df["label"].apply(lambda x: 1 if x == -1 else 0)

X = df.drop("label", axis=1)
y = df["label"]

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaled version for SVM, LR, KNN, MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "SVM": (
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        X_train_scaled,
        X_test_scaled
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        X_train,
        X_test
    ),
    "Logistic Regression": (
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        X_train_scaled,
        X_test_scaled
    ),
    "KNN": (
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ),
        X_train_scaled,
        X_test_scaled
    ),
    "MLP": (
        MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=300,
            early_stopping=True,
            random_state=42
        ),
        X_train_scaled,
        X_test_scaled
    ),
}

plt.figure(figsize=(8, 6))

for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    y_probs = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Phishing Detection ROC Curve Comparison")
plt.legend()
plt.grid(True)

output_path = os.path.join(BASE_DIR, "..", "..", "figures", "Phishing", "phishing_roc_comparison.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"ROC comparison chart saved to: {output_path}")