import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# ------------------------------------------------------------------
# KEEP YOUR EXISTING DATA LOADING BLOCK HERE
# Make sure that by the time you reach below, you already have:
# X = dataframe of predictor columns
# y = target label column
# ------------------------------------------------------------------

# Example only:
# df = pd.read_csv("your_ethereum_dataset.csv")
# X = df.drop("FLAG", axis=1)
# y = df["FLAG"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "..", "..", "data", "transaction_dataset.csv")

df = pd.read_csv(data_path)

# Drop non-numeric columns
df = df.drop(columns=[
    "Unnamed: 0",
    "Index",
    "Address",
    " ERC20 most sent token type",
    " ERC20_most_rec_token_type"
])

X = df.drop("FLAG", axis=1)
y = df["FLAG"]

X = X.fillna(0)

# Split first to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature selection on training data only
selector = SelectKBest(score_func=mutual_info_classif, k=45)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_feature_names = X.columns[selector.get_support()]
print("\nEthereum Fraud Detection - Random Forest + Feature Selection")
print(f"\nNumber of selected features: {len(selected_feature_names)}")
print("Selected features:")
print(selected_feature_names.tolist())

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_selected, y_train)

y_pred = model.predict(X_test_selected)
y_prob = model.predict_proba(X_test_selected)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n")
print(cm)

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc_auc)

# Save confusion matrix image
os.makedirs(os.path.join(BASE_DIR, "..", "..", "..", "figures", "Ethereum"), exist_ok=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Ethereum Fraud Detection - Random Forest + Feature Selection")
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "..", "..", "..", "figures", "Ethereum", "ethereum_rf_fs_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()