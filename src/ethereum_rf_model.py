import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "transaction_dataset.csv")

# Load dataset
df = pd.read_csv(data_path)

# Drop identifier / non-numeric columns
df = df.drop(columns=[
    "Unnamed: 0",
    "Index",
    "Address",
    " ERC20 most sent token type",
    " ERC20_most_rec_token_type"
], errors="ignore")

# Features and target
X = df.drop("FLAG", axis=1)
y = df["FLAG"]

# Fill missing values
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nEthereum Fraud Detection - Random Forest\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_probs))