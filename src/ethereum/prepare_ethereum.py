import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "..", "data", "transaction_dataset.csv")

df = pd.read_csv(data_path)

# Drop non-numeric / identifier columns
df = df.drop(columns=[
    "Unnamed: 0",
    "Index",
    "Address",
    " ERC20 most sent token type",
    " ERC20_most_rec_token_type"
])

X = df.drop("FLAG", axis=1)
y = df["FLAG"]

# Replace NaNs with 0 (absence-of-behaviour assumption)
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_probs))