import pandas as pd
import os
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "Training Dataset.arff")

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

# MLP needs scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="constant",
    max_iter=300,
    early_stopping=True,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("\nPhishing Detection - MLP\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_probs))