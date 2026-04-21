import pandas as pd
import os
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "Training Dataset.arff")

# Load ARFF
data, meta = arff.loadarff(data_path)
df = pd.DataFrame(data)

# Decode byte columns if present
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Drop missing rows
df = df.dropna()

# Split features and label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert labels
y = y.replace(-1, 0)
y = y.replace(1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
k = min(20, X_train.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]

# Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_selected, y_train)

# Predictions
y_pred = model.predict(X_test_selected)
y_prob = model.predict_proba(X_test_selected)[:, 1]

# Results
print("Phishing Detection - KNN + Feature Selection\n")
print(f"Number of selected features: {k}")
print("Selected features:")
print(list(selected_features))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))