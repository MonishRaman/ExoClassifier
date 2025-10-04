import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Drop rows where koi_disposition is missing
data = data.dropna(subset=["koi_disposition"])

# Separate features and target
X = data.drop(columns=["koi_disposition"])
y = data["koi_disposition"]

# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Remove columns that are entirely NaN
X = X.dropna(axis=1, how='all')

# Replace inf/-inf with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill all remaining NaNs with column medians
for col in X.columns:
    X[col].fillna(X[col].median(), inplace=True)

# Drop any rows that still have NaN (as last resort)
X = X.dropna()

# Sanity check
print("✅ Remaining NaN count:", X.isna().sum().sum())

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Match y to X (drop labels for dropped rows)
y = y[:len(X)]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

import joblib

# Save trained model
joblib.dump(model, "exoplanet_model.pkl")
print("✅ Model saved as exoplanet_model.pkl")

# import joblib
joblib.dump(scaler, "scaler.pkl")
