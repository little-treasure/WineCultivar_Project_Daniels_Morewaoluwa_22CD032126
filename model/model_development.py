# ================================
# WINE CULTIVAR MODEL BUILDING
# ================================

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load Wine Dataset
# -------------------------------
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# -------------------------------
# Feature Selection (6 only)
# -------------------------------
selected_features = [
    'alcohol',
    'malic_acid',
    'ash',
    'flavanoids',
    'color_intensity',
    'proline'
]

X = X[selected_features]

# -------------------------------
# Check Missing Values
# -------------------------------
print(X.isnull().sum())

# -------------------------------
# Train / Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Feature Scaling (MANDATORY)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Model Training
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save Model & Scaler
# -------------------------------
joblib.dump(model, "wine_cultivar_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")
