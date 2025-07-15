import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Get root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

# Create models/ directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save(X, y, model_prefix):
    if len(y.unique()) < 2:
        print(f"Skipping {model_prefix}: Only one class present in the data.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        'logistic': LogisticRegression(max_iter=500),
        'random_forest': RandomForestClassifier(n_estimators=50)
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        model_path = os.path.join(MODEL_DIR, f"{model_prefix}_{name}.pkl")
        joblib.dump(model, model_path)

    scaler_path = os.path.join(MODEL_DIR, f"{model_prefix}_scaler.pkl")
    joblib.dump(scaler, scaler_path)


# Train on all datasets
heart = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
train_and_save(heart.drop("target", axis=1), heart["target"], "heart")

diabetes = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
train_and_save(diabetes.drop("Outcome", axis=1), diabetes["Outcome"], "diabetes")

cancer = pd.read_csv(os.path.join(DATA_DIR, "breast_cancer.csv"))
cancer["diagnosis"] = cancer["diagnosis"].map({"M": 1, "B": 0})
train_and_save(cancer.drop("diagnosis", axis=1), cancer["diagnosis"], "cancer")



