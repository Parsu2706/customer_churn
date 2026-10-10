import json
import os

import joblib

model_path = os.getenv("model_path", "models/churn_pipeline.pkl")
threshold_path = os.getenv("threshold_path", "models/threshold.json")


def load_pipeline():
    return joblib.load(model_path)


def load_threshold(default = 0.5):
    try:
        with open(threshold_path) as f:
            return float(json.load(f)["threshold"])
    except (FileNotFoundError , KeyError , ValueError):

        return default

def predict_churn_proba(pipeline, df):
    return pipeline.predict_proba(df)[:, 1]


def risk_level(prob , threshold , high = 0.5):
    if prob>= high:
        return "High Risk"

    if prob >= threshold:
        return "Medium Risk"

    return "Low Risk"
