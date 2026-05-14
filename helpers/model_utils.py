import joblib

import os

model_path = os.getenv("model_path", "models/churn_pipeline.pkl")


def load_pipeline():
    return joblib.load(model_path)


def predict_churn_proba(pipeline, df):
    return pipeline.predict_proba(df)[:, 1]
