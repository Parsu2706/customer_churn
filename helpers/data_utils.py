import pandas as pd
import os 

def load_default_dataset(path="data/raw/churn_dataset.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None

def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    if "customerID" in df.columns:
        return df.drop(columns=["customerID"])
    return df
