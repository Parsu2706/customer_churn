import pandas as pd

def load_default_dataset(path=r"C:\Users\Prasad.LAPTOP-R00KVI21\OneDrive\Desktop\Customer_churn_pred\data\raw\churn_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    if "customerID" in df.columns:
        return df.drop(columns=["customerID"])
    return df
