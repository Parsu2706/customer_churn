import pandas as pd 

service_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
def total_services_col(df : pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'] , errors="coerce")
    df['total_services'] = (df[service_cols] == "Yes").sum(axis=1)

    return df 
