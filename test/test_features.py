from helpers.feature_engineering import total_services_col
import pandas as pd

def test_total_services_col():

    df = pd.DataFrame({
        "PhoneService": ["Yes"],
        "MultipleLines": ["Yes"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["Yes"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["Yes"],
        "TotalCharges": ["100"]
    })

    result = total_services_col(df)

    assert result["total_services"].iloc[0] == 5