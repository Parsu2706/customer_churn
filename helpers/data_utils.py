import pandas as pd
import os


def load_default_dataset(path="data/raw/churn_dataset.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None



def load_uploaded_csv(uploaded_file):

    if uploaded_file is None:
        return None

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("Uploaded CSV is empty.")
            return None

        return df

    except EmptyDataError:
        st.error("The uploaded CSV file is empty.")
        return None

    except pd.errors.ParserError:
        st.error("Invalid CSV format.")
        return None

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def drop_customer_id(df):
    if df is None:
        return None
    if "customerID" in df.columns:
        return df.drop(columns=["customerID"])
    return df
