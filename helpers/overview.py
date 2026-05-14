
import pandas as pd
import streamlit as st


def dataset_overview(df):
    st.title("Dataset Overview")
    st.subheader("Dataset Shape")
    st.write(f"Rows : {df.shape[0]}")
    st.write(f"Columns : {df.shape[1]}")

    st.subheader("Dataset Sample")
    st.dataframe(df.head(5))

    st.subheader("Columns data types")
    dtype_df = pd.DataFrame({"columns": df.columns, "data type": df.dtypes.values})
    st.dataframe(dtype_df)

    st.subheader("Statistical Summary (Numerical)")
    describe = df.describe()
    st.dataframe(describe)

    if "Churn" in df.columns:
        st.subheader("Target Variable Distribution")
        churn_counts = df["Churn"].value_counts()
        st.bar_chart(churn_counts)

    cat_col = df.select_dtypes(include="object").columns
    cat_col = cat_col.drop(["Churn", "customerID", "TotalCharges"], errors="ignore")

    for col in cat_col:
        st.subheader(col)
        value_count = df[col].value_counts()
        st.bar_chart(value_count)
