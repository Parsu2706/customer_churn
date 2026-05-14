import pandas as pd
import streamlit as st

from helpers.overview import dataset_overview
from helpers.data_utils import load_default_dataset, load_uploaded_csv, drop_customer_id
from helpers.feature_engineering import total_services_col
from helpers.model_utils import load_pipeline, predict_churn_proba


st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

page = st.sidebar.selectbox("Navigation", ["Dataset Overview", "Churn Prediction"])

uploaded_file = st.sidebar.file_uploader("Upload Csv", type="csv")


def get_dataset():
    if uploaded_file:
        df = load_uploaded_csv(uploaded_file)
        st.sidebar.success("CSV uploaded")
    else:
        df = load_default_dataset()
        if df is None:
            st.sidebar.warning("No default dataset found . Please upload a csv")
            return None

    df = drop_customer_id(df)
    return df


if page == "Dataset Overview":
    df = get_dataset()
    if df is None:
        st.info("Please upload a CSV file to view dataset overview.")
        st.stop()
    dataset_overview(df)

if page == "Churn Prediction":
    pipeline = load_pipeline()
    st.title("Customer Churn Prediction")
    st.header("Customer Input")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Customer Information"):
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider(
                "Tenure (months)", min_value=0, max_value=72, value=12, step=1
            )
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )

    with col2:
        with st.expander("Internet and Streaming services"):
            internet_service = st.selectbox(
                "Internet services", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "online_security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "online_backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "device_protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "tech_support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "streaming_tv", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "streaming_movies", ["No", "Yes", "No internet service"]
            )

    with st.expander("Other Information"):
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.slider(
            "Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0
        )
        total_charges = st.slider(
            "Total Charges", min_value=0.0, max_value=20000.0, value=0.0
        )

        if total_charges == 0.0 and tenure > 0:
            total_charges = tenure * monthly_charges

    input_df = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )
    input_df = total_services_col(input_df)

    if st.button("Predict Churn"):
        probability = predict_churn_proba(pipeline, input_df)[0] * 100

        if probability >= 70:
            st.error(f"High Risk of Churn : {probability:.1f}%")
        elif probability >= 40:
            st.warning(f"Medium Risk of churn:{probability:.1f}%")
        else:
            st.warning(f"Low Risk if Churn:{probability:.1f}%")

st.subheader("Batch Prediction")
if uploaded_file:

    batch_df = get_dataset()

    if batch_df is None:
        st.stop()

    df_batch = total_services_col(batch_df)

    probs = predict_churn_proba(pipeline, df_batch)

    df_batch["Churn_Probability"] = probs * 100
    df_batch = total_services_col(batch_df)
    probs = predict_churn_proba(pipeline, df_batch)
    df_batch["Churn_Probability"] = probs * 100

    def risk(p):
        if p >= 70:
            return "High Risk"
        elif p >= 40:
            return "Medium Risk"
        else:
            return "Low Risk"

    df_batch["Risk_Level"] = df_batch["Churn_Probability"].apply(risk)
    st.dataframe(df_batch)
    st.download_button(
        "Download_Predictions",
        df_batch.to_csv(index=False),
        "churn_prediction.csv",
        "text/csv",
    )
    value_count = df_batch["Risk_Level"].value_counts()

    st.bar_chart(value_count)
