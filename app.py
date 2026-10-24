import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("logistic_model.pkl")

@st.cache_data
def load_data():
    return joblib.load("df_reference.pkl")

@st.cache_data
def load_features():
    numerical = joblib.load("numerical_features.pkl")
    categorical = joblib.load("categorical_features.pkl")
    return numerical, categorical

model = load_model()
df = load_data()
numerical_features, categorical_features = load_features()

TARGET = "Churned"
feature_columns = [c for c in df.columns if c != TARGET]

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details below to predict whether "
    "the customer is likely to churn."
)

with st.form("churn_form"):
    input_data = {}

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):
        container = col1 if i % 2 == 0 else col2

        with container:
            if feature in categorical_features or (
                feature in df.columns
                and (
                    df[feature].dtype == "object"
                    or isinstance(df[feature].dtype, pd.CategoricalDtype)
                    or df[feature].dtype == "bool"
                )
            ):
                options = df[feature].dropna().unique().tolist()

                if options:
                    input_data[feature] = st.selectbox(
                        feature.replace("_", " ").title(),
                        options=options
                    )
                else:
                    input_data[feature] = st.text_input(
                        feature.replace("_", " ").title()
                    )

            else:
                values = pd.to_numeric(
                    df[feature], errors="coerce"
                ).dropna()

                if values.empty:
                    input_data[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        value=0.0
                    )
                else:
                    minimum = float(values.min())
                    maximum = float(values.max())
                    default = float(values.median())

                    input_data[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=min(minimum, default),
                        max_value=max(maximum, default),
                        value=default
                    )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")
        st.write(f"**Predicted class:** {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.write("**Prediction probabilities:**")

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability (%)": (probabilities * 100).round(2)
            })
            st.dataframe(probability_df, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Please check that the saved model, feature columns, "
            "and package versions match your training notebook."
        )
      
