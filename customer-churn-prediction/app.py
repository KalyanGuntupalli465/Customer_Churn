import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Loading model files


model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
training_columns = joblib.load("models/model_columns.pkl")



st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)



st.title("📊 Customer Churn Prediction System")



# Sidebar Inputs


st.sidebar.header("Customer Details")

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

predict_button = st.sidebar.button("Predict Churn")


# Prediction Logic


if predict_button:

    # Creating dataframe
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'InternetService': [internet_service],
        'PaymentMethod': [payment_method]
    })

  

    input_data['ChargesPerMonth'] = (
        input_data['TotalCharges'] /
        (input_data['tenure'] + 1)
    )

    input_data['IsNewCustomer'] = (
        input_data['tenure'] < 6
    ).astype(int)

    input_data['IsHighSpender'] = (
        input_data['MonthlyCharges'] > 70
    ).astype(int)


    # One-hot Encoding


    input_data = pd.get_dummies(input_data)

    # Match Training Columns
   

    input_data = input_data.reindex(
        columns=training_columns,
        fill_value=0
    )


    # Scaling Numerical Features
  

    num_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'ChargesPerMonth'
    ]

    input_data[num_cols] = scaler.transform(
        input_data[num_cols]
    )


    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # Displaying Results
  

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    # Probability
    st.write(f"### Churn Probability: {probability:.2%}")

    st.progress(float(probability))

    # Risk Level
  

    if probability < 0.3:

        st.success("🟢 Low Risk Customer")

        recommendation = """
        Customer appears stable.

        Maintain regular engagement and service quality.
        """

    elif probability < 0.7:

        st.warning("🟡 Medium Risk Customer")

        recommendation = """
        Customer shows moderate churn risk.

        Consider offering loyalty rewards or discounts.
        """

    else:

        st.error("🔴 High Risk Customer")

        recommendation = """
        High churn risk detected.

        Recommended retention actions:
        - Personalized offers
        - Contract discounts
        - Customer support outreach
        """

    # Business Recommendation
   
    st.subheader("Business Recommendation")

    st.info(recommendation)



    st.subheader("Top Factors Affecting Churn")

    importance_df = pd.DataFrame({
        'Feature': training_columns,
        'Importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(
        importance_df['Feature'],
        importance_df['Importance']
    )

    ax.invert_yaxis()

    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")

    st.pyplot(fig)