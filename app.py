import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_credit_model.pkl") # load the best model which is xgboost here
encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

st.title("Credit Risk Prediction App")

st.write("Enter applicant information to predict if the credit risk is good or bad")

# Next we are going to define the input files
age = st.number_input("Age", min_value = 18, max_value = 80, value = 30) # default value is 30
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value = 0, max_value = 3, value = 1) # default value = 1
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value = 0, value = 12) # lets not put max value to this
duration = st.number_input("Duration (months)", min_value = 1, value = 12)

# Now, lets prepare the input for the model
input_df = pd.DataFrame({
    "Age" : [age],
    "Sex" : [encoders["Sex"].transform([sex])[0]],
    "Job" : [job],
    "Housing" : [encoders["Housing"].transform([housing])[0]],
    "Saving accounts" : [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account" : [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount" : [credit_amount],
    "Duration" : [duration]
})

# Next going to create a prediction button
if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    
    if pred == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")