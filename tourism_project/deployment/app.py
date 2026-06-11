import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="jayaprakashp1/tourism-package-prediction", filename="best_tourism_package_prediction_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for 'Visit With us' staff  that predicts whether the customers are likely to purchase the 'Wellness Tourism package' or not.")
st.write("Kindly enter the customer details to check whether they are likely to purchase or not.")

# Collect user input
Age = st.number_input("Age (customer's age in years) ", min_value=10, max_value=100, value=65)
TypeofContact = st.selectbox("Type of Contact ", ["Self Enquiry", "Company Invited"])
CityTier = st.number_input("City Tier ", min_value=1, max_value=5, value=3)
DurationOfPitch = st.number_input("Duration of Pitch", value=32.0)
Occupation = st.selectbox("Occupation ", ["Salaried", "Small Business","Large Business"])
Gender = st.selectbox("Gender ", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting ", value = 3)
NumberOfFollowups = st.number_input("Number Of Followups ", value = 3.0)
ProductPitched = st.selectbox("Product Pitched ", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.number_input("Preferred Property Star ", value = 4.0)
MaritalStatus = st.selectbox("Marital Status ", ["Married", "Divorced", "Unmarried", "Single"])
NumberOfTrips = st.number_input("Number Of Trips ", value = 20.0)
Passport = st.number_input("Passport(Customer has a passport or not?) ", value = 1.0)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score ", value = 4.0)
OwnCar = st.number_input("Own Car(Customer has a car or not?) ", value = 1.0)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting ", value = 3.0)
Designation = st.selectbox("Designation ", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income ", value = 21000.0)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': 1 if Gender == "Female" else 0,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
