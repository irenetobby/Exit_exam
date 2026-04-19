import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("house_price_model.pkl", "rb"))
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)
st.title("🏠 Bengaluru House Price Prediction")

# Input fields
sqft = st.number_input("Total Area (sqft)", min_value=500, max_value=10000, step=50)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)
pps = st.number_input("Price per sqft", min_value=1000, max_value=100000, step=100)

# Prediction button
if st.button("Predict"):
    features = np.array([[sqft, bhk, pps]])
    prediction = model.predict(features)
    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")
