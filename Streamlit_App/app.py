import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("../models/Random_Forest_House_Price.pkl")
columns = joblib.load("../models/columns.pkl")

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict its price.")

# ---------------- Numerical Inputs ----------------

area = st.number_input("Area (sq.ft)", min_value=500, max_value=20000, value=5000)

bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5,6])

bathrooms = st.selectbox("Bathrooms", [1,2,3,4,5])

stories = st.selectbox("Stories", [1,2,3,4])

parking = st.selectbox("Parking Spaces", [0,1,2,3])

# ---------------- Categorical Inputs ----------------

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished","unfurnished"]
)

# ---------------- Prediction ----------------

if st.button("Predict Price"):

    data = {
        "area":[area],
        "bedrooms":[bedrooms],
        "bathrooms":[bathrooms],
        "stories":[stories],
        "parking":[parking],
        "mainroad":[mainroad],
        "guestroom":[guestroom],
        "basement":[basement],
        "airconditioning":[airconditioning],
        "prefarea":[prefarea],
        "furnishingstatus":[furnishingstatus]
    }

    df = pd.DataFrame(data)

    # Convert categorical variables
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    st.success(f"🏡 Estimated House Price: ₹ {prediction:,.0f}")


