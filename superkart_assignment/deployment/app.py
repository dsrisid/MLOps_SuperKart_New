import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="dsrisid/super-kart-sales-prediction/superkart_revenue_model", filename="best_superkart_revenue_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("SuperKart Revenue Prediction")
st.write("""
This application predicts the expected **revenue** of SuperKart store
based on its characteristics such as Product_Weight, Product_Allocated_Area, Product_MRP, and Store_Establishment_Year.
Please enter the details below to get a revenue prediction.
""")

# User input
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Type = st.selectbox("Product Type", ["Fruits and Vegetables", "Snack Foods","Frozen Foods","Dairy","Household","Baking Goods","Canned","Health and Hygiene","Meat","Soft Drinks","Breads","Hard Drinks","Others","Starchy Foods","Breakfast","Seafood"])
Store_Size = st.selectbox("Store Size", ["Medium", "High", "Small"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"])


Product_Weight = st.number_input("Product weight", min_value=1.0, max_value=25.0, value=10.0, step=1)
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0001, max_value=0.300, value=0.0, step=0.1)
Product_MRP = st.number_input("Product MRP", min_value=10, max_value=300, value=50, step=10)
Store_Establishment_Year = st.number_input("Store Establishment Year", min_value=1987, max_value=2050, value=2000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Type': Product_Type,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type,
    'Product_Weight': Product_Weight,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
}])

# Predict button
if st.button("Predict Revenue"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Revenue: **${prediction:,.2f} USD**")
