import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from PIL import Image
import io

# ==========================================================
# Load Trained Model
# ==========================================================
model_path = "models/car_price_model.pkl"

st.set_page_config(page_title="🚗 Car Price Prediction", layout="wide")

st.title("🚗 Car Price Prediction App")
st.write("Upload a car photo and enter details below to estimate its resale price.")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Trained model not found! Please run `model_training.py` first.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# ==========================================================
# Car Image Upload + Analyzer
# ==========================================================
st.markdown("### 🖼️ Car Photo Analyzer")

uploaded_file = st.file_uploader("Upload Car Image (optional)", type=["jpg", "jpeg", "png"])

image_condition_score = 0  # Default no change

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Car Image", use_container_width=True)

    # Simulated "visual condition" scoring (mock analysis)
    # You can later replace this with a CNN model for real analysis
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    size_score = len(file_bytes) / 10000  # simple heuristic
    image_condition_score = min(size_score, 1.0)  # between 0 and 1

    if image_condition_score > 0.8:
        st.success("✅ The car appears to be in excellent visual condition.")
    elif image_condition_score > 0.5:
        st.info("ℹ️ The car appears in good condition.")
    else:
        st.warning("⚠️ The image suggests an older or unclear car condition.")

# ==========================================================
# User Input Section
# ==========================================================
st.markdown("---")
st.subheader("📋 Enter Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Select Car Brand", [
        "Maruti", "Hyundai", "Tata", "Honda", "Mahindra", "Kia",
        "Toyota", "Ford", "Renault", "Volkswagen", "Nissan",
        "Skoda", "BMW", "Audi", "Mercedes", "MG", "Jaguar",
        "Porsche", "Ferrari", "Bentley", "Citroen", "Volvo", "Jeep"
    ])

with col2:
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

with col3:
    engine_size = st.number_input("Engine Size (in CC)", min_value=600, max_value=5000, value=1197)

mileage = st.number_input("Mileage (in km)", min_value=1000, max_value=300000, value=50000)
model_year = st.number_input("Model Year", min_value=1995, max_value=datetime.now().year, value=2018)

# ==========================================================
# Prepare Input Data
# ==========================================================
age = datetime.now().year - model_year

input_data = pd.DataFrame({
    "age": [age],
    "mileage": [mileage],
    "engine_size": [engine_size],
    "brand": [brand],
    "transmission": [transmission]
})

# One-hot encoding (same as training)
input_encoded = pd.get_dummies(input_data, columns=["brand", "transmission"], drop_first=True)

# Align features with model
try:
    model_columns = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
except Exception:
    input_encoded = input_encoded.reindex(columns=input_encoded.columns, fill_value=0)

# ==========================================================
# Predict Button
# ==========================================================
if st.button("🔍 Predict Car Price", key="predict_price"):
    try:
        base_price = model.predict(input_encoded)[0]

        # Adjust based on image condition (simple multiplier)
        adjusted_price = base_price * (1 + 0.1 * image_condition_score)

        st.success(f"💰 Estimated Car Price: ₹{adjusted_price:,.2f}")
        st.caption(f"(Adjusted for visual condition score: {image_condition_score:.2f})")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# ==========================================================
# Brand Logos Display
# ==========================================================
st.markdown("---")
st.subheader("🏷️ Popular Car Brands")

logos = {
    "Maruti": "https://upload.wikimedia.org/wikipedia/en/3/3b/Maruti_Suzuki_logo.svg",
    "Hyundai": "https://upload.wikimedia.org/wikipedia/commons/4/45/Hyundai_logo.svg",
    "Tata": "https://upload.wikimedia.org/wikipedia/en/2/2e/Tata_logo.svg",
    "Honda": "https://upload.wikimedia.org/wikipedia/en/7/70/Honda-logo.svg",
    "Kia": "https://upload.wikimedia.org/wikipedia/commons/0/0d/Kia_logo.svg",
    "BMW": "https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg",
    "Mercedes": "https://upload.wikimedia.org/wikipedia/commons/9/90/Mercedes-Logo.svg",
    "Toyota": "https://upload.wikimedia.org/wikipedia/commons/9/9d/Toyota_carlogo.svg"
}

cols = st.columns(4)
for i, (name, url) in enumerate(logos.items()):
    with cols[i % 4]:
        st.image(url, width=80)
        st.caption(name)
