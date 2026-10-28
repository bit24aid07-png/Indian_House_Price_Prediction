import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configure the website
st.set_page_config(
    page_title="Indian House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# Load the saved ML model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "bengaluru_house_price_model.pkl"
    saved = joblib.load(model_path)
    return saved["model"], saved["preprocessor"]

model, preprocessor = load_model()

# Get known locations from the trained preprocessor
known_locations = list(
    preprocessor.named_transformers_["location"].categories_[0]
)

# Website heading
st.title("🏡 Indian House Price Prediction")
st.write(
    "Estimate a property's price in Bengaluru using Machine Learning."
)

st.divider()

# Input form
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        "Select Location",
        sorted(known_locations)
    )

    area = st.number_input(
        "Total Area (sq. ft.)",
        min_value=100.0,
        max_value=50000.0,
        value=1200.0,
        step=50.0
    )

with col2:
    bath = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=20,
        value=2
    )

    bhk = st.number_input(
        "Number of Bedrooms (BHK)",
        min_value=1,
        max_value=20,
        value=2
    )

# Prediction button
if st.button("🔍 Predict House Price", use_container_width=True):

    # Prepare input in the format expected by the model
    house = pd.DataFrame([{
        "location": location,
        "total_sqft": area,
        "bath": bath,
        "bhk": bhk
    }])

    # Apply existing preprocessing
    house_encoded = preprocessor.transform(house)

    # Predict price
    price = model.predict(house_encoded)[0]

    # Display result
    st.success("Prediction completed!")

    st.metric(
        label="Estimated House Price",
        value=f"₹ {price:.2f} Lakhs"
    )

    st.caption(
        "This is a machine learning estimate, not a guaranteed market price."
    )

st.divider()

st.caption("Built with Python, Streamlit and Scikit-learn.")
