
import pandas as pd
import joblib

# Load the saved model
saved = joblib.load("bengaluru_house_price_model.pkl")
model = saved["model"]
preprocessor = saved["preprocessor"]

# Ask the user for inputs
print("=== House Price Predictor ===")

location = input("Enter location: ").strip()
area = float(input("Enter area in sq ft: "))
bath = int(input("Enter number of bathrooms: "))
bhk = int(input("Enter number of bedrooms (BHK): "))

# Handle locations not known by the model
known_locations = set(
    preprocessor.named_transformers_["location"].categories_[0]
)

if location not in known_locations:
    location = "Other"

# Prepare input and predict
house = pd.DataFrame([{
    "location": location,
    "total_sqft": area,
    "bath": bath,
    "bhk": bhk
}])

house_encoded = preprocessor.transform(house)
price = model.predict(house_encoded)[0]

print(f"\nEstimated house price: ₹{price:.2f} lakhs")

