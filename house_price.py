
import pandas as pd

df = pd.read_csv("Bengaluru_House_Data.csv")

print(df.head())
print("Dataset shape:", df.shape)
print(df.columns)

print("\nMissing values in each column:")
print(df.isnull().sum())


# Remove rows with missing values in important columns
df = df.dropna(subset=["location", "size", "total_sqft", "bath", "price"])

# Keep only the columns needed for our first model
df = df[["location", "size", "total_sqft", "bath", "price"]]

print("\nDataset after cleaning:")
print("Shape:", df.shape)
print(df.isnull().sum())
print(df.head())


# Extract bedroom count from the size column
df["bhk"] = df["size"].apply(
    lambda x: int(x.split()[0])
)

print("\nSize converted into numbers:")
print(df[["size", "bhk"]].head(10))

print("\nUnique bedroom counts:")
print(sorted(df["bhk"].unique()))


print("\nSample total_sqft values:")
print(df["total_sqft"].unique()[:30])

print("\nData type of total_sqft:")
print(df["total_sqft"].dtype)


def convert_sqft(value):
    parts = value.split("-")

    if len(parts) == 2:
        return (float(parts[0]) + float(parts[1])) / 2

    try:
        return float(value)
    except ValueError:
        return None


df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

print("\nConverted area values:")
print(df[["total_sqft"]].head(10))

print("\nArea data type:", df["total_sqft"].dtype)
print("Invalid area values:", df["total_sqft"].isnull().sum())


# Remove rows where area could not be converted
df = df.dropna(subset=["total_sqft"]).copy()

print("\nDataset after removing invalid area values:")
print("Shape:", df.shape)

print("\nMissing values:")
print(df[["location", "size", "total_sqft", "bath", "price", "bhk"]].isnull().sum())


print("\nPrice statistics (in lakhs):")
print(df["price"].describe())

print("\nSmallest 10 prices:")
print(df["price"].nsmallest(10).to_list())

print("\nLargest 10 prices:")
print(df["price"].nlargest(10).to_list())

print("\nBedroom counts:")
print(df["bhk"].value_counts().sort_index())


# Calculate price per square foot (price is in lakhs)
df["price_per_sqft"] = (
    df["price"] * 100000 / df["total_sqft"]
)

print("\nPrice per square foot statistics:")
print(df["price_per_sqft"].describe())

print("\nLowest 10 price-per-sqft values:")
print(df.nsmallest(10, "price_per_sqft")[
    ["location", "total_sqft", "bhk", "price", "price_per_sqft"]
])

print("\nHighest 10 price-per-sqft values:")
print(df.nlargest(10, "price_per_sqft")[
    ["location", "total_sqft", "bhk", "price", "price_per_sqft"]
])


# Keep plausible values for our first model
df = df[
    (df["total_sqft"] >= 300) &
    (df["total_sqft"] <= 10000) &
    (df["price"] > 0) &
    (df["bhk"] >= 1) &
    (df["bhk"] <= 10)
].copy()

# Recalculate price per square foot after filtering
df["price_per_sqft"] = (
    df["price"] * 100000 / df["total_sqft"]
)

print("\nDataset after filtering:")
print("Shape:", df.shape)

print("\nPrice per square foot statistics:")
print(df["price_per_sqft"].describe())

print("\nBedroom count range:")
print(df["bhk"].min(), "to", df["bhk"].max())


print("\nNumber of unique locations:")
print(df["location"].nunique())

print("\nTop 15 locations by number of listings:")
print(df["location"].value_counts().head(15))


# Count listings for each location
location_counts = df["location"].value_counts()

# Find locations with fewer than 10 listings
rare_locations = location_counts[location_counts < 10].index

print("Number of rare locations:", len(rare_locations))

# Group rare locations into "Other"
df["location"] = df["location"].apply(
    lambda x: "Other" if x in rare_locations else x
)

print("\nUnique locations after grouping:")
print(df["location"].nunique())

print("\nTop 15 locations after grouping:")
print(df["location"].value_counts().head(15))


# Select input features
X = df[["location", "total_sqft", "bath", "bhk"]]

# Select the value we want to predict
y = df["price"]

print("Input features shape:", X.shape)
print("Target shape:", y.shape)

print("\nInput features:")
print(X.head())

print("\nTarget prices:")
print(y.head())


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Convert location text into numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"),
         ["location"])
    ],
    remainder="passthrough"
)

# Learn the location categories from training data
X_train_encoded = preprocessor.fit_transform(X_train)

# Apply the same conversion to testing data
X_test_encoded = preprocessor.transform(X_test)

print("Encoded training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)


from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model using training data
model.fit(X_train_encoded, y_train)

print("Model training completed successfully!")


# Predict prices for test houses
y_pred = model.predict(X_test_encoded)

print("First 10 actual prices (in lakhs):")
print(y_test.head(10).to_list())

print("\nFirst 10 predicted prices (in lakhs):")
print(y_pred[:10].round(2).tolist())


from sklearn.metrics import mean_absolute_error, r2_score

# Calculate average absolute prediction error
mae = mean_absolute_error(y_test, y_pred)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", round(mae, 2), "lakhs")
print("R-squared score:", round(r2, 4))


# Predict the average training price for every test house
baseline_pred = [y_train.mean()] * len(y_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)

print("Baseline MAE:", round(baseline_mae, 2), "lakhs")
print("Linear Regression MAE:", round(mae, 2), "lakhs")


from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train_encoded, y_train)

# Predict test prices
rf_pred = rf_model.predict(X_test_encoded)

# Evaluate the predictions
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("MAE:", round(rf_mae, 2), "lakhs")
print("R-squared score:", round(rf_r2, 4))


print("\nPrice percentiles (in lakhs):")
print(df["price"].quantile([0.50, 0.75, 0.90, 0.95, 0.99]))

print("\nHouses priced above 300 lakhs:")
print((df["price"] > 300).sum())

print("\nHouses priced above 500 lakhs:")
print((df["price"] > 500).sum())


# Inspect houses with very high prices per square foot
print("\nHouses above ₹50,000 per square foot:")

print(
    df[df["price_per_sqft"] > 50000][
        ["location", "total_sqft", "bath", "bhk",
         "price", "price_per_sqft"]
    ].head(10)
)

print("\nNumber of houses above ₹50,000 per square foot:")
print((df["price_per_sqft"] > 50000).sum())


print("\nFinal model comparison")

print("Baseline MAE:", round(baseline_mae, 2), "lakhs")
print("Linear Regression MAE:", round(mae, 2), "lakhs")
print("Linear Regression R²:", round(r2, 4))

print("Random Forest MAE:", round(rf_mae, 2), "lakhs")
print("Random Forest R²:", round(rf_r2, 4))


# Function to predict a Bengaluru house price
def predict_house_price(location, total_sqft, bath, bhk):
    house = pd.DataFrame([{
        "location": location,
        "total_sqft": total_sqft,
        "bath": bath,
        "bhk": bhk
    }])

    house_encoded = preprocessor.transform(house)
    predicted_price = rf_model.predict(house_encoded)[0]

    return round(predicted_price, 2)


# Test with an example house
price = predict_house_price(
    location="Whitefield",
    total_sqft=1200,
    bath=2,
    bhk=2
)

print("\nHouse Price Prediction")
print("Location: Whitefield")
print("Area: 1200 sq ft")
print("Bathrooms: 2")
print("Bedrooms: 2")
print("Predicted price:", price, "lakhs")


import joblib

# Save the trained model and location encoder together
joblib.dump(
    {
        "model": rf_model,
        "preprocessor": preprocessor
    },
    "bengaluru_house_price_model.pkl"
)

print("\nModel saved successfully!")


print("\n=== Bengaluru House Price Predictor ===")

location = input("Enter location: ")
area = float(input("Enter house area in sq ft: "))
bath = int(input("Enter number of bathrooms: "))
bhk = int(input("Enter number of bedrooms (BHK): "))

predicted_price = predict_house_price(
    location, area, bath, bhk
)

print("\nEstimated House Price: ₹", predicted_price, "lakhs")
