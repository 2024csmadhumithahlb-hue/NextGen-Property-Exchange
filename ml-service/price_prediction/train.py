import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


# Get ml-service folder
BASE_DIR = Path(__file__).resolve().parents[1]

# File paths
DATA_PATH = BASE_DIR / "datasets" / "properties.csv"
MODEL_PATH = BASE_DIR / "models" / "price_prediction_model.pkl"


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df)


# Input features
features = [
    "location",
    "size_sqft",
    "road_width",
    "facilities_count"
]

X = df[features]

# Target value
y = df["price"]


# Categorical feature
categorical_features = [
    "location"
]

# Numerical features
numerical_features = [
    "size_sqft",
    "road_width",
    "facilities_count"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# Random Forest Regression model
regressor = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
pipeline.fit(X_train, y_train)


# Predict test data
predictions = pipeline.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print()
print("====================================")
print("Price Prediction Model Trained!")
print("====================================")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# Save model
MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print()
print(f"Model saved at: {MODEL_PATH}")