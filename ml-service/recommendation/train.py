import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors


# Get the ml-service folder path
BASE_DIR = Path(__file__).resolve().parents[1]


# File paths
DATA_PATH = BASE_DIR / "datasets" / "properties.csv"
MODEL_PATH = BASE_DIR / "models" / "recommendation_model.pkl"


# Load property data
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df)


# Features used for recommendation
features = [
    "location",
    "size_sqft",
    "purpose",
    "price"
]

X = df[features]


# Categorical features
categorical_features = [
    "location",
    "purpose"
]

# Numerical features
numerical_features = [
    "size_sqft",
    "price"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# Convert data into ML format
X_processed = preprocessor.fit_transform(X)


# Create recommendation model
model = NearestNeighbors(
    n_neighbors=min(10, len(df)),
    metric="cosine"
)

model.fit(X_processed)


# Create models folder if it does not exist
MODEL_PATH.parent.mkdir(exist_ok=True)


# Save the model
joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "data": df
    },
    MODEL_PATH
)


print()
print("====================================")
print("Recommendation model trained!")
print("====================================")
print(f"Model saved at: {MODEL_PATH}")