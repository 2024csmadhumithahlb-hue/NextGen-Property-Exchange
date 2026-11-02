from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ============================================
# LOAD RECOMMENDATION MODEL
# ============================================

recommendation_data = joblib.load(
    MODEL_DIR / "recommendation_model.pkl"
)

recommendation_model = recommendation_data["model"]
recommendation_preprocessor = recommendation_data["preprocessor"]
properties_df = recommendation_data["data"]


# ============================================
# LOAD PRICE PREDICTION MODEL
# ============================================

price_model = joblib.load(
    MODEL_DIR / "price_prediction_model.pkl"
)


# ============================================
# LOAD CHATBOT MODEL
# ============================================

chatbot_data = joblib.load(
    MODEL_DIR / "chatbot_model.pkl"
)

chatbot_vectorizer = chatbot_data["vectorizer"]
chatbot_question_vectors = chatbot_data["question_vectors"]
chatbot_answers = chatbot_data["answers"]


# ============================================
# CREATE FASTAPI APP
# ============================================

app = FastAPI(
    title="NextGen Property Exchange ML API",
    version="1.0"
)


# ============================================
# CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================
# REQUEST MODELS
# ============================================

class RecommendationRequest(BaseModel):
    budget: float
    location: str
    plot_size: float
    purpose: str


class PricePredictionRequest(BaseModel):
    location: str
    size: float
    road_width: float
    facilities: List[str]


class ChatbotRequest(BaseModel):
    message: str


# ============================================
# HOME API
# ============================================

@app.get("/")
def home():
    return {
        "message": "NextGen Property Exchange ML Service is running"
    }


# ============================================
# RECOMMENDATION API
# ============================================

@app.post("/recommend")
def recommend_property(request: RecommendationRequest):

    input_data = pd.DataFrame([
        {
            "location": request.location,
            "size_sqft": request.plot_size,
            "purpose": request.purpose,
            "price": request.budget
        }
    ])

    input_processed = recommendation_preprocessor.transform(
        input_data
    )

    distances, indices = recommendation_model.kneighbors(
        input_processed,
        n_neighbors=min(10, len(properties_df))
    )

    results = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        property_data = properties_df.iloc[index]

        if property_data["price"] <= request.budget:

            similarity = 1 - float(distance)

            results.append({
                "id": int(property_data["id"]),
                "location": property_data["location"],
                "size_sqft": float(property_data["size_sqft"]),
                "purpose": property_data["purpose"],
                "price": float(property_data["price"]),
                "similarity_score": round(
                    similarity,
                    3
                )
            })

    return {
        "success": True,
        "count": len(results),
        "recommendations": results
    }


# ============================================
# PRICE PREDICTION API
# ============================================

@app.post("/predict-price")
def predict_price(request: PricePredictionRequest):

    facilities_count = len(request.facilities)

    input_data = pd.DataFrame([
        {
            "location": request.location,
            "size_sqft": request.size,
            "road_width": request.road_width,
            "facilities_count": facilities_count
        }
    ])

    predicted_price = price_model.predict(
        input_data
    )[0]

    predicted_price = max(
        0,
        float(predicted_price)
    )

    return {
        "success": True,
        "location": request.location,
        "size_sqft": request.size,
        "estimated_price": round(
            predicted_price,
            2
        ),
        "facilities_count": facilities_count
    }


# ============================================
# CHATBOT API
# ============================================

@app.post("/chatbot")
def chatbot(request: ChatbotRequest):

    user_message = request.message.strip()

    if not user_message:

        return {
            "success": False,
            "answer": "Please enter a question."
        }

    user_vector = chatbot_vectorizer.transform(
        [user_message]
    )

    similarities = cosine_similarity(
        user_vector,
        chatbot_question_vectors
    )[0]

    best_index = int(
        np.argmax(similarities)
    )

    best_score = float(
        similarities[best_index]
    )

    if best_score < 0.20:

        return {
            "success": True,
            "answer": "Sorry, I couldn't understand your question. Please ask about properties, plots, booking, pricing or documents.",
            "confidence": round(best_score, 3)
        }

    return {
        "success": True,
        "answer": chatbot_answers[best_index],
        "confidence": round(best_score, 3)
    }