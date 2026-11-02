import json
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# Get ml-service folder
BASE_DIR = Path(__file__).resolve().parents[1]

# File paths
DATA_PATH = BASE_DIR / "datasets" / "faq.json"
MODEL_PATH = BASE_DIR / "models" / "chatbot_model.pkl"


# Load FAQ data
with open(DATA_PATH, "r", encoding="utf-8") as file:
    faq_data = json.load(file)


# Extract questions and answers
questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)


# Convert questions into vectors
question_vectors = vectorizer.fit_transform(questions)


# Make sure models folder exists
MODEL_PATH.parent.mkdir(exist_ok=True)


# Save the chatbot model
joblib.dump(
    {
        "vectorizer": vectorizer,
        "question_vectors": question_vectors,
        "questions": questions,
        "answers": answers
    },
    MODEL_PATH
)


print("====================================")
print("Chatbot model trained successfully!")
print("====================================")
print(f"Model saved at: {MODEL_PATH}")