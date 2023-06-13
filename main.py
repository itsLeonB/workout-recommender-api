import os
import logging
import uvicorn
import pandas as pd
import numpy as np
import pickle

from typing import List
from fastapi import FastAPI, Path
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
logger = logging.getLogger(__name__)

def load_encodings(file_path: str):
    # Function to load the encodings from a pickle file
    with open(file_path, 'rb') as pkl:
        encodings = pickle.load(pkl)
    encodings_matrix = np.vstack(encodings['encodings'].to_numpy())
    return encodings, encodings_matrix

def get_recommendations(workout_index: int, encodings, encodings_matrix) -> List[int]:
    # Function to get recommendations based on workout index and encodings
    workout_encoding = np.reshape(encodings.iloc[workout_index]['encodings'], (1, -1))
    sim_scores = cosine_similarity(workout_encoding, encodings_matrix)[0]
    sim_scores = pd.DataFrame(sim_scores, columns=['similarity_score'])
    sim_scores = sim_scores.sort_values(by='similarity_score', ascending=False)
    sim_scores = sim_scores.drop(workout_index)
    top_recommendations = sim_scores.head(3)
    return top_recommendations.index.tolist()

encodings, encodings_matrix = load_encodings('encodings.pickle')

@app.get("/")
def index():
    # Test or health check endpoint
    return "Hello world from ML endpoint!"

@app.get("/recommendations/{workout_index}")
def get_recommendations_endpoint(workout_index: int = Path(..., ge=0, lt=len(encodings))):
    try:
        recommendations = get_recommendations(workout_index, encodings, encodings_matrix)
        return {
            "recommendations": recommendations
        }
    except Exception as e:
        logger.exception("Error occurred while getting recommendations")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = os.environ.get("PORT", 8080)
    logger.info(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host='0.0.0.0', port=port)