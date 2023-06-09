import os
import uvicorn
import traceback
import pandas as pd
import numpy as np
import pickle

from pydantic import BaseModel
# from urllib.request import Request
from fastapi import FastAPI, Response
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Hello world from ML endpoint!"

class RequestId(BaseModel):
    idx:int

@app.post("/load_encodings")
def load_encodings():
    file_path = "encodings.pickle"
    with open(file_path, "rb") as pkl:
        encodings = pd.DataFrame(pickle.load(pkl))
    encodings_matrix = np.vstack(encodings["encodings"].to_numpy())    
    return encodings, encodings_matrix

@app.post("/get_recommendations")
def get_recommendations(req: RequestId, response: Response, encodings, encodings_matrix):
    try:
        workout_index = req.idx
        workout_encoding = np.reshape(encodings.iloc[workout_index]['encodings'], (1, -1))
        sim_scores = cosine_similarity(workout_encoding, encodings_matrix)[0]
        sim_scores = pd.DataFrame(sim_scores, columns=['similarity_score'])
        sim_scores = sim_scores.sort_values(by='similarity_score', ascending=False)
        sim_scores = sim_scores.drop(workout_index)
        top3 = sim_scores.head(3)
        return top3.index
    
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)