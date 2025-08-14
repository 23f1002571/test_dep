from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# CORS: allow frontend from Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    x: float

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict(np.array([[data.x]]))
    return {"x": data.x, "y": prediction[0]}
