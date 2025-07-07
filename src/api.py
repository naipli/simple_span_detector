# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Uygulama oluştur
app = FastAPI()

# Model ve vektörizer yükleniyor
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Request modeli
class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(message: Message):
    if not message.text:
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    vector = vectorizer.transform([message.text])
    prediction = model.predict(vector)[0]

    return {"prediction": prediction}