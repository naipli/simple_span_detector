# src/predict.py
import joblib

# Model ve vektör yükle
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Kullanıcıdan giriş al
text = input("Mesajı girin: ")

# Vektöre çevir
text_vect = vectorizer.transform([text])

# Tahmin et
prediction = model.predict(text_vect)[0]

print(f"Tahmin: {prediction.upper()}")