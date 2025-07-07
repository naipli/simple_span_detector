# src/train.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Veri yükle
df = pd.read_csv('data/emails.csv')

# Özellik ve hedef
X = df['text']
y = df['label']

# Vektörleştir
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# Model eğit
model = MultinomialNB()
model.fit(X_vect, y)

# Model ve vektör kaydet
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model başarıyla eğitildi ve kaydedildi.")