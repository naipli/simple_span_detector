# Temel Python imajı
FROM python:3.11-slim

# Ortam değişkenleri
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Çalışma dizini oluştur
WORKDIR /app

# Gereken dosyaları kopyala
COPY . .

# Gerekli paketleri yükle
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Modeli eğit (veri varsa)
RUN python3 src/train.py

# Uvicorn ile başlat
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]