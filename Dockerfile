# ─────────────────────────────────────────────
# Dockerfile — CardioPredict AI
# ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entraîner le modèle lors du build (génère model.pkl)
RUN python train_model.py

# Exposer le port FastAPI
EXPOSE 8000

# Variables d'environnement
ENV PYTHONUNBUFFERED=1

# Lancer FastAPI via uvicorn
# Pour Streamlit à la place, utilise :
#   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]