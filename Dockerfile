# ─────────────────────────────────────────────
# Dockerfile — CardioPredict AI (Streamlit)
# ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers du projet
COPY . .

# Variables d'environnement
ENV PYTHONUNBUFFERED=1

# Exposer le port Streamlit
EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501","--server.address=0.0.0.0", "--server.headless=true"]