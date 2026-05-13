from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

# ─────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ model.pkl introuvable. Lance d'abord : python train_model.py")

with open(MODEL_PATH, "rb") as f:
    data    = pickle.load(f)
    model   = data["model"]
    scaler  = data["scaler"]
    FEATURES = data["features"]

# ─────────────────────────────────────────────
# INITIALISATION FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="CardioPredict AI — API",
    description="""
## ❤️ API de prédiction de maladies cardiaques

Cette API expose un modèle **Random Forest** entraîné sur le dataset
**Heart Disease UCI (Cleveland)** pour prédire la présence d'une maladie cardiaque.

### Endpoints disponibles
- `GET  /`          — Message de bienvenue
- `GET  /health`    — Vérification que l'API est active
- `POST /predict`   — Prédiction à partir des données du patient
- `GET  /features`  — Liste des features attendues
""",
    version="1.0.0",
)

# CORS — permet les appels depuis Streamlit ou n'importe quel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# SCHÉMA D'ENTRÉE (Pydantic)
# ─────────────────────────────────────────────
class PatientData(BaseModel):
    age:      int   = Field(..., ge=1,   le=120, example=52,   description="Âge du patient")
    sex:      int   = Field(..., ge=0,   le=1,   example=1,    description="Sexe : 0=Femme, 1=Homme")
    cp:       int   = Field(..., ge=0,   le=3,   example=0,    description="Type douleur thoracique (0-3)")
    trestbps: int   = Field(..., ge=80,  le=220, example=125,  description="Pression artérielle au repos (mm Hg)")
    chol:     int   = Field(..., ge=100, le=600, example=212,  description="Cholestérol sérique (mg/dl)")
    fbs:      int   = Field(..., ge=0,   le=1,   example=0,    description="Glycémie à jeun > 120 mg/dl : 0=Non, 1=Oui")
    restecg:  int   = Field(..., ge=0,   le=2,   example=0,    description="Résultats ECG au repos (0-2)")
    thalach:  int   = Field(..., ge=60,  le=220, example=168,  description="Fréquence cardiaque maximale atteinte")
    exang:    int   = Field(..., ge=0,   le=1,   example=0,    description="Angine induite par effort : 0=Non, 1=Oui")
    oldpeak:  float = Field(..., ge=0.0, le=7.0, example=1.0,  description="Dépression ST (effort vs repos)")
    slope:    int   = Field(..., ge=0,   le=2,   example=2,    description="Pente segment ST (0=montante, 1=plate, 2=descendante)")
    ca:       int   = Field(..., ge=0,   le=3,   example=0,    description="Nb vaisseaux colorés (fluoroscopie) (0-3)")
    thal:     int   = Field(..., ge=1,   le=3,   example=2,    description="Thalassémie : 1=normal, 2=défaut fixe, 3=défaut réversible")

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Général"])
def root():
    return {
        "message": "Bienvenue sur CardioPredict AI 🫀",
        "docs": "/docs",
        "predict": "/predict"
    }


@app.get("/health", tags=["Général"])
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/features", tags=["Modèle"])
def get_features():
    """Retourne la liste des features attendues par le modèle."""
    return {"features": FEATURES, "count": len(FEATURES)}


@app.post("/predict", tags=["Prédiction"])
def predict(patient: PatientData):
    """
    Prédit si un patient est atteint d'une maladie cardiaque.

    - **0** → Pas de maladie cardiaque détectée ✅
    - **1** → Maladie cardiaque probable ⚠️
    """
    # Mise en forme
    input_array = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps,
        patient.chol, patient.fbs, patient.restecg, patient.thalach,
        patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]])

    # Normalisation
    input_scaled = scaler.transform(input_array)

    # Prédiction
    prediction = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0].tolist()

    return {
        "prediction": prediction,
        "label": "Maladie cardiaque probable ⚠️" if prediction == 1 else "Pas de maladie détectée ✅",
        "probability": {
            "sain":   round(probabilities[0] * 100, 2),
            "malade": round(probabilities[1] * 100, 2),
        },
        "input_received": patient.model_dump(),
    }
