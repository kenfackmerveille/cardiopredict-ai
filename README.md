# ❤️ CardioPredict AI

Prédiction de maladies cardiaques par Machine Learning (Random Forest) — déployé sur Streamlit + Render.

## 📁 Structure du projet

```
heart-disease-predictor/
├── app.py               # Interface Streamlit
├── train_model.py       # Entraînement du modèle
├── model.pkl            # Modèle sauvegardé (généré après entraînement)
├── requirements.txt     # Dépendances Python
├── Dockerfile           # Conteneur Docker
├── deployment_link.txt  # Lien de déploiement Render
└── README.md
```

## 🚀 Lancer en local

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner le modèle
python train_model.py

# 3. Lancer l'application
streamlit run app.py
```

## 🐳 Lancer avec Docker

```bash
docker build -t cardiopredict .
docker run -p 8501:8501 cardiopredict
```

Puis ouvre : http://localhost:8501

## ☁️ Déploiement sur Render

1. Push le projet sur GitHub
2. Va sur [render.com](https://render.com) → **New Web Service**
3. Connecte ton repo GitHub
4. Configure :
   - **Build Command** : `pip install -r requirements.txt && python train_model.py`
   - **Start Command** : `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Clique sur **Deploy** — le lien apparaît automatiquement

## 📊 Dataset

**Heart Disease UCI (Cleveland)**  
- 303 patients, 13 features cliniques  
- Classification binaire : 0 = Sain, 1 = Malade cardiaque  
- Source : UCI Machine Learning Repository

## 🧠 Modèle

- **Algorithme** : Random Forest Classifier
- **Accuracy** : ~85%
- **Preprocessing** : StandardScaler
- **Split** : 80% train / 20% test (stratifié)
