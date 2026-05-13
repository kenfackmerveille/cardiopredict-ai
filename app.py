import streamlit as st
import numpy as np
import pickle
import os

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fond dégradé */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Titre principal */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.sub-title {
    text-align: center;
    color: #a0aec0;
    font-size: 1rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Carte de résultat */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    font-family: 'Syne', sans-serif;
}

.result-danger {
    background: linear-gradient(135deg, #ff4757, #c0392b);
    border: 2px solid #ff6b6b;
    box-shadow: 0 0 30px rgba(255, 71, 87, 0.4);
}

.result-safe {
    background: linear-gradient(135deg, #2ed573, #1abc9c);
    border: 2px solid #6bcb77;
    box-shadow: 0 0 30px rgba(46, 213, 115, 0.4);
}

.result-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
}

.result-prob {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
}

/* Section */
.section-label {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #e2e8f0;
    font-size: 1rem;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid #ff6b6b;
    padding-left: 0.6rem;
}

/* Bouton */
.stButton > button {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    margin-top: 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(255, 107, 107, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255, 107, 107, 0.6) !important;
}

/* Sliders & inputs */
.stSlider > div > div {
    background-color: #ff6b6b !important;
}

/* Fond des widgets */
.stSelectbox > div > div, .stNumberInput > div > div {
    background-color: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Modèle introuvable. Lance d'abord `python train_model.py`.")
        st.stop()
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

model, scaler = load_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">❤️ CardioPredict AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Détection de maladies cardiaques par Machine Learning</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# FORMULAIRE
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">👤 Informations personnelles</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Âge", 20, 80, 50)
with col2:
    sex = st.selectbox("Sexe", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")

st.markdown('<div class="section-label">🩺 Paramètres cardiaques</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    cp = st.selectbox("Type de douleur thoracique", options=[0, 1, 2, 3],
                      format_func=lambda x: ["Asymptomatique", "Angine typique", "Angine atypique", "Douleur non angineuse"][x])
    trestbps = st.slider("Pression artérielle au repos (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholestérol sérique (mg/dl)", 100, 600, 240)
with col4:
    fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
    restecg = st.selectbox("Résultats ECG au repos", options=[0, 1, 2],
                           format_func=lambda x: ["Normal", "Anomalie ST-T", "Hypertrophie ventriculaire"][x])
    thalach = st.slider("Fréquence cardiaque maximale", 60, 210, 150)

st.markdown('<div class="section-label">📈 Tests d\'effort</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    exang = st.selectbox("Angine induite par l'effort", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
    oldpeak = st.slider("Dépression ST (effort vs repos)", 0.0, 6.5, 1.0, step=0.1)
with col6:
    slope = st.selectbox("Pente du segment ST", options=[0, 1, 2],
                         format_func=lambda x: ["Montante", "Plate", "Descendante"][x])
    ca = st.selectbox("Nb de vaisseaux colorés (fluoroscopie)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassémie", options=[0, 1, 2, 3],
                        format_func=lambda x: ["Normal", "Inconnu", "Défaut fixe", "Défaut réversible"][x])

# ─────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────
if st.button("🔍 Analyser mon profil cardiaque"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    prob_disease = proba[1] * 100
    prob_healthy = proba[0] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-danger">
            <div class="result-title">⚠️ Risque de maladie cardiaque détecté</div>
            <div class="result-prob">Probabilité : <strong>{prob_disease:.1f}%</strong> — Consultez un cardiologue.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-safe">
            <div class="result-title">✅ Profil cardiaque sain</div>
            <div class="result-prob">Probabilité d'être sain : <strong>{prob_healthy:.1f}%</strong> — Continuez vos bonnes habitudes !</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Détail des probabilités")
    col_a, col_b = st.columns(2)
    col_a.metric("🟢 Sain", f"{prob_healthy:.1f}%")
    col_b.metric("🔴 Malade", f"{prob_disease:.1f}%")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#718096; font-size:0.8rem;'>"
    "⚠️ Cet outil est à des fins éducatives uniquement. Il ne remplace pas un avis médical professionnel."
    "</p>",
    unsafe_allow_html=True
)
