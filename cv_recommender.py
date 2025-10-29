# cv_recommender.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import os
import warnings
from io import StringIO
from PyPDF2 import PdfReader

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================
# ⚙️ CONFIGURATION
# ==============================
st.set_page_config(page_title="AI CV Recommender", layout="wide")
st.title("🤖 Prédiction & Recommandation à partir des CVs (Kaggle Dataset)")

# ==============================
# 📂 CHARGEMENT DU DATASET KAGGLE
# ==============================
@st.cache_data
def load_kaggle_dataset():
    st.info("📂 Chargement du dataset Kaggle : Resume.csv ...")

    if not os.path.exists("Resume.csv"):
        st.error("❌ Le fichier 'Resume.csv' est introuvable dans le dossier actuel.")
        st.stop()

    df = pd.read_csv("Resume.csv", encoding="utf-8", on_bad_lines="skip")

    # Détection de la colonne contenant le texte du CV
    resume_col = None
    for col in df.columns:
        if "resume" in col.lower():
            resume_col = col
            break
    if resume_col is None:
        resume_col = df.columns[0]

    df.rename(columns={resume_col: "Resume"}, inplace=True)

    # Ajoute une colonne "Category" si elle n'existe pas
    if "Category" not in df.columns:
        df["Category"] = "Unknown"

    # Nettoyage
    df.dropna(subset=["Resume"], inplace=True)
    df = df[df["Resume"].str.len() > 50].reset_index(drop=True)

    st.success(f"✅ {len(df)} CVs chargés avec succès depuis Kaggle.")
    return df


df = load_kaggle_dataset()

# ==============================
# 🧠 CHARGEMENT DU MODÈLE
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()

# ==============================
# 🔢 ENCODAGE DES CVs
# ==============================
@st.cache_data(show_spinner=True)
def compute_embeddings(resumes, _model):
    embeddings = _model.encode(resumes, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


embeddings = compute_embeddings(df["Resume"].astype(str).tolist(), _model=model)

# ==============================
# 🎯 PRÉDICTION DE DOMAINE (FIT POSTE)
# ==============================
st.header("🎯 Prédiction de l’adéquation au poste (Clustering automatique)")

num_clusters = st.slider("Nombre de domaines de métiers à détecter :", 3, 15, 8)

if st.button("🚀 Lancer la prédiction des domaines"):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    df["Predicted_Domain"] = labels

    st.success("✅ Domaines de métiers détectés avec succès !")

    # ✅ Correction du paramètre de couleurs pour Plotly
    fig = px.histogram(
    df,
    x="Predicted_Domain",
    color="Predicted_Domain",
    title="Distribution des domaines prédits",
    color_discrete_sequence=px.colors.qualitative.Set3  # ✅ palette qualitative stable
)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Exemples par domaine :")
    for c in range(num_clusters):
        st.markdown(f"### 🧩 Domaine {c}")
        subset = df[df["Predicted_Domain"] == c]
        examples = subset.sample(min(2, len(subset))).reset_index(drop=True)
        for i, row in examples.iterrows():
            st.markdown(f"• {row['Resume'][:400]}...")
        st.markdown("---")

# ==============================
# 💡 SYSTÈME DE RECOMMANDATION
# ==============================
st.header("💡 Système de recommandation à partir d’un CV")

st.markdown("""
🧠 Vous pouvez **coller le texte d’un CV** ci-dessous **ou uploader un fichier .txt / .pdf**
pour trouver les **CVs les plus similaires** et estimer le **domaine professionnel probable**.
""")

uploaded_file = st.file_uploader("📄 Upload du CV (.txt ou .pdf)", type=["txt", "pdf"])
cv_input = ""

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == ".txt":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        cv_input = stringio.read()
    elif file_extension == ".pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        cv_input = text

# Si l’utilisateur veut coller le texte manuellement
manual_text = st.text_area("Ou collez ici le texte d’un CV :", height=200)
if manual_text.strip():
    cv_input = manual_text.strip()

if st.button("🔎 Trouver les CVs similaires"):
    if not cv_input.strip():
        st.warning("Veuillez coller un CV ou uploader un fichier pour lancer la recherche.")
    else:
        query_emb = model.encode([cv_input])[0]
        sims = cosine_similarity([query_emb], embeddings)[0]
        scaler = MinMaxScaler()
        sims_scaled = scaler.fit_transform(sims.reshape(-1, 1)).flatten()

        df_rec = pd.DataFrame({
            "Resume": df["Resume"],
            "Category": df["Category"],
            "Similarity_Score": sims_scaled
        }).sort_values("Similarity_Score", ascending=False).head(5)

        st.subheader("🎯 CVs les plus similaires trouvés :")
        for i, row in df_rec.iterrows():
            st.markdown(f"**Catégorie :** {row['Category']}")
            st.markdown(f"**Score de similarité :** `{row['Similarity_Score']:.2f}`")
            st.markdown(f"{row['Resume'][:500]}...")
            st.markdown("---")

        # Domaine prédictif du CV uploadé (via KMeans)
        try:
            kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
            kmeans.fit(embeddings)
            pred_label = kmeans.predict([query_emb])[0]
            st.success(f"🧩 Domaine probable du CV analysé : **Domaine {pred_label}**")
        except Exception:
            st.warning("Impossible de prédire le domaine sur cet échantillon.")
