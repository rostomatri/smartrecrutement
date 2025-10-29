import re
import torch
import numpy as np
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# ---------------------------
# Initialisation LayoutLMv3
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(device)

# ---------------------------
# Fonctions utilitaires
# ---------------------------
def extract_paragraphs(text, min_len=40):
    """Découpe le texte OCR en paragraphes ou lignes pertinentes"""
    blocks = [b.strip() for b in text.split("\n") if len(b.strip()) > min_len]
    # Retirer les lignes qui contiennent des emails, URLs, téléphones
    filtered = [b for b in blocks if not re.search(r'@|https?://|\d{2,}', b)]
    return filtered if filtered else [text]

def extract_embedding(text):
    return sbert_model.encode(text, normalize_embeddings=True)

def guess_title(section_texts):
    """Déduit un titre de section à partir du contenu"""
    # 1. Si une ligne est toute en majuscules → probable titre
    for t in section_texts:
        lines = t.split("\n")
        for line in lines:
            if line.strip().isupper() and 3 <= len(line.strip()) <= 40:
                return line.strip().title()

    # 2. Sinon, chercher les mots-clés fréquents
    joined = " ".join(section_texts).upper()
    common_titles = ["EDUCATION", "EXPERIENCE", "PROJECT", "SKILL", "CERTIFICATION", "SUMMARY", "INTEREST"]
    found = [t for t in common_titles if t in joined]
    if found:
        return found[0].capitalize()

    # 3. Sinon, top mot majuscule le plus fréquent
    candidates = re.findall(r'\b[A-Z][A-Za-z/&]{3,}\b', joined)
    if candidates:
        return Counter(candidates).most_common(1)[0][0].capitalize()

    return "Other"

def optimal_k(X, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_k = k
            best_score = score
    return best_k


def detect_sections(text):
    """Clusterise les paragraphes pour détecter dynamiquement les sections principales"""
    paras = extract_paragraphs(text)
    if len(paras) < 2:
        return [{"section_name": "General", "content": text}]

    # Extraire les embeddings pour chaque paragraphe
    embeddings = np.array([extract_embedding(p) for p in paras])

    # Trouver le nombre optimal de clusters
    best_k = optimal_k(embeddings)
    print(f"✅ Nombre de clusters optimal: {best_k}")

    # Clustering
    kmeans = KMeans(n_clusters=min(best_k, len(paras)), random_state=42).fit(embeddings)

    # Résultats
    results = []
    for cid in sorted(set(kmeans.labels_)):
        cluster_texts = [paras[i] for i, c in enumerate(kmeans.labels_) if c == cid]
        title = guess_title(cluster_texts)
        results.append({
            "section_name": title,
            "content": "\n".join(cluster_texts)
        })
    return results
