# ocr_layout1.py
import os
import json
import re
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import easyocr

# === CONFIGURATION ===
POPPLER_PATH = r"C:\Users\user\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
PDF_DIR = "data/new_cv_pdfs/ENGINEERING"
OUT_IMG_DIR = "data/images2"
OUT_JSON_DIR = "data/annotations2"
RULES_PATH = "data/keyword_rules.json"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_JSON_DIR, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=True)

# === ÉTAPE 1 — Conversion PDF -> Images ===
def pdf_to_images(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=200)
    image_paths = []
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, page in enumerate(pages):
        out_path = os.path.join(OUT_IMG_DIR, f"{base}_page{i+1}.png")
        page.save(out_path, "PNG")
        image_paths.append(out_path)
    return image_paths

# === ÉTAPE 2 — Extraction texte + boxes ===
def extract_words_and_boxes(image_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    results = reader.readtext(img)
    words, boxes = [], []
    for (bbox, text, prob) in results:
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        words.append(text)
        boxes.append([x0, y0, x1, y1])
    return words, boxes

# === ÉTAPE 3 — Chargement des règles de mots-clés ===
def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# === ÉTAPE 4 — Génération automatique des labels ===
def auto_label(words, rules):
    labels = []
    for word in words:
        label = "O"
        w_lower = word.lower()
        for section, keywords in rules.items():
            if any(re.search(rf"\b{k}\b", w_lower) for k in keywords):
                label = f"B-{section.upper()}"
                break
        labels.append(label)
    return labels

# === ÉTAPE 5 — Création d'une annotation complète ===
def create_annotation(image_path, rules):
    words, boxes = extract_words_and_boxes(image_path)
    labels = auto_label(words, rules)

    ann = {
        "image_path": image_path,
        "words": words,
        "boxes": boxes,
        "labels": labels
    }
    json_name = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    out_path = os.path.join(OUT_JSON_DIR, json_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    print(f"✅ Annotation créée : {out_path}")

# === ÉTAPE 6 — Pipeline complète ===
def process_all_pdfs():
    rules = load_rules()
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf)
        print(f"📄 Traitement de {pdf_path} ...")
        imgs = pdf_to_images(pdf_path)
        for img in imgs:
            create_annotation(img, rules)

if __name__ == "__main__":
    process_all_pdfs()
