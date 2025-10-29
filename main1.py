import os
import json
import traceback
import streamlit as st
from PIL import Image
import torch
import pytesseract
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from ocr.extract_text import extract_text_from_image, extract_text_from_pdf
from models.nlp_extraction import extract_skills_with_spacy
from pytesseract import Output

# ==============================
# CONFIGURATION
# ==============================
st.set_page_config(page_title="🧠 Smart CV Parser", layout="wide")
st.title("🧠 Smart CV Parser")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPLER_PATH = r"C:\Users\user\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
MODEL_DIR = "./outputs/models/layoutlmv3_finetuned"

os.makedirs("data/Resume", exist_ok=True)
os.makedirs("outputs/json", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)

label_list = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-CONTACT", "I-CONTACT",
    "B-SUMMARY", "I-SUMMARY",
    "B-EDUCATION", "I-EDUCATION",
    "B-EXPERIENCE", "I-EXPERIENCE",
    "B-SKILLS", "I-SKILLS",
    "B-PROJECTS", "I-PROJECTS",
    "B-CERTIFICATIONS", "I-CERTIFICATIONS",
    "B-LANGUAGES", "I-LANGUAGES",
    "B-PUBLICATIONS", "I-PUBLICATIONS",
    "B-REFERENCES", "I-REFERENCES",
    "B-OTHER", "I-OTHER"
]
id2label = {i: l for i, l in enumerate(label_list)}

# ==============================
# UTILS
# ==============================
def extract_words_and_boxes(image: Image.Image):
    """Use OCR to extract words and their normalized coordinates."""
    width, height = image.size
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words, boxes = [], []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        box = [
            int(1000 * x / width),
            int(1000 * y / height),
            int(1000 * (x + w) / width),
            int(1000 * (y + h) / height),
        ]
        words.append(text)
        boxes.append(box)
    return words, boxes


def infer_sections(image_path):
    """Detect sections from an image using LayoutLMv3 and OCR."""
    image = Image.open(image_path).convert("RGB")
    words, boxes = extract_words_and_boxes(image)
    
    st.write(f"DEBUG: {len(words)} words detected by OCR")  # OCR check
    if not words:
        raise ValueError("No words detected in the image. Check PDF quality or OCR settings.")

    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        predicted_labels = [id2label[p] for p in predictions]

    st.write("DEBUG predicted_labels:", predicted_labels[:50])  # Check first predictions

    sections = {}
    for word, label in zip(words, predicted_labels):
        if label != "O":
            label_clean = label.replace("B-", "").replace("I-", "")
            sections.setdefault(label_clean, []).append(word)

    # Merge words into full sections
    sections = {k: " ".join(v) for k, v in sections.items()}
    st.write("DEBUG detected sections:", sections)  # Final section check
    return sections


def convert_pdf_to_images(pdf_path):
    """Convert a PDF into PNG images (one per page)."""
    pages = convert_from_path(pdf_path, dpi=200)
    paths = []
    for i, p in enumerate(pages):
        tmp = f"temp_page_{i+1}.png"
        p.save(tmp, "PNG")
        paths.append(tmp)
    return paths


# ==============================
# STREAMLIT INTERFACE
# ==============================
uploaded_file = st.file_uploader("📄 Upload a CV (PDF or image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    try:
        save_path = os.path.join("data/Resume", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File saved: {save_path}")

        ext = uploaded_file.name.split(".")[-1].lower()

        # OCR extraction
        if ext in ["png", "jpg", "jpeg"]:
            st.image(Image.open(save_path), use_container_width=True)
            text = extract_text_from_image(save_path)
            image_paths = [save_path]
        else:
            st.info("📄 PDF detected — converting to images...")
            image_paths = convert_pdf_to_images(save_path)
            text = extract_text_from_pdf(save_path, poppler_path=POPLER_PATH)

        st.subheader("📝 Extracted Text (OCR)")
        st.text_area("OCR Text", text[:3000], height=250)

        # ========== 1️⃣ SECTION DETECTION ==========
        st.subheader("📚 Detected Sections (LayoutLMv3)")
        all_sections = []
        with st.spinner("🔍 Analyzing document..."):
            for img_path in image_paths:
                try:
                    sections = infer_sections(img_path)
                    all_sections.append(sections)
                except Exception as e:
                    st.warning(f"⚠️ Error on {img_path}: {e}")
            st.success("✅ Document analysis completed")

        # Display results
        if all_sections:
            for i, page_sections in enumerate(all_sections, start=1):
                st.markdown(f"### 📄 Page {i}")
                if page_sections:
                    for name, content in page_sections.items():
                        with st.expander(f"📖 {name}"):
                            st.write(content[:1500])
                else:
                    st.info("No sections detected on this page.")
        else:
            st.info("No sections detected.")

        # Cleanup temporary files
        for img_path in image_paths:
            if os.path.exists(img_path) and img_path.startswith("temp_page_"):
                os.remove(img_path)

        # ========== 2️⃣ SKILL EXTRACTION ==========
        st.subheader("🧩 Detected Skills (spaCy NER)")
        try:
            skills_ner = extract_skills_with_spacy(text)
            if skills_ner:
                st.write(skills_ner)
            else:
                st.info("No skills detected.")
        except Exception:
            st.error("spaCy NER error")
            st.text(traceback.format_exc())

        # ========== 3️⃣ SAVE RESULTS ==========
        out = {
            "filename": uploaded_file.name,
            "sections": all_sections,
            "skills_ner": skills_ner if "skills_ner" in locals() else [],
        }
        out_path = os.path.join("outputs/json", uploaded_file.name + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        st.success(f"📁 Results saved to {out_path}")

    except Exception:
        st.error("Main error")
        st.text(traceback.format_exc())
