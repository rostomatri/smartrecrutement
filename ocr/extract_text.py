# ocr/extract_text.py
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

# Configure paths (modifie selon ton installation)
POPLER_PATH = r"C:\Users\user\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Assigne si nécessaire
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_from_image(image_path, lang="fra+eng"):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang=lang)

def extract_text_from_pdf(pdf_path, poppler_path=POPLER_PATH, lang="fra+eng"):
    if not os.path.exists(poppler_path):
        raise FileNotFoundError(f"Poppler non trouvé : {poppler_path}")
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang=lang) + "\n"
    return text
