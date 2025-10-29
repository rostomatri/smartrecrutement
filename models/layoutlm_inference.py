# models/layoutlm_inference.py
import os
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pdf2image import convert_from_path
import pytesseract
import json
import traceback

# ============== CONFIG ==============
MODEL_DIR = "./outputs/models/layoutlmv3_finetuned"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running inference on {DEVICE}")

# try load processor & model
try:
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
except Exception as e:
    print("[ERROR] Failed to load model/processor from", MODEL_DIR)
    print(e)
    raise

# ============== LABELS (must match fine-tuning) ==============
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
    "B-INTERESTS", "I-INTERESTS",
    "B-AWARDS", "I-AWARDS",
    "B-OTHER", "I-OTHER"
]
id2label = {i: l for i, l in enumerate(label_list)}

# ============== HELP: extract words & boxes via pytesseract ==============
def extract_words_and_boxes(image: Image.Image):
    """Return (words, boxes) where boxes are in LayoutLMv3 scale (0..1000 ints)."""
    width, height = image.size
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words, boxes = [], []
    for i in range(len(data["text"])):
        text = data["text"][i]
        if not text or str(text).strip() == "":
            continue
        # ensure safe numeric types
        try:
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
        except Exception:
            # fallback skip if bad coordinates
            continue
        # normalize to 0..1000
        box = [
            int(1000 * x / width),
            int(1000 * y / height),
            int(1000 * (x + w) / width),
            int(1000 * (y + h) / height),
        ]
        words.append(str(text))
        boxes.append(box)
    return words, boxes

# ============== convert pdf pages to images ==============
def convert_pdf_to_images(pdf_path):
    pages = convert_from_path(pdf_path, dpi=200)
    image_paths = []
    for i, page in enumerate(pages):
        img_path = f"temp_page_{i+1}.png"
        page.save(img_path, "PNG")
        image_paths.append(img_path)
    return image_paths

# ============== inference on a single image ==============
def infer_on_image(image_path):
    print(f"[INFO] infer_on_image -> {image_path}")
    image = Image.open(image_path).convert("RGB")

    # extract words & boxes
    words, boxes = extract_words_and_boxes(image)
    print(f"  → OCR found {len(words)} words")

    # Defensive: ensure lists are correct types
    if words:
        words = [str(w) for w in words]
    if boxes:
        boxes = [[int(b) for b in box] for box in boxes]

    encoding = None
    # Try to call processor with words+boxes; if it fails, fallback to image-only
    try:
        if words and boxes and len(words) == len(boxes):
            # Debug print: what we send (only small sample)
            print("  → Calling processor with words+boxes (sample):", words[:5], boxes[:5])
            encoding = processor(image, words=words, boxes=boxes, return_tensors="pt", truncation=True)
        else:
            print("  → Words/boxes empty or mismatched, using image-only processor fallback")
            encoding = processor(image, return_tensors="pt")
    except KeyError as ke:
        print("[WARN] Processor KeyError:", ke)
        print("Falling back to image-only call.")
        try:
            encoding = processor(image, return_tensors="pt")
        except Exception as e:
            print("[ERROR] processor fallback failed:", e)
            # save debug JSON
            dbg = {
                "image_path": image_path,
                "words_len": len(words) if words is not None else 0,
                "boxes_len": len(boxes) if boxes is not None else 0,
                "error": str(e),
            }
            dbg_path = image_path + ".debug.json"
            with open(dbg_path, "w", encoding="utf-8") as f:
                json.dump(dbg, f, indent=2)
            raise

    # move tensors to device
    for k, v in encoding.items():
        if torch.is_tensor(v):
            encoding[k] = v.to(DEVICE)

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits  # shape (batch, seq_len, num_labels)
        preds = logits.argmax(-1).squeeze().cpu().tolist()

    # tokens & map to labels
    input_ids = encoding["input_ids"].squeeze().cpu().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    # preds may be int or list; ensure iterable
    if isinstance(preds, int):
        preds = [preds]
    results = []
    for tok, pred_id in zip(tokens, preds):
        lbl = id2label.get(pred_id, "O")
        if lbl != "O":
            results.append({"token": tok, "label": lbl})
    return results

# ============== inference on a pdf ==============
def infer_on_pdf(pdf_path):
    print(f"[INFO] Running inference on PDF: {pdf_path}")
    image_paths = convert_pdf_to_images(pdf_path)

    all_results = {}
    for i, img_path in enumerate(image_paths):
        try:
            page_results = infer_on_image(img_path)
        except Exception:
            print("[ERROR] infer_on_image failed for", img_path)
            traceback.print_exc()
            page_results = []
        all_results[f"page_{i+1}"] = page_results
        # cleanup
        try:
            os.remove(img_path)
        except Exception:
            pass

    out_file = os.path.splitext(pdf_path)[0] + "_inference.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[✅] Results saved to {out_file}")
    return all_results

# ============== main test ==============
if __name__ == "__main__":
    TEST_PDF = "data/new_cv_pdfs/ENGINEERING/10030015.pdf"  # change to your file
    results = infer_on_pdf(TEST_PDF)
    for page, ents in results.items():
        print(f"\n--- {page} ---")
        for e in ents[:80]:
            print(e)
