# merge_annotations.py
import os, json
from glob import glob

ANNOTATION_DIR = "data/annotations2"
MERGED_DIR = "data/annotations_merged"
os.makedirs(MERGED_DIR, exist_ok=True)

pdf_bases = set("_".join(f.split("_")[:-1]) for f in os.listdir(ANNOTATION_DIR) if f.endswith(".json"))

for base in pdf_bases:
    merged_words, merged_boxes, merged_labels = [], [], []
    files = sorted(glob(os.path.join(ANNOTATION_DIR, f"{base}_page*.json")))
    for f in files:
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            merged_words.extend(data["words"])
            merged_boxes.extend(data["boxes"])
            merged_labels.extend(data["labels"])
    out_path = os.path.join(MERGED_DIR, f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as jf:
        json.dump({
            "image_paths": files,
            "words": merged_words,
            "boxes": merged_boxes,
            "labels": merged_labels
        }, jf, ensure_ascii=False, indent=2)
    print(f"✅ Merged {base} -> {out_path}")
