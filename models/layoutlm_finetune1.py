# layoutlm_finetune1.py
import os
import json
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# -------------------------------
# Dataset
# -------------------------------
class CVDataset(Dataset):
    def __init__(self, data_dir, processor, label2id, limit=None):
        self.samples = []
        self.processor = processor
        self.label2id = label2id
        files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        if limit:
            files = files[:limit]
        for f in files:
            path = os.path.join(data_dir, f)
            with open(path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def normalize_boxes(boxes, width, height):
        """Normalize bounding boxes to 0-1000 scale."""
        return [
            [
                int(1000 * (b[0] / width)),
                int(1000 * (b[1] / height)),
                int(1000 * (b[2] / width)),
                int(1000 * (b[3] / height)),
            ]
            for b in boxes
        ]
    def clean_annotations(data_dir):
     """
     Vérifie que chaque image_path référencée dans les fichiers JSON existe.
     Si l'image est manquante, supprime le JSON correspondant.
     """
     files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
     removed = 0
     for f in files:
        path = os.path.join(data_dir, f)
        try:
            with open(path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            image_path = data.get("image_path") or (data.get("image_paths")[0] if "image_paths" in data else None)
            if image_path is None or not os.path.exists(image_path):
                print(f"[INFO] Image manquante pour {f}. Suppression du JSON...")
                os.remove(path)
                removed += 1
        except Exception as e:
            print(f"[WARN] Erreur avec {f}: {e}")
        print(f"[INFO] Nettoyage terminé. {removed} fichiers JSON supprimés.")
    def __getitem__(self, idx):
        item = self.samples[idx]
        # Correction ici : utiliser "image_path" et non "image_paths"
        image = Image.open(item["image_path"]).convert("RGB")
        words = item["words"]
        boxes = self.normalize_boxes(item["boxes"], *image.size)
        labels = [self.label2id[l] for l in item["labels"]]
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Retourner chaque tensor sans batch dimension
        return {k: v.squeeze() for k, v in encoding.items()}


# -------------------------------
# Setup device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Processor LayoutLMv3
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# -------------------------------
# Labels
# -------------------------------
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
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}

# -------------------------------
# Dataset & Model
# -------------------------------

train_dataset = CVDataset("data/annotations2", processor, label2id, limit=None)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

# -------------------------------
# Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./outputs/models/layoutlmv3_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=9,
    save_steps=50,
    logging_steps=10,
    
    save_total_limit=2,
    fp16=True,
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# -------------------------------
# Run training
# -------------------------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./outputs/models/layoutlmv3_finetuned")
    processor.save_pretrained("./outputs/models/layoutlmv3_finetuned")
