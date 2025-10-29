# models/ner_train.py
import spacy
from spacy.training import Example
import pandas as pd
import random
import re
import os
from tqdm import tqdm

DATA_PATH = "data/Resume.csv"
SKILLS_PATH = "data/employment_skills.txt"
OUTPUT_MODEL = "models/ner_model"

# 1) Load the data
df = pd.read_csv(DATA_PATH)
# ⚠️ QUICK TEST: take only a small sample
df = df.sample(50, random_state=42)
texts = df["Resume_str"].dropna().tolist()

# 2) Load the skills
with open(SKILLS_PATH, "r", encoding="utf-8") as f:
    skills = [line.strip() for line in f if line.strip()]

# 🔹 Limit to 300 skills maximum
if len(skills) > 300:
    skills = random.sample(skills, 300)

print(f"✅ {len(skills)} skills loaded (limited to 300)")

# 3) Automatically generate training examples (SKILL)
TRAIN_DATA = []
for text in tqdm(texts[:50]):  # limit if dataset is too large
    entities = []
    for skill in skills:
        for match in re.finditer(rf"\b{re.escape(skill)}\b", text, flags=re.IGNORECASE):
            start, end = match.span()
            entities.append((start, end, "SKILL"))
    if entities:
        TRAIN_DATA.append((text, {"entities": entities}))

# 3b) Generate artificial sentences to improve diversity
for skill in skills:
    for template in [
        f"I am proficient in {skill}.",
        f"Skill: {skill}",
        f"Project completed using {skill}.",
        f"Experience with {skill}.",
        f"Training in {skill}.",
    ]:
        start = template.find(skill)
        TRAIN_DATA.append((template, {"entities": [(start, start + len(skill), "SKILL")]}))

print(f"📘 {len(TRAIN_DATA)} training examples generated")

# 4) Create or load spaCy English model
try:
    nlp = spacy.load("en_core_web_md")
    print("⚙️ Using en_core_web_md as base model")
except Exception:
    nlp = spacy.blank("en")
    print("⚙️ en_core_web_md not found — creating a blank 'en' model")

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

ner.add_label("SKILL")

# 5) Training
optimizer = nlp.resume_training() if os.path.exists(OUTPUT_MODEL) else nlp.begin_training()
n_iter = 20
for itn in range(n_iter):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, ann in TRAIN_DATA:
        try:
            example = Example.from_dict(nlp.make_doc(text), ann)
            nlp.update([example], sgd=optimizer, drop=0.2, losses=losses)
        except Exception:
            continue
    print(f"Epoch {itn+1}/{n_iter} — losses: {losses}")

# 6) Save model
os.makedirs(OUTPUT_MODEL, exist_ok=True)
nlp.to_disk(OUTPUT_MODEL)
print(f"✅ English NER model saved in {OUTPUT_MODEL}")
