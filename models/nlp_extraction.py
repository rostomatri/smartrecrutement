# models/nlp_extraction.py
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import spacy
import os

# Config paths
LAYOUT_MODEL_PATH = "models/layoutlmv3-resume"  # ton modèle LayoutLMv3 fine-tuné (si tu en as un)
SPACY_NER_PATH = "models/ner_model"             # modèle spaCy entraîné par ner_train.py

# Charger LayoutLMv3 processor/model si existants (sinon None)
processor = None
layout_model = None
if os.path.exists(LAYOUT_MODEL_PATH):
    try:
        processor = LayoutLMv3Processor.from_pretrained(LAYOUT_MODEL_PATH)
        layout_model = LayoutLMv3ForTokenClassification.from_pretrained(LAYOUT_MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Impossible de charger LayoutLMv3 depuis {LAYOUT_MODEL_PATH}: {e}")

# Charger spaCy NER (SKILL)
if os.path.exists(SPACY_NER_PATH):
    try:
        nlp = spacy.load(SPACY_NER_PATH)
        print("✅ spaCy NER chargé depuis", SPACY_NER_PATH)
    except Exception as e:
        print("⚠️ Erreur chargement spaCy NER:", e)
        nlp = None
else:
    print("⚠️ Aucun modèle spaCy NER trouvé à", SPACY_NER_PATH)
    nlp = None



def extract_skills_with_spacy(text):
    """
    Détection SKILL via spaCy (modèle entraîné).
    """
    if nlp is None:
        return []
    doc = nlp(text)
    return list({ent.text for ent in doc.ents if ent.label_ == "SKILL"})

"""
def extract_skills_by_list(text, skill_list):
   #"""
   # Matching simple skills list -> retourne compétences présentes.
    #"""
"""    if not skill_list:
        return []
    lower = text.lower()
    found = [s for s in skill_list if s.lower() in lower]
    return list(set(found))
"""
