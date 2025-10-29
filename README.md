# Smart Recruitment - CV Analysis System

This project implements an advanced CV analysis system using state-of-the-art models for text extraction, section detection, and skills identification.

## 1. Text Extraction

The system begins with text extraction from various CV formats (PDF, images) using OCR technology. This process is handled in the `ocr` directory:

- `ocr_layout1.py`: Implements layout analysis and text extraction
- `extract_text.py`: Handles the core OCR functionality

### Features:
- Support for multiple document formats (PDF, PNG, JPG)
- Layout-aware text extraction
- Preservation of structural information
- Output stored in JSON format in `outputs/json/` and `outputs/extracted_json/`

## 2. Section Detection with LayoutLMv3

The system uses LayoutLMv3, a powerful document understanding model, for detecting and classifying different sections within CVs.

### Components:
- **Fine-tuning**: `models/layoutlm_finetune1.py`
  - Adapts LayoutLMv3 for CV section detection
  - Uses custom training data
  - Handles document layout and text content simultaneously

- **Inference**: `models/layoutlm_inference.py`
  - Applies the fine-tuned model to new CVs
  - Identifies key sections (Education, Experience, Skills, etc.)
  - Outputs structured section information

### Model Configuration:
- Pre-trained model: LayoutLMv3
- Model files located in `models/layoutlmv3-resume/`
- Includes configuration files for tokenizer and model parameters

## 3. NER - Skills Detection with spaCy

The final component uses spaCy for Named Entity Recognition (NER) to identify and extract skills from CV content.

### Features:
- Custom NER model trained for skills recognition
- Located in `models/ner_model/`
- Training script: `models/ner_train.py`
- Skills extraction implementation: `models/nlp_extraction.py`

### Capabilities:
- Identifies technical skills
- Extracts soft skills
- Recognizes technologies and tools
- Handles multiple languages

### Model Components:
- Trained NER model with custom entities
- Vocabulary and vectors for accurate skill recognition
- Configuration files for model behavior

## Project Structure

```
├── models/
│   ├── layoutlm_finetune1.py      # LayoutLMv3 fine-tuning
│   ├── layoutlm_inference.py      # Section detection inference
│   ├── ner_train.py              # Skills NER training
│   └── nlp_extraction.py         # NLP processing
├── ocr/
│   ├── extract_text.py           # Text extraction
│   └── ocr_layout1.py            # Layout analysis
└── outputs/
    ├── extracted_json/           # Processed CV data
    └── json/                     # Raw extraction results
```

## Setup and Requirements

The project dependencies are listed in `requirements.txt`. 