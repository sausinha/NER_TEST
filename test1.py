import os
import pdfplumber
import pytesseract
from PIL import Image
import spacy

# Load the pre-trained spaCy NER model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    entities = []
    doc = nlp(text)
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

def extract_data_from_text(text):
    entities = extract_entities(text)
    data = {}
    for entity, label in entities:
        data[label] = entity
    return data

def extract_data_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return extract_data_from_text(text)

def extract_text_from_image(image_path):
    text = pytesseract.image_to_string(Image.open(image_path), lang='eng')
    return text

def extract_data_from_image_file(image_file_path):
    text = extract_text_from_image(image_file_path)
    return extract_data_from_text(text)

def extract_data_from_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return extract_data_from_pdf(file_path)
    elif file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        return extract_data_from_image_file(file_path)
    else:
        print("Unsupported file format.")
        return {}

# Example usage
file_path = "example_broadband_bill.pdf"  # Provide path to your broadband bill file
if os.path.exists(file_path):
    data_from_file = extract_data_from_file(file_path)
    print("Data extracted from file:", data_from_file)
else:
    print("File not found.")
