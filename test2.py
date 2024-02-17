import os
import pdfplumber
from transformers import BertTokenizer, BertForTokenClassification
import torch

# Load the pre-trained BERT model and tokenizer for token classification (NER)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

def extract_entities(tokens, labels):
    entities = []
    current_entity = ''
    current_label = None
    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            current_entity += token[2:]
        else:
            if current_entity:
                entities.append((current_entity, current_label))
                current_entity = ''
            current_entity = token
            current_label = label
    if current_entity:
        entities.append((current_entity, current_label))
    return entities

def extract_data_from_text(text):
    # Tokenize input text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    # Predict labels using the pre-trained BERT model
    with torch.no_grad():
        outputs = model(input_ids)
    predictions = torch.argmax(outputs.logits, dim=2).squeeze(0).tolist()
    # Map label indices to labels
    labels = [model.config.id2label[label_id] for label_id in predictions]
    # Extract entities from tokens and labels
    entities = extract_entities(tokens, labels)
    # Filter entities by removing 'O' labels
    entities = [(entity, label) for entity, label in entities if label != 'O']
    data = {}
    # Group entities by label
    for entity, label in entities:
        if label not in data:
            data[label] = entity
        else:
            data[label] += ' ' + entity
    return data

def extract_data_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    # Preprocess extracted text (e.g., remove extra spaces, newlines)
    text = text.strip()
    return extract_data_from_text(text)

# Example usage
file_path = "example_broadband_bill.pdf"  # Provide path to your broadband bill file
if os.path.exists(file_path):
    data_from_file = extract_data_from_pdf(file_path)
    print("Data extracted from file:", data_from_file)
else:
    print("File not found.")
