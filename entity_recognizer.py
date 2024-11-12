import spacy
from spacy import displacy

def recognize_entities(text):
    # Load the Spacy model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
    