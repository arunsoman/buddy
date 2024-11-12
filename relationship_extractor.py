import spacy
import networkx as nx

def extract_relationships(text):
    # Load the Spacy model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Extract relationships
    relationships = []
    for token in doc:
        if token.dep_ == "nsubj" or token.dep_ == "dobj":
            relationships.append((token.text, token.head.text, token.dep_))
    return relationships
