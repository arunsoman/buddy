

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import PyPDF2
import re
# srs_preprocessor.py (updated)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import networkx as nx
import pandas as pd

# SRS Text Preprocessing
def preprocess_srs_text(srs_text):
    # Tokenize the SRS text
    tokens = word_tokenize(srs_text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token.lower() not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# Domain Knowledge Integration
def integrate_domain_knowledge(tokens, knowledge_base):
    # Identify relevant concepts and relationships
    relevant_concepts = []
    for token in tokens:
        if token in knowledge_base:
            relevant_concepts.append(token)
    
    return relevant_concepts

# Contextual Analysis
def analyze_context(relevant_concepts, knowledge_base):
    # Represent the SRS context as a graph
    context_graph = nx.DiGraph()
    
    # Add nodes for relevant concepts
    for concept in relevant_concepts:
        context_graph.add_node(concept)
    
    # Add edges for relationships between concepts
    for concept in relevant_concepts:
        for related_concept in knowledge_base[concept]:
            context_graph.add_edge(concept, related_concept)
    
    return context_graph

# Implicit Requirement Inference
def infer_implicit_requirements(context_graph, knowledge_base):
    # Analyze the context graph to identify potential implicit requirements
    implicit_requirements = []
    for node in context_graph.nodes():
        # Check if the node has any related concepts that are not in the SRS text
        for related_concept in knowledge_base[node]:
            if related_concept not in context_graph.nodes():
                implicit_requirements.append(related_concept)
    
    return implicit_requirements

# Example Usage
if __name__ == "__main__":
    # Load the SRS text
    srs_text = "Your SRS text here..."
    
    # Preprocess the SRS text
    tokens = preprocess_srs_text(srs_text)
    
    # Load the domain knowledge base
    knowledge_base = {
        "concept1": ["related_concept1", "related_concept2"],
        "concept2": ["related_concept3", "related_concept4"]
    }
    
    # Integrate domain knowledge
    relevant_concepts = integrate_domain_knowledge(tokens, knowledge_base)
    
    # Analyze the context
    context_graph = analyze_context(relevant_concepts, knowledge_base)
    
    # Infer implicit requirements
    implicit_requirements = infer_implicit_requirements(context_graph, knowledge_base)
    
    # Output implicit requirements
    print("Implicit Requirements:")
    for requirement in implicit_requirements:
        print(requirement)

def preprocess_srs_text(text):
    # Text Preprocessing: Concatenate multi-line requirements and remove line breaks
    text = text.replace("\n", " ").replace("\r", " ")

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Heuristics-based Requirement Boundary Detection
    detected_requirements = detect_requirements_heuristics(sentences)

    # ML-based Requirement Boundary Detection (Naive Bayes Classifier)
    ml_detected_requirements = detect_requirements_ml(sentences)

    return detected_requirements, ml_detected_requirements

def detect_requirements_heuristics(sentences):
    # Heuristics-based Approach: Keyword-based and Punctuation-based
    detected_requirements = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in ["requirement", "shall", "should"]):
            detected_requirements.append(sentence)
        elif sentence.endswith((". ", ".\n", ".\r", ".\r\n")):
            detected_requirements.append(sentence)

    return detected_requirements

def detect_requirements_ml(sentences):
    # ML-based Approach: Naive Bayes Classifier
    # Prepare the dataset
    dataset = [
        ("The system shall provide a user-friendly interface.", 1),
        ("It should be accessible on various devices.", 1),
        ("This is not a requirement.", 0),
        # Add more samples to the dataset...
    ]

    # Split the dataset into training and testing sets
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        [sentence for sentence, _ in dataset],
        [label for _, label in dataset],
        test_size=0.2,
        random_state=42
    )

    # Create a Naive Bayes Classifier pipeline
    classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the classifier
    classifier.fit(train_sentences, train_labels)

    # Detect requirements using the trained classifier
    detected_requirements = []
    for sentence in sentences:
        prediction = classifier.predict([sentence])
        if prediction[0] == 1:
            detected_requirements.append(sentence)

    return detected_requirements


# Load the SpaCy English model for tokenization and entity recognition
nlp = spacy.load("en_core_web_sm")

# Initialize the NLTK data needed for the script
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define a function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    num_pages = pdf_reader.numPages
    text = ''
    for page in range(num_pages):
        page_obj = pdf_reader.getPage(page)
        text += page_obj.extractText()
    pdf_file_obj.close()
    return text

# Define a function to clean and normalize the text
def clean_and_normalize_text(text):
    """
    Cleans and normalizes the text by removing special characters, 
    converting to lowercase, removing stopwords, and lemmatizing.

    Args:
        text (str): The text to be cleaned and normalized.

    Returns:
        str: The cleaned and normalized text.
    """
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a string
    text = ' '.join(words)

    return text

# Define a function to preprocess the SRS text
def preprocess_srs_text(file_path):
    """
    Preprocesses the SRS text by extracting the text from the PDF file, 
    cleaning and normalizing it, and then tokenizing it into sentences.

    Args:
        file_path (str): The path to the SRS PDF file.

    Returns:
        list: A list of preprocessed SRS sentences.
    """
    # Extract the text from the PDF file
    text = extract_text_from_pdf(file_path)

    # Clean and normalize the text
    text = clean_and_normalize_text(text)

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    return sentences

# Example usage
if __name__ == "__main__":
    file_path = "path_to_your_srs_pdf_file.pdf"
    preprocessed_sentences = preprocess_srs_text(file_path)
    for sentence in preprocessed_sentences:
        print(sentence)
# Example usage

    text = """
    The system shall provide a user-friendly interface.
    It should be accessible on various devices, including desktops, laptops, and mobile phones.
    The interface shall be responsive and adapt to different screen sizes.
    """
    detected_requirements_heuristics, detected_requirements_ml = preprocess_srs_text(text)

    print("Heuristics-based Detected Requirements:")
    for requirement in detected_requirements_heuristics:
        print(requirement)

    print("\nML-based Detected Requirements:")
    for requirement in detected_requirements_ml:
        print(requirement)
