

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
from srs_preprocessor import preprocess_srs_text

def identify_potential_requirements(text):
    # Preprocess the SRS text
    detected_requirements_heuristics, _ = preprocess_srs_text(text)

    # Identify potential requirements from the detected requirements
    potential_requirements = []
    for requirement in detected_requirements_heuristics:
        # Tokenize the requirement into words
        words = word_tokenize(requirement)

        # Check if the requirement contains keywords indicative of a requirement
        if any(word.lower() in ["shall", "should", "must"] for word in words):
            potential_requirements.append(requirement)

    return potential_requirements

# Load the preprocessed SRS sentences
def load_preprocessed_srs_sentences(file_path):
    """
    Loads the preprocessed SRS sentences from a file.

    Args:
        file_path (str): The path to the file containing the preprocessed SRS sentences.

    Returns:
        list: A list of preprocessed SRS sentences.
    """
    with open(file_path, 'r') as file:
        sentences = [line.strip() for line in file.readlines()]
    return sentences

# Define a function to identify potential requirements
def identify_potential_requirements(sentences):
    """
    Identifies potential requirements from the preprocessed SRS sentences.

    Args:
        sentences (list): A list of preprocessed SRS sentences.

    Returns:
        list: A list of potential requirements.
    """
    # Initialize an empty list to store potential requirements
    potential_requirements = []

    # Iterate over each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)

        # Check if the sentence contains keywords indicative of a requirement
        if any(word.lower() in ['shall', 'should', 'must', 'will', 'requirement'] for word in words):
            # If it does, add the sentence to the list of potential requirements
            potential_requirements.append(sentence)

    return potential_requirements

# Define a function to train a machine learning model to identify requirements
def train_requirement_identifier_model(sentences, labels):
    """
    Trains a machine learning model to identify requirements.

    Args:
        sentences (list): A list of preprocessed SRS sentences.
        labels (list): A list of labels corresponding to each sentence (0 for non-requirement, 1 for requirement).

    Returns:
        sklearn.ensemble.RandomForestClassifier: A trained RandomForestClassifier model.
    """
    # Split the data into training and testing sets
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the training data and transform both the training and testing data
    X_train = vectorizer.fit_transform(sentences_train)
    y_train = labels_train
    X_test = vectorizer.transform(sentences_test)
    y_test = labels_test

    # Train a RandomForestClassifier model on the training data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

# Example usage
if __name__ == "__main__":
    # Load preprocessed SRS sentences
    file_path = "path_to_your_preprocessed_srs_sentences.txt"
    sentences = load_preprocessed_srs_sentences(file_path)

    # Identify potential requirements
    potential_requirements = identify_potential_requirements(sentences)
    print("Potential Requirements:")
    for requirement in potential_requirements:
        print(requirement)

    # Train a machine learning model to identify requirements (assuming you have a labeled dataset)
    # labels = [0, 1, 1, 0, ...]  # Replace with your actual labels
    # model = train_requirement_identifier_model(sentences, labels)
