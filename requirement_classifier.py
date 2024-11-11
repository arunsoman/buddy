

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re

# Load the identified potential requirements
def load_potential_requirements(file_path):
    """
    Loads the identified potential requirements from a file.

    Args:
        file_path (str): The path to the file containing the identified potential requirements.

    Returns:
        list: A list of identified potential requirements.
    """
    with open(file_path, 'r') as file:
        requirements = [line.strip() for line in file.readlines()]
    return requirements

# Define a function to classify requirements as functional or non-functional
def classify_requirements(requirements):
    """
    Classifies the identified potential requirements as functional or non-functional.

    Args:
        requirements (list): A list of identified potential requirements.

    Returns:
        list: A list of tuples containing the requirement and its classification (functional or non-functional).
    """
    # Initialize an empty list to store classified requirements
    classified_requirements = []

    # Iterate over each requirement
    for requirement in requirements:
        # Tokenize the requirement into words
        words = word_tokenize(requirement)

        # Check if the requirement contains keywords indicative of a functional requirement
        if any(word.lower() in ['shall', 'will', 'must', 'functional'] for word in words):
            # If it does, classify the requirement as functional
            classified_requirements.append((requirement, 'functional'))
        else:
            # Otherwise, classify the requirement as non-functional
            classified_requirements.append((requirement, 'non-functional'))

    return classified_requirements

# Define a function to train a machine learning model to classify requirements
def train_requirement_classifier_model(requirements, labels):
    """
    Trains a machine learning model to classify requirements.

    Args:
        requirements (list): A list of identified potential requirements.
        labels (list): A list of labels corresponding to each requirement (0 for non-functional, 1 for functional).

    Returns:
        sklearn.ensemble.RandomForestClassifier: A trained RandomForestClassifier model.
    """
    # Split the data into training and testing sets
    requirements_train, requirements_test, labels_train, labels_test = train_test_split(requirements, labels, test_size=0.2, random_state=42)

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the training data and transform both the training and testing data
    X_train = vectorizer.fit_transform(requirements_train)
    y_train = labels_train
    X_test = vectorizer.transform(requirements_test)
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
    # Load identified potential requirements
    file_path = "path_to_your_identified_potential_requirements.txt"
    requirements = load_potential_requirements(file_path)

    # Classify requirements
    classified_requirements = classify_requirements(requirements)
    print("Classified Requirements:")
    for requirement, classification in classified_requirements:
        print(f"{requirement} - {classification}")

    # Train a machine learning model to classify requirements (assuming you have a labeled dataset)
    # labels = [0, 1, 1, 0, ...]  # Replace with your actual labels
    # model = train_requirement_classifier_model(requirements, labels)
