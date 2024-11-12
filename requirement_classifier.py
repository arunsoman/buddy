

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
import neo4j



def classify_requirements(requirements):
    # ...
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    with driver.session() as session:
        for requirement in requirements:
            # Store classification as a property of the requirement node
            session.run("""
                MATCH (r:Requirement {id: $requirement_id})
                SET r.classification = $classification
            """, requirement_id=requirement_id, classification=classification)

    driver.close()

# Load the dataset for training the classifier
def load_dataset(file_path):
    """
    Loads the dataset for training the classifier.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        pd.DataFrame: The dataset as a Pandas DataFrame.
    """
    dataset = pd.read_csv(file_path)
    return dataset

# Preprocess the text data
def preprocess_text(text):
    """
    Preprocesses the text data by tokenizing, removing stopwords, and lemmatizing.

    Args:
        text (str): The text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatize the tokens
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a string
    text = ' '.join(tokens)
    return text

# Train a classifier to classify requirements
def train_classifier(dataset):
    """
    Trains a classifier to classify requirements.

    Args:
        dataset (pd.DataFrame): The dataset for training the classifier.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained classifier.
    """
    # Split the dataset into training and testing sets
    X = dataset['text']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the text data
    X_train = X_train.apply(preprocess_text)
    X_test = X_test.apply(preprocess_text)

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the training data and transform both the training and testing data
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train a RandomForestClassifier model on the training data
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the classifier on the testing data
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return classifier

# Classify a requirement using the trained classifier
def classify_requirement(classifier, requirement):
    """
    Classifies a requirement using the trained classifier.

    Args:
        classifier (sklearn.ensemble.RandomForestClassifier): The trained classifier.
        requirement (str): The requirement to be classified.

    Returns:
        str: The predicted label (functional or non-functional).
    """
    # Preprocess the requirement text
    requirement = preprocess_text(requirement)

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Transform the requirement text into a vector
    requirement_vector = vectorizer.transform([requirement])

    # Predict the label using the classifier
    predicted_label = classifier.predict(requirement_vector)[0]

    return predicted_label


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
# Example usage
if __name__ == "__main__":
    # Load the dataset
    file_path = "path_to_your_dataset.csv"
    dataset = load_dataset(file_path)

    # Train the classifier
    classifier = train_classifier(dataset)

    # Classify a requirement
    requirement = "This is a sample functional requirement."
    predicted_label = classify_requirement(classifier, requirement)
    print("Predicted Label:", predicted_label)
