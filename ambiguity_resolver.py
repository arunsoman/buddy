

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

# Load the classified requirements
def load_classified_requirements(file_path):
    """
    Loads the classified requirements from a file.

    Args:
        file_path (str): The path to the file containing the classified requirements.

    Returns:
        list: A list of tuples containing the requirement and its classification (functional or non-functional).
    """
    with open(file_path, 'r') as file:
        requirements = [line.strip().split(' - ') for line in file.readlines()]
    return requirements

# Define a function to resolve ambiguities in the classified requirements
def resolve_ambiguities(requirements):
    """
    Resolves ambiguities in the classified requirements by measuring the cosine similarity between requirements.

    Args:
        requirements (list): A list of tuples containing the requirement and its classification (functional or non-functional).

    Returns:
        list: A list of tuples containing the requirement, its classification, and its similarity score.
    """
    # Initialize an empty list to store resolved requirements
    resolved_requirements = []

    # Iterate over each requirement
    for i, (requirement, classification) in enumerate(requirements):
        # Tokenize the requirement into words
        words = word_tokenize(requirement)

        # Remove stopwords from the tokenized words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]

        # Join the words back into a string
        requirement_text = ' '.join(words)

        # Initialize a variable to store the maximum similarity score
        max_similarity = 0

        # Iterate over each other requirement
        for j, (other_requirement, other_classification) in enumerate(requirements):
            # Skip if the requirements are the same
            if i == j:
                continue

            # Tokenize the other requirement into words
            other_words = word_tokenize(other_requirement)

            # Remove stopwords from the tokenized words
            other_words = [word for word in other_words if word.lower() not in stop_words]

            # Join the words back into a string
            other_requirement_text = ' '.join(other_words)

            # Create a TfidfVectorizer object
            vectorizer = TfidfVectorizer()

            # Fit the vectorizer to the requirement texts and transform them into vectors
            req_vector = vectorizer.fit_transform([requirement_text])
            other_req_vector = vectorizer.transform([other_requirement_text])

            # Calculate the cosine similarity between the vectors
            similarity = cosine_similarity(req_vector, other_req_vector)[0][0]

            # Update the maximum similarity score if the current similarity is higher
            if similarity > max_similarity:
                max_similarity = similarity

        # Append the requirement, its classification, and its similarity score to the resolved requirements list
        resolved_requirements.append((requirement, classification, max_similarity))

    return resolved_requirements

# Define a function to determine if a requirement is ambiguous based on its similarity score
def determine_ambiguity(resolved_requirements, threshold=0.5):
    """
    Determines if a requirement is ambiguous based on its similarity score.

    Args:
        resolved_requirements (list): A list of tuples containing the requirement, its classification, and its similarity score.
        threshold (float, optional): The minimum similarity score to consider a requirement ambiguous. Defaults to 0.5.

    Returns:
        list: A list of tuples containing the ambiguous requirement, its classification, and its similarity score.
    """
    # Initialize an empty list to store ambiguous requirements
    ambiguous_requirements = []

    # Iterate over each resolved requirement
    for requirement, classification, similarity in resolved_requirements:
        # Check if the similarity score is greater than or equal to the threshold
        if similarity >= threshold:
            # If it is, append the requirement to the ambiguous requirements list
            ambiguous_requirements.append((requirement, classification, similarity))

    return ambiguous_requirements

# Example usage
if __name__ == "__main__":
    # Load classified requirements
    file_path = "path_to_your_classified_requirements.txt"
    requirements = load_classified_requirements(file_path)

    # Resolve ambiguities in the classified requirements
    resolved_requirements = resolve_ambiguities(requirements)
    print("Resolved Requirements:")
    for requirement, classification, similarity in resolved_requirements:
        print(f"{requirement} - {classification} - Similarity: {similarity}")

    # Determine if a requirement is ambiguous based on its similarity score
    ambiguous_requirements = determine_ambiguity(resolved_requirements)
    print("\nAmbiguous Requirements:")
    for requirement, classification, similarity in ambiguous_requirements:
       print(f"{requirement} - {classification} - Similarity: {similarity}")
       
