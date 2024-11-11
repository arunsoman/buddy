

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import PyPDF2
import re

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
