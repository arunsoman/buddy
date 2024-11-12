import pandas as pd
from fuzzywuzzy import fuzz

# Load KG requirements and SRS document
kg_requirements = pd.read_csv('kg_requirements.csv')
srs_document = open('srs_document.txt', 'r').read()

# Preprocess SRS document
def preprocess_srs_document(srs_document):
    # Remove headers, footers, and tables
    srs_text = srs_document.replace("\n", " ")
    srs_text = srs_text.replace("\t", " ")
    return srs_text

srs_text = preprocess_srs_document(srs_document)

# Initialize validation scores
validation_scores = []

# Iterate through KG requirements
for index, kg_requirement in kg_requirements.iterrows():
    # Fuzzy matching with SRS text
    similarity_score = fuzz.token_sort_ratio(kg_requirement['description'], srs_text)
    
    # Assign validation score based on similarity score
    if similarity_score >= 0.9:
        validation_score = 100
    elif similarity_score >= 0.7:
        validation_score = 80 + (similarity_score - 0.7) * 20
    elif similarity_score >= 0.5:
        validation_score = 60 + (similarity_score - 0.5) * 20
    elif similarity_score >= 0.3:
        validation_score = 40 + (similarity_score - 0.3) * 20
    else:
        validation_score = (similarity_score - 0) * 40
    
    # Apply weightage based on requirement category
    if kg_requirement['category'] == 'Functional':
        weightage = 0.3
    elif kg_requirement['category'] == 'Non-Functional':
        weightage = 0.25
    elif kg_requirement['category'] == 'Interface':
        weightage = 0.2
    else:
        weightage = 0.25
    
    weighted_validation_score = validation_score * weightage
    
    # Store validation scores
    validation_scores.append({
        'KG Requirement': kg_requirement['description'],
        'SRS Match': srs_text,
        'Similarity Score': similarity_score,
        'Validation Score (points)': validation_score,
        'Weightage (%)': weightage * 100,
        'Weighted Validation Score': weighted_validation_score
    })

# Calculate average validation score
average_validation_score = sum([score['Validation Score (points)'] for score in validation_scores]) / len(validation_scores)

# Calculate final validation score with weightage
final_validation_score = sum([score['Weighted Validation Score'] for score in validation_scores])

# Print validation scores
print(pd.DataFrame(validation_scores))

# Print average and final validation scores
print("Average Validation Score:", average_validation_score)
print("Final Validation Score (with weightage):", final_validation_score)
