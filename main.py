

import os
import sys
from srs_preprocessor import preprocess_srs_text
from requirement_identifier import identify_potential_requirements
from requirement_classifier import classify_requirements
from ambiguity_resolver import resolve_ambiguities, determine_ambiguity
from db_connector import connect_to_db, create_requirements_table, insert_requirement

def main():
    # Set the file path to the SRS document
    srs_file_path = "path_to_your_srs_document.pdf"

    # Preprocess the SRS text
    preprocessed_srs_text = preprocess_srs_text(srs_file_path)

    # Identify potential requirements
    potential_requirements = identify_potential_requirements(preprocessed_srs_text)

    # Classify requirements
    classified_requirements = classify_requirements(potential_requirements)

    # Resolve ambiguities
    resolved_requirements = resolve_ambiguities(classified_requirements)

    # Determine ambiguous requirements
    ambiguous_requirements = determine_ambiguity(resolved_requirements)

    # Connect to the database
    db_name = "requirements.db"
    conn = connect_to_db(db_name)

    # Create the requirements table
    create_requirements_table(conn)

    # Insert requirements into the database
    for requirement, classification, similarity in resolved_requirements:
        insert_requirement(conn, (requirement, classification, similarity))

    # Close the database connection
    conn.close()

    # Print the ambiguous requirements
    print("Ambiguous Requirements:")
    for requirement, classification, similarity in ambiguous_requirements:
        print(f"{requirement} - {classification} - Similarity: {similarity}")

if __name__ == "__main__":
    main()