# Enhanced Software Requirements Specification (SRS) Analysis and Enhancement System

## Objective Statement

### Primary Goal
Develop a comprehensive system that leverages Natural Language Processing (NLP), Knowledge Graph (KG), and Large Language Model (LLM) technologies to analyze Software Requirements Specification (SRS) documents, identify areas for improvement, and enhance the requirements for optimal software development outcomes.

### Specific Objectives

- **Accuracy**: Achieve an accuracy rate of 90% or higher in extracting and categorizing requirements from SRS documents.
- **Efficiency**: Reduce the time spent on manual requirement analysis and enhancement by at least 50% through automation.
- **Effectiveness**: Improve the quality of software development outcomes by enhancing the clarity, consistency, and completeness of SRS documents, resulting in a 20% reduction in software defects and a 15% increase in customer satisfaction.
- **Adaptability**: Ensure the system's adaptability to various SRS document formats, structures, and domains, with a minimum of 80% compatibility.
- **User Experience**: Provide an intuitive and user-friendly interface for stakeholders to interact with the system, resulting in a user satisfaction rating of 85% or higher.

## Key Performance Indicators (KPIs)

1. Requirement Extraction Accuracy
2. Time Reduction in Manual Analysis
3. Software Defect Reduction
4. Customer Satisfaction Increase
5. System Compatibility
6. User Satisfaction Rating

## Success Metrics

- Number of Successfully Analyzed SRS Documents
- Number of Enhanced Requirements
- User Engagement and Feedback
- System Uptime and Reliability
- Continuous Improvement and Update Cycles
- Comprehensive Report

## Project Overview

- **Project Name**: Software Requirements Specification (SRS) Analysis and Enhancement using Natural Language Processing (NLP) and Knowledge Graph (KG)
- **Project Objective**: Develop a system that analyzes SRS documents, identifies areas for improvement, and enhances the requirements using NLP and KG techniques

### Achievements

#### SRS Text Preprocessing

- **Tokenization**: Split SRS text into individual words/tokens
- **Stopword Removal**: Remove common words (e.g., "the," "and") that don't add much value
- **Stemming/Lemmatization**: Reduce words to their base form
- **Text Normalization**: Convert all text to lowercase
- **Python File**: `srs_preprocessor.py`
  - **Description**: Preprocesses the SRS text for further analysis

#### Requirement Extraction

- **Identified Requirements**: Extracted individual requirements from the preprocessed SRS text
- **Requirement Categorization**: Classified requirements into functional and non-functional categories
- **Python File**: `requirement_extractor.py`
  - **Description**: Extracts and categorizes requirements from the preprocessed SRS text

#### Knowledge Graph (KG) Construction

- **KG Creation**: Built a KG to store and represent the extracted requirements and their relationships
- **Entity Recognition**: Identified key entities (e.g., systems, users, components) in the requirements
- **Relationship Extraction**: Defined relationships between entities (e.g., "System A" interacts with "System B")
- **Python File**: `knowledge_graph_constructor.py`
  - **Description**: Constructs the KG from the extracted requirements and entities

#### NLP-based Requirement Analysis

- **Part-of-Speech (POS) Tagging**: Identified the grammatical categories of words in the requirements
- **Named Entity Recognition (NER)**: Identified named entities in the requirements
- **Dependency Parsing**: Analyzed the grammatical structure of the requirements
- **Python File**: `nlp_analyzer.py`
  - **Description**: Performs NLP-based analysis on the extracted requirements

#### Implicit Requirement Inference

- **Domain Knowledge Integration**: Utilized domain-specific knowledge to infer implicit requirements
- **Contextual Analysis**: Analyzed the context to identify potential implicit requirements
