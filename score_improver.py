import torch
from transformers import LLaMAForConditionalGeneration, LLaMATokenizer
import networkx as nx

# Load the LLM model and tokenizer
model = LLaMAForConditionalGeneration.from_pretrained("llama")
tokenizer = LLaMATokenizer.from_pretrained("llama")

# Load the Knowledge Graph (KG)
G = nx.read_gpickle("knowledge_graph.gpickle")

# Load the Software Requirements Specification (SRS)
with open("srs.txt", "r") as f:
    srs_text = f.read()

# Define a function to analyze the requirement using the LLM
def analyze_requirement(requirement, context, srs_text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        f"Requirement: {requirement}\nContext: {context}\nSRS Text: {srs_text}",
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # Generate an analysis using the LLM
    outputs = model.generate(**inputs)

    # Decode the analysis
    analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return analysis

# Define a function to interact with the user and gather missing information
def gather_missing_info(requirement, analysis, srs_text):
    print(f"Analysis: {analysis}")

    # Ask the user what is missing in the requirement
    missing_info = input("What is missing in this requirement? (or 'none' if nothing is missing): ")

    if missing_info.lower() != "none":
        # Update the KG with the missing information
        G.nodes[requirement]["missing_info"] = missing_info

        # Ask the user if they want to update the SRS with the missing information
        update_srs = input("Do you want to update the SRS with the missing information? (yes/no): ")

        if update_srs.lower() == "yes":
            # Update the SRS with the missing information
            with open("srs.txt", "a") as f:
                f.write(f"\n{missing_info}")

            print("SRS updated successfully!")

    else:
        print("No missing information provided.")

# Define a function to process the low-scoring requirements
def process_low_scoring_requirements(G, srs_text):
    # Iterate over the nodes in the KG
    for node in G.nodes():
        # Check if the node has a low score
        if G.nodes[node]["score"] < 0.5:
            # Analyze the requirement using the LLM
            analysis = analyze_requirement(node, G.nodes[node]["context"], srs_text)

            # Interact with the user and gather missing information
            gather_missing_info(node, analysis, srs_text)

# Process the low-scoring requirements
process_low_scoring_requirements(G, srs_text)
