from transformers import BertTokenizer, BertModel
import torch

def generate_contextual_embeddings(text):
    # Load the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Generate contextual embeddings
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings
