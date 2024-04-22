from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from dataCleaning import data_cleaning
import torch

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)


def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    # Use the mean of the last layer's features as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def read_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        documents = file.readlines()
    return documents

file_path = 'data/training/test1.txt'

raw_documents = read_documents(file_path)
sentences = [data_cleaning(doc) for doc in raw_documents]

# Embed each sentence
sentence_embeddings = [get_sentence_embedding(sentence) for sentence in sentences]

# Calculate the document embedding
document_embedding = get_sentence_embedding(' '.join(sentences))

# Convert embeddings to numpy arrays for cosine similarity calculation
document_embedding_np = document_embedding.detach().numpy()
sentence_embeddings_np = [se.detach().numpy() for se in sentence_embeddings]

# Calculate similarity of each sentence to the document
similarities = [cosine_similarity(document_embedding_np, sentence_embedding_np.reshape(1, -1))[0][0] for sentence_embedding_np in sentence_embeddings_np]

# Rank sentences by their similarity
ranked_sentences = [sentence for _, sentence in sorted(zip(similarities, sentences), reverse=True)]

N = 5  # Number of sentences for the summary
summary = ' '.join(ranked_sentences[:N])
