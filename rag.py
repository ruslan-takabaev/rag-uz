from huggingface_hub import login
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import os, pickle, faiss

# Log in to HF account
login()

# Load Pre-built Index and Knowledge Base
print("Loading pre-built FAISS index and knowledge base...")
index = faiss.read_index("index.idx")
with open("knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

# Load a pre-trained Sentence Transformer model (multilingual)
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cuda')

# Retrieving
def retrieve_relevant_documents(query, index, knowledge_base, embedding_model, top_k=3):
    """Retrieves the top_k most relevant documents."""
    index.nprobe = 10 # Search in 10 closest clusters
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [knowledge_base[i] for i in indices[0]]

# Generating
generator = pipeline(
    "text-generation",
    model="google/gemma-1.1-7b-it", # or model="google/gemma-2-9b-it" (newer and more powerful)
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def generate_answer_with_context(query, retrieved_documents, generator):
    """Generates an answer based on the query and retrieved documents."""
    context = "\n\n".join(retrieved_documents)
    prompt = f"Based on the following context, answer the question. \nContext: {context}\nQuestion: {query}"
    output = generator(prompt, max_new_tokens=200, num_return_sequences=1)
    answer = output[0]['generated_text'].split("Answer:")[1].strip()
    return answer

# Test
query = ""  # Question to answer
retrieved_docs = retrieve_relevant_documents(query, index, knowledge_base, embedding_model)
print("Retrieved Document(s):")
for i, doc in enumerate(retrieved_docs):
    print(f"{i}- {doc[:128]}...")

answer = generate_answer_with_context(query, retrieved_docs, generator)
print("\nGenerated Answer:")
print(answer)