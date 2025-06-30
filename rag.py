import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import login
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import pickle, faiss, sys
import unicodedata

# Log in
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN environment variable not set. Trying the regular log in procedure.")
    login()

# Use GPU for embedding model
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: '{device}' for embedding model.")

# Load Pre-built Index and Knowledge Base
print("Loading pre-built FAISS index and knowledge base...")
index = faiss.read_index("index_chunked.idx")
with open("knowledge_base_chunked.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

# Load the embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

# Remove any sus characters
def sanitize_input(text: str) -> str:
    """
    Normalizes Unicode, removes non-printable characters, and ensures the
    input is a valid string before passing it to a tokenizer.
    """
    if not isinstance(text, str):
        return ""

    # 1. Normalize the string to a standard form
    normalized_text = unicodedata.normalize('NFKC', text)

    # 2. Remove any non-printable characters
    sanitized_text = "".join(c for c in normalized_text if c.isprintable())
    
    return sanitized_text.strip()

# Retrieving
def retrieve_relevant_documents(query, index, knowledge_base, embedding_model, top_k=8):
    """Retrieves the top_k most relevant documents."""
    sanitized_query = sanitize_input(query)
    
    if not sanitized_query:
        print("Warning: Query became empty after sanitization. Skipping retrieval.")
        return []
    
    index.nprobe = 10
    query_embedding = embedding_model.encode([sanitized_query], convert_to_tensor=True, device=device)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [knowledge_base[i] for i in indices[0]]

# Loading generator model (transformers -> llama_cpp)
print("Loading local GGUF generator model across two GPUs with llama-cpp-python...")
generator = Llama(
    model_path="models/gemma-3-27b-it-q4_0.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,

    tensor_split=None
)

# Generating
def generate_answer_with_context(query, retrieved_documents, generator):
    """Generates an answer using the llama-cpp-python Llama object."""
    context = "\n\n".join(retrieved_documents)

    # Initial prompt template
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always answer in the same language in which the question was asked. If you do not know the answer and it is not given in the context, simply say that you do not possess that information (also in the same language)."},
        {"role": "user", "content": f"### Context:\n{context}\n\n### Question:\n{query}"}
    ]

    # Get response
    response = generator.create_chat_completion(
        messages=messages,  # prompt template
        max_tokens=1024,  # response token limit (1024 = 4096 characters or about 768 words)
        temperature=0.25,  # recommended value is in [0.20, 0.30]; higher = more creativity
        top_p=0.75  # recommended value is in [0.50, 0.80]; higher = more diversity
    )

    # Extract answer from response
    answer = response['choices'][0]['message']['content']
    return answer.strip()


if __name__ == "__main__":
    while True:
        raw_query = input("Type in your prompt (Ru/Uz): ")

        if raw_query.lower() is in ('quit', 'exit', 'stop', 'finish', 'end'):
             print("Exiting.")
             break
        
        query = sanitize_input(raw_query)
        
        if not query.strip():
            print("Error: The query is empty or contains invalid characters. Please try again.\n")
            continue

        retrieved_docs = retrieve_relevant_documents(query, index, knowledge_base, embedding_model)

        print("\n Retrieved Document(s) ")
        for i, doc in enumerate(retrieved_docs):
            print(f"Doc {i+1}: {doc[:128].replace(chr(10), ' ')}...")

        final_answer = generate_answer_with_context(query, retrieved_docs, generator)

        print("\n Generated Answer ")
        print(final_answer)

