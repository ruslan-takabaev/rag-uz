import os
import faiss
import pickle # To save the Python list of documents
from sentence_transformers import SentenceTransformer
import torch

print("[1/7] Loading documents...")
knowledge_base = []
folder_path = "knowledge_base/"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            knowledge_base.append(f.read())

print(f"Loaded {len(knowledge_base)} documents.")

# Create embeddings
print("[2/7] Loading embedding model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: '{device}'.")
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)  # new multilingual embedder

print("[3/7] Encoding documents... (This may take a while)")
embeddings = embedding_model.encode(knowledge_base, show_progress_bar=True, convert_to_numpy=True)

# Build the optimized FAISS index
print("[4/7] Building FAISS index...")
embedding_dimension = embeddings.shape[1]

# Rule of thumb for nlist: sqrt(N) where N is the number of vectors; sqrt(144547) is 380.19 -> set nlist = 381
nlist = 381
quantizer = faiss.IndexFlatL2(embedding_dimension)
index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist)  # faster than IndexFlatL2 for a large knowledge base

# Train the index on the embeddings
print("[5/7] Training index...")
index.train(embeddings)

# Add the embeddings to the trained index
print("[6/7] Adding embeddings to index...")
index.add(embeddings)

print(f"Index is trained and populated. Total vectors in index: {index.ntotal}")

# Save the index and the knowledge base to disk
print("[7/7] Saving index and knowledge base to disk...")
faiss.write_index(index, "index.idx")
with open("knowledge_base.pkl", "wb") as f:
    pickle.dump(knowledge_base, f)

print("Finished.")
