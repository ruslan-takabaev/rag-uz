import os
import faiss
import pickle # To save the Python list of documents
from sentence_transformers import SentenceTransformer
import numpy as np # FAISS works well with numpy arrays
import torch

print("Loading documents...")
knowledge_base = []
# Assuming your files are in a folder named 'knowledge_base'
folder_path = "knowledge_base/"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            knowledge_base.append(f.read())

print(f"Loaded {len(knowledge_base)} documents.")

# 2. Create embeddings
print("Loading embedding model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: '{device}'.")
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)  # multilingual embedder

print("Encoding documents... (This may take a while for 144,547 documents)")
embeddings = embedding_model.encode(knowledge_base, show_progress_bar=True, convert_to_numpy=True)

# 3. Build the optimized FAISS index
print("Building FAISS index...")
embedding_dimension = embeddings.shape[1]

# Rule of thumb for nlist: sqrt(N) where N is the number of vectors; sqrt(144547) is 380.19 -> set nlist = 381
nlist = 381
quantizer = faiss.IndexFlatL2(embedding_dimension)
index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist)  # faster than IndexFlatL2 for a large knowledge base

# 4. Train the index on the embeddings
print("Training index...")
index.train(embeddings)

# 5. Add the embeddings to the trained index
print("Adding embeddings to index...")
index.add(embeddings)

print(f"Index is trained and populated. Total vectors in index: {index.ntotal}")

# 6. Save the index and the knowledge base to disk
print("Saving index and knowledge base to disk...")
faiss.write_index(index, "index.idx")
with open("knowledge_base.pkl", "wb") as f:
    pickle.dump(knowledge_base, f)

print("Setup complete. You can now run the query notebook.")