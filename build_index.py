import os
import faiss
import pickle
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Worker for loading model on a GPU (also shows progress bars)
def encode_worker(process_id, device, model_name, data_chunk):
    """
    Worker function to load a model on a specific GPU and encode data with its own progress bar.
    """
    # Load the model for this specific process
    model = SentenceTransformer(model_name, device=device)
    
    # Manually batch the data to control the progress bar
    batch_size = 128
    encoded_batches = []
    
    # Create a tqdm progress bar for this worker, assigning it a unique line in the terminal
    with tqdm(total=len(data_chunk), desc=f"Worker {process_id} on {device}", position=process_id) as pbar:
        for i in range(0, len(data_chunk), batch_size):
            batch = data_chunk[i:i+batch_size]
            # Encode one batch
            batch_embeddings = model.encode(batch, show_progress_bar=False) # Internal bar must be off
            encoded_batches.append(batch_embeddings)
            # Update this worker's progress bar
            pbar.update(len(batch))
            
    # Combine all encoded batches into a single numpy array
    return np.vstack(encoded_batches)


def main():
    """Main function to manage the indexing process."""
    # 1. Load documents
    print("Loading documents...")
    all_doc_texts = []
    folder_path = "knowledge_base/"
    for filename in tqdm(os.listdir(folder_path), desc="Loading documents"):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().replace("toggle_night_mode", "").strip()
                all_doc_texts.append(content)
    print(f"Loaded {len(all_doc_texts)} full documents.")

    # 2. Chunk documents
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    knowledge_base_chunks = []
    for doc_text in tqdm(all_doc_texts, desc="Chunking documents"):
        chunks = text_splitter.split_text(doc_text)
        knowledge_base_chunks.extend(chunks)
    print(f"Split {len(all_doc_texts)} documents into {len(knowledge_base_chunks)} chunks.")

    # 3. Manually manage multi-GPU embedding creation
    model_name = 'intfloat/multilingual-e5-large'
    num_gpus = torch.cuda.device_count()
    chunk_splits = np.array_split(knowledge_base_chunks, num_gpus)
    worker_args = []
    for i in range(num_gpus):
        device = f'cuda:{i}'
        data_chunk = chunk_splits[i].tolist()
        worker_args.append((i, device, model_name, data_chunk))
        print(f"Prepared chunk of {len(data_chunk)} items for GPU {i}")

    print("Starting multiprocessing pool...")
    with mp.Pool(processes=num_gpus) as pool:
        print("Dispatching encoding tasks to workers...")
        results = pool.starmap(encode_worker, worker_args)

    print("\nAll workers finished. Combining results...")
    embeddings = np.vstack(results)
    
    # 4. Build and train FAISS index
    print("Building and training FAISS index...")
    embedding_dimension = embeddings.shape[1]
    nlist = int(len(knowledge_base_chunks)**0.5)
    print(f"Using nlist={nlist} for the IVFFlat index.")
    quantizer = faiss.IndexFlatL2(embedding_dimension)
    index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist)
    index.train(embeddings)
    index.add(embeddings)
    print(f"Index is trained and populated. Total vectors in index: {index.ntotal}")

    # 5. Save artifacts
    print("Saving index and knowledge base chunks to disk...")
    faiss.write_index(index, "index_chunked.idx")
    with open("knowledge_base_chunked.pkl", "wb") as f:
        pickle.dump(knowledge_base_chunks, f)
    print("Setup complete. Your RAG system is ready.")


if __name__ == '__main__':
    # CRITICAL: Set the start method to 'spawn' for CUDA safety.
    mp.set_start_method('spawn', force=True)

    gpu_count = torch.cuda.device_count()
    print(f"--- Sanity Check ---")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Detected {gpu_count} GPUs.")
    print(f"--------------------")
    if gpu_count == 0:
        raise RuntimeError("No GPUs detected by PyTorch. Please check your installation.")

    main()
