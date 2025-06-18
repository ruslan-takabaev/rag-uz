from huggingface_hub import login
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch, os, pickle, faiss, sys

# Log in using an environment variable HF_TOKEN
# Set this in terminal: export HF_TOKEN='token'
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN environment variable not set. Trying the regular log in procedure.")
    login()

# Use GPU if available 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: '{device}'.")

# Load Pre-built Index and Knowledge Base 
print("Loading pre-built FAISS index and knowledge base...")
index = faiss.read_index("index.idx")
with open("knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

# Load the embeddig model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

# Retrieving
def retrieve_relevant_documents(query, index, knowledge_base, embedding_model, top_k=10):
    """Retrieves the top_k most relevant documents."""
    index.nprobe = 10
    query_embedding = embedding_model.encode([query], convert_to_tensor=True, device=device)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [knowledge_base[i] for i in indices[0]]

# Generator
print("Loading generator model...")
generator = pipeline(
    "text-generation",
    model="google/gemma-3-27b-it",  # Need to quantize this down to 8bit or 4bit
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Generation
def generate_answer_with_context(query, retrieved_documents, generator):
    """Generates an answer by treating everything after the prompt as the response."""
    context = "\n\n".join(retrieved_documents)

    messages = [
        {"role": "user", "content": f"You are a helpful assistant. Use the following context to answer the question. Always answer in the same language in which the question was asked. If the answer is not given in the context, say that you do not know (also in the same language). \n### Context: {context}\n### Question: {query}\n"},
        {"role": "assistant", "content": "### Answer: "}
    ]
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = generator(
        prompt,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.35,
        top_p=0.95
    )

    generated_text = outputs[0]['generated_text']
    answer = generated_text[len(prompt):].strip()
    return answer


if __name__ == "__main__":
    while True:
        query = input("Type in your prompt (Ru/Uz): ")
        
        if not query.strip():
            print("Error: The query cannot be empty.")
            sys.exit(1)
           
        # Retrieve
        retrieved_docs = retrieve_relevant_documents(query, index, knowledge_base, embedding_model)
        print("\n--- Retrieved Document(s) ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"Doc {i+1}: {doc[:128].replace(chr(10), ' ')}...")

        # Generate
        final_answer = generate_answer_with_context(query, retrieved_docs, generator)
        print("\n--- Generated Answer ---")
        print(final_answer)
