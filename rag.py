from huggingface_hub import login
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch, os, pickle, faiss, sys

# --- Updated: Log in using an environment variable or token ---
# This prevents the script from hanging if a token isn't stored.
# Set this in terminal: export HF_TOKEN='your_token'
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN environment variable not set. Trying the regular log in procedure.")
    login()

# --- Use GPU if available ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: '{device}'.")

# --- Load Pre-built Index and Knowledge Base ---
print("Loading pre-built FAISS index and knowledge base...")
index = faiss.read_index("index.idx")
with open("knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

# Load the same models used for building the index
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)

# --- Retrieving ---
def retrieve_relevant_documents(query, index, knowledge_base, embedding_model, top_k=3):
    """Retrieves the top_k most relevant documents."""
    index.nprobe = 10
    query_embedding = embedding_model.encode([query], convert_to_tensor=True, device=device)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [knowledge_base[i] for i in indices[0]]

# --- Generator Pipeline ---
print("Loading generator model...")
generator = pipeline(
    "text-generation",
    model="google/gemma-1.1-7b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# --- Generation (Updated) ---
def generate_answer_with_context(query, retrieved_documents, generator):
    """Generates an answer by treating everything after the prompt as the response."""
    context = "\n\n".join(retrieved_documents)

    messages = [
        {"role": "user", "content": f"You are a helpful assistant. Use the following context to answer the question. Answer in the same language in which the question was asked. If the answer is not in the context, say that you do not know. \n\n### Context:\n{context}\n\n### Question:\n{query}"},
        {"role": "assistant", "content": "### Answer:\n"}
    ]
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = generator(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1, # Increase for more creative answers
        top_p=0.95
    )

    generated_text = outputs[0]['generated_text']
    answer = generated_text[len(prompt):].strip()
    return answer

# --- Update: Better Command-Line Argument Handling ---
if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ('-p', '-prompt'):
        print(f"Your prompt: {sys.argv[2]}")
        query = sys.argv[2]
    elif len(sys.argv) == 2 and sys.argv[1] in ('-p', '-prompt'):
        print(f"Please provide your prompt after -p flag (in '')")
        sys.exit(1)
    elif len(sys.argv) == 1:
        query = str(input("Type in your prompt (Uz/Ru): "))
    else:
        print(f"Unknown arguments: {sys.argv}")
        print(f"Run the script with no arguments or provide the prompt after [-p] or [-prompt] flag.")
        print(f"Example 1: python3 {sys.argv[0]} ")
        print(f"Example 2: python3 {sys.argv[0]} -p 'Your prompt here' ")
        sys.exit(1)

    if not query.strip():
        print("Error: The query cannot be empty.")
        sys.exit(1)

    # Retrieve
    retrieved_docs = retrieve_relevant_documents(query, index, knowledge_base, embedding_model)
    print("\n--- Retrieved Document(s) ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"Doc {i+1}: {doc[:128].replace(chr(10), ' ')}...")  # Print first 128 chars, replacing newlines

    # Generate
    final_answer = generate_answer_with_context(query, retrieved_docs, generator)
    print("\n--- Generated Answer ---")
    print(final_answer)
