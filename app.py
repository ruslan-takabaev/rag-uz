import gradio as gr
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import unicodedata


# 1. LOAD MODELS AND DATA (only once at startup)
print("Loading all models and data... This may take a moment.")

# Use GPU for embedding model if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: '{device}' for embedding model.")

# Load embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

# Load FAISS index and knowledge base chunks
index = faiss.read_index("index_chunked.idx")
with open("knowledge_base_chunked.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

# Load the GGUF model for generation using llama-cpp-python
generator = Llama(
    model_path="models/gemma-3-27b-it-q4_0.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    tensor_split=None # Automatically split across available GPUs
)

print("All models loaded successfully!")


# 2. DEFINE CORE RAG LOGIC
def sanitize_input(text: str) -> str:
    """Cleans the input string."""
    if not isinstance(text, str): return ""
    normalized_text = unicodedata.normalize('NFKC', text)
    return "".join(c for c in normalized_text if c.isprintable()).strip()

def retrieve_relevant_documents(query, top_k=5):
    """Retrieves documents from the knowledge base."""
    safe_query = sanitize_input(query)
    if not safe_query: return []
    query_embedding = embedding_model.encode([safe_query], convert_to_tensor=True, device=device)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [knowledge_base[i] for i in indices[0]]


# 3. THE CHAT FUNCTION WITH STREAMING 
def chat_function(message, history):
    """
    This is the core function that Gradio will call.
    It takes a message and the chat history, and streams back the response.
    """
    # 1. Retrieve context
    retrieved_docs = retrieve_relevant_documents(message)
    context = "\n\n".join(retrieved_docs)

    # 2. Prepare the prompt for the model
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always answer in the same language in which the question was asked. If the answer is not given in the context, say that you do not know (also in the same language)."},
        {"role": "user", "content": f"### Context:\n{context}\n\n### Question:\n{message}"}
    ]

    # 3. Use the streaming capabilities of the generator
    # The `yield` keyword is what makes streaming work in Gradio
    response_stream = generator.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.35,
        stream=True  # <-- IMPORTANT: Enable streaming
    )

    # 4. Yield each token as it comes in
    full_response = ""
    for completion_chunk in response_stream:
        token = completion_chunk['choices'][0].get('delta', {}).get('content')
        if token is not None:
            full_response += token
            yield full_response


# 4. LAUNCH THE GRADIO UI
demo = gr.ChatInterface(
    fn=chat_function,
    title="Legal RAG Chatbot (Uz/Ru)",
    description="Ask questions about the legal documents. The model will use the knowledge base to answer.",
    theme="glass",
    examples=[
        ["Men daraxtni kessim nima bo'ladi?"],
        ["Какая ответственность за незаконную вырубку деревьев?"]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=42069, share=True)
