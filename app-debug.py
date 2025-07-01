import gradio as gr
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import unicodedata

# 1. LOAD MODELS AND DATA (no changes)
print("Loading all models and data... This may take a moment.")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: '{device}' for embedding model.")
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
index = faiss.read_index("index_chunked.idx")
with open("knowledge_base_chunked.pkl", "rb") as f:
    knowledge_base = pickle.load(f)
generator = Llama(
    model_path="models/gemma-3-27b-it-q4_0.gguf",
    n_gpu_layers=-1, n_ctx=4096, tensor_split=None
)
print("All models loaded successfully!")


# 2. DEFINE CORE RAG LOGIC (with the new rewrite function)
def sanitize_input(text: str) -> str:
    if not isinstance(text, str): return ""
    normalized_text = unicodedata.normalize('NFKC', text)
    return "".join(c for c in normalized_text if c.isprintable()).strip()

# --- NEW: Function to rewrite the query using chat history ---
def rewrite_query_with_history(message: str, history: list):
    if not history: # If there's no history, the query is already standalone
        return message

    # Build a simple history string
    history_str = ""
    for user_msg, assistant_msg in history[-3:]: # Use last 3 turns
        history_str += f"User: {user_msg}\nAssistant: {assistant_msg}\n"

    # Create a specific prompt for the rewriting task
    rewrite_prompt = f"""You are an expert at rephrasing a follow-up question to be a self-contained, standalone question. You will be given a conversation history and a follow-up question. Your task is to rewrite the follow-up question by incorporating the necessary context from the history.

    ### EXAMPLE 1 ###
    Chat History:
    User: Что будет за незаконную ловлю рыбы?
    Assistant: За незаконную ловлю рыбы предусмотрен штраф...
    Follow-up Question: Что насчёт охоты?

    Standalone Question: Какая ответственность за незаконную охоту?

    ### EXAMPLE 2 ###
    Chat History:
    User: Fransiya poytaxti nima?
    Assistant: Fransiyani poytaxti - Parij.
    Follow-up Question: Uning maydoni qancha?

    Standalone Question: Parijni maydoni qancha?

    ### ACTUAL TASK ###
    Chat History:
    {history_str}
    Follow-up Question: {message}

    Standalone Question:"""

    # Use the LLM to generate the rewritten query (non-streaming)
    response = generator(rewrite_prompt, max_tokens=150, temperature=0.1, stop=["\n", "###"])
    rewritten_query = response['choices'][0]['text'].strip()

    # As a fallback, if the model returns an empty string, use the original message
    return rewritten_query if rewritten_query else message


def retrieve_relevant_documents(query, top_k=5, n_probe_value=32):
    # ... (no changes to this function)
    safe_query = sanitize_input(query)
    if not safe_query: return []
    query_with_prefix = f"query: {safe_query}"
    index.nprobe = n_probe_value
    query_embedding = embedding_model.encode([query_with_prefix], convert_to_tensor=True, device=device)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [knowledge_base[i] for i in indices[0]]


# 3. CORE FUNCTION (Updated to include the rewrite step)
def respond_with_rag(message, history):
    # --- NEW STEP 1: Rewrite the query based on history ---
    rewritten_query = rewrite_query_with_history(message, history)
    print(f"[Debug] Original Query: '{message}'")
    print(f"[Debug] Rewritten Query for Retrieval: '{rewritten_query}'")

    # --- STEP 2: Retrieve context using the REWRITTEN query ---
    retrieved_docs = retrieve_relevant_documents(rewritten_query)
    context = "\n\n".join(retrieved_docs)

    # Format context for display (no change)
    formatted_context = "### Retrieved Context\n"
    if not retrieved_docs: formatted_context += "No documents were retrieved."
    else:
        for i, doc in enumerate(retrieved_docs): formatted_context += f"--- Document {i+1} ---\n{doc}\n\n"

    # --- STEP 3: Build the final prompt for the LLM ---
    # We still use the ORIGINAL history and message for the final prompt
    system_prompt = {"role": "system", "content": "You are a helpful assistant.  Always answer in the same language in which the question was asked. If the answer is not given in the context, say that you do not know (also in the same language)."}
    history_to_include = history[-2:]
    messages = [system_prompt]
    for user_msg, assistant_msg in history_to_include:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg: messages.append({"role": "assistant", "content": assistant_msg})

    # The final question still uses the original user message, but with the new context
    rag_prompt = f"### Context:\n{context}\n\n### Question:\n{message}"
    messages.append({"role": "user", "content": rag_prompt})

    # Get the streaming generator (no change)
    response_stream = generator.create_chat_completion(
        messages=messages, max_tokens=1024, temperature=0.35, stream=True
    )

    # Stream the response to the UI (no change)
    history.append([message, ""])
    for completion_chunk in response_stream:
        token = completion_chunk['choices'][0].get('delta', {}).get('content')
        if token is not None:
            history[-1][1] += token
            yield history, formatted_context

# 4. LAUNCH THE GRADIO UI using gr.Blocks (Corrected Structure)
with gr.Blocks(theme="glass") as demo:
    gr.Markdown("# Legal RAG Chatbot (Uz/Ru)\nAsk questions about the legal documents. The model will use the knowledge base to answer.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History")

            msg = gr.Textbox(label="Your Message", placeholder="Type your question here...", container=False, scale=7)
            submit_btn = gr.Button("Submit", variant="primary", scale=1)

        with gr.Column(scale=1):
            context_viewer = gr.Markdown(label="Retrieved Context")

    def clear_textbox():
        return ""

    submit_btn.click(
        fn=respond_with_rag,
        inputs=[msg, chatbot],
        outputs=[chatbot, context_viewer],
        queue=True
    ).then(clear_textbox, outputs=[msg])

    msg.submit(
        fn=respond_with_rag,
        inputs=[msg, chatbot],
        outputs=[chatbot, context_viewer],
        queue=True
    ).then(clear_textbox, outputs=[msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=42069, share=True)
