## About project
This work is greatly inspired by [this repository](https://github.com/NadejdaSt/Retrieval_Augmented_Generation) by [Nadejda Stekolnikova](https://github.com/NadejdaSt).

## Environment setup
1 - Install dependencies.
```
$pip install torch numpy faiss-cpu transformers sentence_transformers huggingface_hub hf_xet accelerate, tqdm, langchain, unicodedata, llama-cpp-python gradio
```
2 - Prepare the knowledge base: create knowledge_base/ subdirectory inside your working directory and move your .txt files there.

3 - Build Index: this will take a long time for a large set. It took us almost 16 hours on 2xT4 GPUs (~114k documents, ~5 pages average length, chunked into ~1,800,000 chunks)
```
$python3 build_index.py
```
4 - Run app.py to start the Gradio server
```
$python3 app.py
```
this will generate a public URL.

5 - Open this URL in web browser to use the chatbot.

6 (optional) - For quick testing and debugging, use console version:
```
$python3 rag.py
```

## Main changes to original work

**1 - all-mpnet-base-v2 -> multilingual-e5-large**: changed the embedding model to a multilingual one. 

**2 - IndexFlatL2 -> IndexIVFFlat**: using an optimized pre-computed Index to minimize search and startup time for large knowledge base.

**3 - gpt2 -> gemma-3-27b-it**: used quantized int4 GEMMA-3-27b model to fit into 32GB VRAM.

## Original work

**1 - multi-GPU support**: added logic to load model on two GPUs.

**2 - Gradio interface**: deployed the app on Gradio.

## License 
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ruslan-takabaev/rag-uz/blob/main/LICENSE) file for details.

## Authors
Ruslan Takabaev: ruslantakabaev495@gmail.com, [GitHub](https://github.com/ruslan-takabaev) 

[Ilya, Firdavs, Tyson, Diana]
