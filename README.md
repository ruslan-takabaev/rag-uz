## About project
This work is greatly inspired by [this repository](https://github.com/NadejdaSt/Retrieval_Augmented_Generation) by [Nadejda Stekolnikova](https://github.com/NadejdaSt).

## Environment setup
1 - Install dependencies.
```
$pip install torch numpy faiss-cpu transformers sentence_transformers huggingface_hub hf_xet accelerate, tqdm, langchain, unicodedata, llama-cpp-python
```
2 - Prepare the knowledge base: create knowledge_base/ subdirectory inside your working directory and move your .txt files there.

3 - Build Index: this will take a long time for a large set. It took us almost 16 hours on 2xT4 GPUs (~114k documents, ~5 pages average length, chunked into ~1,800,000 chunks)
```
$python3 build_index.py
```
4 - Run rag.py to test RAG

## Main changes to original work
**1 -** adapted the code to run on multiple (2) GPUs.

**2 - all-mpnet-base-v2 -> multilingual-e5-large**: changed the embedding model to a multilingual one. 

**3 - IndexFlatL2 -> IndexIVFFlat**: using an optimized pre-computed Index to minimize search and startup time for large knowledge base.

**4 - gpt2 -> gemma-3-27b-it**: used quantized int4 model to fit into 32GB VRAM.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ruslan-takabaev/rag-uz/blob/main/LICENSE) file for details.

## Authors
Ruslan Takabaev: [GitHub](https://github.com/ruslan-takabaev)

Tyson []()

Diana []()

Ilya []()

Firdavs []()
