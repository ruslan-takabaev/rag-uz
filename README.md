## About project
This work is greatly inspired by [this repository](https://github.com/NadejdaSt/Retrieval_Augmented_Generation) by [Nadejda Stekolnikova](https://github.com/NadejdaSt).

## Environment setup
1 - Install dependencies.
```
$pip install torch numpy faiss-cpu transformers sentence_transformers huggingface_hub hf_xet
```
2 - Prepare the knowledge base: create knowledge_base/ subdirectory inside your working directory and move your .txt files there.

3 - Build Index
```
$python3 build_index.py
```
4 - Open and rag.ipynb to test RAG

## Main changes to original work
**1 - all-mpnet-base-v2 -> multilingual-e5-large**: changed the embedding model to a multilingual one. 

**2 - IndexFlatL2 -> IndexIVFFlat**: using an optimized Index to minimize search and startup time for large knowledge base. 

**3 -** Using a pre-computed Index to reduce time.

**4 - gpt2 -> gemma-3-27b-it**: this is another change for introducing multilingualism. A newer but slightly heavier gemma-2-9b-it can be used instead.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ruslan-takabaev/rag-uz/blob/main/LICENSE) file for details.
## Authors
Ruslan Takabaev: [GitHub](https://github.com/ruslan-takabaev), 

Tyson []()

Diana []()

Ilya []()

Firdavs []()
## Reference
https://github.com/NadejdaSt/Retrieval_Augmented_Generation
