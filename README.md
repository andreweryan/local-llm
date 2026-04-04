Local LLM API with RAG (Retrieval-Augmented Generation)

- llama3
    - parameters: 8.0B
    - context length: 8192
    - embedding length: 4096
    - quantization: Q4_0
- nomic-embed-text
    - parameters: 137M
    - context length: 2048
    - embedding length: 768
    - quantization: F16
- Faiss (Facebook AI Similarity Search)
    - library for efficient similarity search and clustering of dense vectors

To get setup initially, run `source setup.sh`. This will automatically install ollama, pull the llama3 and nomic-embed-text models, create a virtual environment, install python dependencies and create two directories, one to be used for adding your RAG context documents and another for storing files related to Faiss. 

Before starting the server, add documents to the `docs` folder. 

After initial setup, run `python main.py` to start the server. Upon starting the server, any documents in the docs folder will be embedded and indexed with Faiss. Your local llama3 instance will use __*ONLY*__ the information contained in your documents for context. Supported document types include .txt, .md, and .pdf. 