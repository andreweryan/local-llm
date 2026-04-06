Local LLM API with Tools

- llama3
    - parameters: 8.0B
    - context length: 8192
    - embedding length: 4096
    - quantization: Q4_0
- mxbai-embed-large
    - architecture: bert
    - parameters: 334M
    - context length: 512
    - embedding length: 1024
    - quantization: F16
- Faiss (Facebook AI Similarity Search)
    - library for efficient similarity search and clustering of dense vectors

To get setup initially, run `source setup.sh`. This will automatically install ollama, pull the llama3 and mxbai-embed-large models, create a virtual environment, install python dependencies and create two directories, one to be used for adding your RAG context documents and another for storing files related to Faiss.

Before starting the server, add documents to the `docs` folder.

After initial setup, run `python main.py` to start the server. Upon starting the server, any documents in the docs folder will be embedded and indexed with Faiss.
