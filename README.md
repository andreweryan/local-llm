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
- FAISS (Facebook AI Similarity Search)
    - library for efficient similarity search and clustering of dense vectors

To get setup initially, run `source setup.sh`. This will automatically install ollama, pull the llama3 and mxbai-embed-large models, create a virtual environment, install python dependencies and create two directories, one to be used for adding your RAG context documents and another for storing files related to FAISS.

Before starting the server, add documents to the `docs` folder.

After initial setup, run `bash chat.sh` to start the server and begin chatting with the model via the terminal. The first time the server is started, any documents in the docs folder will be embedded and indexed with FAISS. Whenever changes are made to the docs folder, the docs will be reprocessed and reindexed.
