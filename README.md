Local LLM Chat with Tools

To get setup initially, run `source setup.sh`. This will automatically install ollama, pull the gemma4, llama3, and mxbai-embed-large models, create a virtual environment, and install python dependencies. Three directories will also be created: one to be used for adding your RAG context documents, one for logging prompt/response metadata, and one for server logs.

Before starting the server, add documents to the `docs` folder.

After initial setup, to start the server and terminal chat run `bash chat.sh <name of session, defaults to auto assigns uuid> <directory with RAG files, defaults to docs> <top k similar chunks, defaults to 10> <model name, defaults to gemma4:latest, can be set to llama3:latest or any ollama thinking model>`. The first time the server is started, any documents in the docs folder will be embedded and indexed with FAISS. Whenever changes are made to the docs folder, the docs will be reprocessed and reindexed. This can take a significant amount of time depending on the number and size of the documents. If you provide a model other than gemma4:latest or llama3:latest. See [Ollama Models](https://ollama.com/search) for other models than can be used. If you supply a model other than gemma4:latest or llama3:latest to the CLI, it will be auto downloaded.

Example: `bash chat.sh research docs 1000 gemma4:latest`

| Model               | Architecture | Parameters | Context Length | Embedding Length | Quantization | Requires | Capabilities                          |
|--------------------|-------------|------------|----------------|------------------|--------------|----------|----------------------------------------|
| gemma4             | gemma4      | 8.0B       | 131072         | 2560             | Q4_K_M       | 0.20.0   | completion, vision, audio, tools, thinking |
| llama3             | llama           | 8.0B       | 8192           | 4096             | Q4_0         | —        | completion                             |
| mxbai-embed-large  | bert        | 334M       | 512            | 1024             | F16          | —        | embedding                              |
