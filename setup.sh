# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3
ollama pull gemma4:latest

# Pull text embedding models
ollama pull mxbai-embed-large

# Create venv
uv venv

# Activate venv
source .venv/bin/activate

# Install Project
uv pip install -e .

# Make RAG documents directory
mkdir docs
