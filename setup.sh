# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3
ollama pull llama3

# Pull text embedding models
ollama pull mxbai-embed-large

# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate

# Install Project
pip install -e .

# Make RAG documents directory
mkdir docs

# Make Faiss directory
mkdir faiss_index
