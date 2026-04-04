# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3
ollama pull llama3

# Pull text embedding model
ollama pull nomic-embed-text

# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate

# Install Project
pip install -e .

# Run
python main.py