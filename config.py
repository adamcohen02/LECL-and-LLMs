import os

# Ollama serves these models locally; no model API key is required.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama model tags. Pull each model once with `ollama pull <model-tag>`.
LLAMA_MODEL_ID = "llama3.2:3b"
GRANITE_MODEL_ID = "granite3.2:2b"
MISTRAL_MODEL_ID = "mistral:7b"
