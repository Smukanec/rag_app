# RAG Application

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

## Installation

Requires Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

Set multiple API keys:

```bash
export API_KEYS="key1,key2"
```

## Running

By default the application listens on port `8000`. To use a different port when
running directly with Python, set the `PORT` environment variable before
launching:

```bash
export PORT=9000
python app.py
```

If you run the app with an ASGI server like Uvicorn or Gunicorn, you can set the
port with the same environment variable or by using their command‑line options:

```bash
# Uvicorn
export PORT=9000
uvicorn app:app --port $PORT   # or: uvicorn app:app --port 9000

# Gunicorn
export PORT=9000
gunicorn app:app -b 0.0.0.0:$PORT   # or: gunicorn app:app -b 0.0.0.0:9000
```

## Production

For server deployments, run the `install.sh` script. It performs the following steps:

1. Creates and enables a `rag_app.service` systemd unit so the application runs on startup.
2. Configures firewall rules (e.g., via `ufw`) to allow incoming traffic on the app's port.

After installation, environment variables can be customized in `/etc/systemd/system/rag_app.service`.
Edit the `[Service]` section to set values such as the listening port and API keys:

```ini
[Service]
Environment="PORT=8000"
Environment="API_KEYS=key1,key2"
```

Run `sudo systemctl daemon-reload && sudo systemctl restart rag_app` after making changes.

## Running Local Models via Docker

You can run OpenAI‑compatible servers locally via Docker. Acquire or build an image
for the model host you prefer:

```bash
# vLLM
docker pull vllm/vllm-openai:latest

# Text Generation Inference
docker pull ghcr.io/huggingface/text-generation-inference:latest

# Ollama (build from source)
git clone https://github.com/ollama/ollama.git
cd ollama && docker build -t ollama .
```

### GPU prerequisites

Running these images requires a GPU-enabled Docker setup with:

- NVIDIA drivers
- CUDA toolkit
- NVIDIA Container Toolkit (`nvidia-docker2`)

Follow the official installation guide for configuring GPU support in Docker:
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>

These containers expect an NVIDIA GPU. Lightweight models may run on GPUs with
roughly 8 GB of VRAM, while larger models can require 16–24 GB or more. Refer to
the upstream repositories for details:

- vLLM – <https://github.com/vllm-project/vllm>
- Text Generation Inference – <https://github.com/huggingface/text-generation-inference>
- Ollama – <https://github.com/ollama/ollama>

## Endpoints

Example request with Bearer token header:

```bash
curl -H "Authorization: Bearer key1" http://localhost:8000/v1/example
```

## Usage examples

### Health check

```bash
curl http://localhost:8000/healthz
```

### List available models

```bash
curl -H "Authorization: Bearer key1" http://localhost:8000/v1/models
```

### Create chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer key1" \
  -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
```

#### Streaming chat completion

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer key1" \
  -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
      }'
```

### Create embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer key1" \
  -d '{
        "model": "text-embedding-ada-002",
        "input": "hello world"
      }'
```

