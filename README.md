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

