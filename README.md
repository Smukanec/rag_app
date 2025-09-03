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

