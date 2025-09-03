cd ~/rag_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# GPU API – zadej bez /v1, app si doplní
export GPU_BASE_URL="https://model.jarvik-ai.tech"
export GPU_API_KEY="mojelokalnikurvitko"

# dokud nevyřešíš cert/ALPN:
export GPU_INSECURE=1
export GPU_FORCE_H1=1
export NO_PROXY="model.jarvik-ai.tech"

./run.sh
