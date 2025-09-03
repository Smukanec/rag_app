#!/usr/bin/env python3
# RAG + WebSearch chat (CPU Ollama + GPU OpenAI-compatible API)
# - CPU: Ollama stream /api/generate
# - GPU: OpenAI-compatible (auto /v1), httpx klient s volitelným verify=False (GPU_INSECURE=1)
# - RAG z knowledge/*.txt, WebSearch (DuckDuckGo), společná paměť memory.jsonl
# - Indikátor práce, Stop, Rychlý mód (★), limit výstupu
# - GPU preflight /models + detailní diagnostika při pádu

import os
import re
import json
import threading
from datetime import datetime
from pathlib import Path
from collections import Counter
from urllib.parse import quote_plus, parse_qs, urlparse, unquote

import requests
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk, messagebox

# ---------- Cesty / konfigurace ----------
OLLAMA_URL = "http://127.0.0.1:11434"
APP_DIR = Path.home() / "rag_app"
KNOW_DIR = APP_DIR / "knowledge"
MEM_PATH = APP_DIR / "memory.jsonl"
APP_DIR.mkdir(parents=True, exist_ok=True)
KNOW_DIR.mkdir(parents=True, exist_ok=True)
if not MEM_PATH.exists():
    MEM_PATH.touch()

# ---------- GPU (OpenAI-compatible) s auto /v1 + httpx ----------
def _normalize_base_url(u: str) -> str:
    u = (u or "").strip().rstrip("/")
    if not u:
        return "http://localhost:8000/v1"
    return (u + "/v1") if not u.endswith("/v1") else u

GPU_BASE_URL_RAW = os.getenv("GPU_BASE_URL", "https://model.jarvik-ai.tech")
GPU_BASE_URL = _normalize_base_url(GPU_BASE_URL_RAW)
GPU_API_KEY = os.getenv("GPU_API_KEY", "mojelokalnikurvitko")
GPU_INSECURE = os.getenv("GPU_INSECURE", "0") == "1"     # 1 => neověřovat TLS (jen dočasně na test)
GPU_FORCE_H1 = os.getenv("GPU_FORCE_H1", "1") == "1"     # 1 => vynutit HTTP/1.1 (http2=False)

GPU_CLIENT = None
GPU_ERR = ""

def _init_gpu_client():
    """Inicializuje OpenAI klient s httpx, volitelně bez TLS verifikace a bez HTTP/2.
       Při selhání vrátí rozšířenou diagnostiku přes raw GET na /v1/models (requests).
    """
    global GPU_CLIENT, GPU_ERR
    try:
        import httpx
        from openai import OpenAI  # openai>=1.0
        http = httpx.Client(
            verify=not GPU_INSECURE,
            timeout=240,
            trust_env=True,          # respektuje HTTP(S)_PROXY/NO_PROXY, system CA, atd.
            http2=not GPU_FORCE_H1,  # defaultně vynucujeme HTTP/1.1
        )
        GPU_CLIENT = OpenAI(base_url=GPU_BASE_URL, api_key=GPU_API_KEY, http_client=http)
        GPU_ERR = ""
    except Exception as e:
        GPU_CLIENT = None
        # Diagnostika přes requests (čisté HTTP/1.1)
        try:
            r = requests.get(
                f"{GPU_BASE_URL}/models",
                headers={"Authorization": f"Bearer {GPU_API_KEY}"},
                timeout=20,
                verify=not GPU_INSECURE,
            )
            diag = f"HTTP {r.status_code} {r.reason}"
            try:
                j = r.json()
                diag += f" | body: {j}"
            except Exception:
                diag += f" | body: {r.text[:300]!r}"
            GPU_ERR = f"{type(e).__name__}: {e} | RAW /models -> {diag}"
        except Exception as e2:
            GPU_ERR = f"{type(e).__name__}: {e} | RAW /models FAIL -> {type(e2).__name__}: {e2}"

def gpu_preflight_models(max_show: int = 7) -> str:
    """Vrátí string s informací o dostupnosti GPU API (seznam modelů nebo chybu)."""
    if GPU_CLIENT is None:
        _init_gpu_client()
    if GPU_CLIENT is None:
        return f"[GPU init error] {GPU_ERR} | BASE={GPU_BASE_URL}"
    try:
        models = GPU_CLIENT.models.list()
        ids = [m.id for m in models.data][:max_show]
        return f"OK models: {', '.join(ids) if ids else '(žádné)'} | BASE={GPU_BASE_URL}"
    except Exception as e:
        return f"[GPU list error] {type(e).__name__}: {e} | BASE={GPU_BASE_URL}"

# ---------- Ollama ----------
def list_local_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        tags = r.json().get("models", [])
        return [m.get("name") for m in tags]
    except Exception:
        return []

def pick_model(question: str, available: list[str]) -> str:
    q = question.lower()
    is_code = any(w in q for w in ["kód", "code", "python", "c++", "c#", "java", "bash", "skript", "funkci", "error", "exception"])
    if is_code:
        for pref in ["chat-code", "codegemma2b-tuned", "codegemma:2b", "stable-code:3b"]:
            if pref in available:
                return pref
    for pref in ["chat-qwen", "qwen3:1.7b", "codegemma2b-tuned", "codegemma:2b", "smollm:1.7b"]:
        if pref in available:
            return pref
    return available[0] if available else "codegemma:2b"

def stream_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.35,
    num_ctx: int = 896,
    timeout: int = 240,
    num_predict: int = 256,
    on_chunk=None,
    is_cancelled=lambda: False,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "repeat_penalty": 1.1,
        },
    }
    with requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        buf = []
        for line in r.iter_lines(decode_unicode=True):
            if is_cancelled():
                break
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            if "response" in data and data["response"]:
                chunk = data["response"]
                buf.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
            if data.get("done"):
                break
        return "".join(buf).strip()

# ---------- GPU generate ----------
def generate_gpu(model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.4, timeout: int = 240) -> str:
    if GPU_CLIENT is None:
        _init_gpu_client()
    if GPU_CLIENT is None:
        raise RuntimeError(f"GPU klient není připravený: {GPU_ERR}")
    resp = GPU_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Odpovídej česky, stručně a přesně."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""

# ---------- RAG ----------
def chunk_text(text: str, max_tokens: int = 220) -> list[str]:
    words = text.split()
    out, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= max_tokens:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out

def load_knowledge_chunks():
    chunks = []
    for p in KNOW_DIR.glob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for ch in chunk_text(txt, max_tokens=220):
            chunks.append({"file": p.name, "text": ch})
    return chunks

def keywords(text: str) -> Counter:
    toks = re.findall(r"[A-Za-zÁ-ž0-9_]+", text.lower())
    stop = {
        "a", "i", "o", "u", "v", "s", "z", "do", "na", "se", "že", "to", "je", "jako",
        "the", "and", "of", "to", "in", "for", "po", "pro", "si", "aby", "nebo", "podle", "dle",
    }
    toks = [t for t in toks if len(t) > 2 and t not in stop]
    return Counter(toks)

def score_chunk(q_kw: Counter, chunk_text: str) -> int:
    c_kw = keywords(chunk_text)
    return sum(min(qv, c_kw[k]) for k, qv in q_kw.items() if k in c_kw)

def rag_retrieve(question: str, k: int = 1) -> list[dict]:
    chunks = load_knowledge_chunks()
    if not chunks:
        return []
    qkw = keywords(question)
    scored = [(score_chunk(qkw, ch["text"]), ch) for ch in chunks]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it[1] for it in scored[:k]]

# ---------- WebSearch ----------
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def duckduckgo_search(query: str, k: int = 1, timeout: int = 10) -> list[dict]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        items = []
        for res in soup.select(".result__body"):
            a = res.select_one(".result__a")
            snip = res.select_one(".result__snippet")
            if not a:
                continue
            title = a.get_text(" ", strip=True)
            href = a.get("href", "")
            if "uddg=" in href:
                try:
                    qs = parse_qs(urlparse(href).query)
                    if "uddg" in qs:
                        href = unquote(qs["uddg"][0])
                except Exception:
                    pass
            snippet = snip.get_text(" ", strip=True) if snip else ""
            items.append({"title": title, "url": href, "snippet": snippet})
            if len(items) >= k:
                break
        return items
    except Exception:
        return []

def build_web_context(results: list[dict]) -> str:
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[WS{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}".strip())
    return "\n---\n".join(lines)

# ---------- Prompt & Memory ----------
def build_prompt(system_prompt: str, question: str, rag_ctx: list[dict], web_ctx: str) -> str:
    parts = [system_prompt]
    if rag_ctx:
        ctx_txt = "\n---\n".join([f"[{c['file']}]\n{c['text']}" for c in rag_ctx])
        parts.append(f"Relevantní znalosti (TXT výňatky):\n{ctx_txt}")
    if web_ctx:
        parts.append(f"WebSearch (výsledky):\n{web_ctx}")
        parts.append("- Pokud jsou ve WebSearch konflikty, uveď nejistotu a nevyvozuj silné závěry.")
    parts.append("Instrukce:\n- Odpovídej česky, stručně a přesně.\n- Pokud něco není jasné, napiš co chybí.\n- Na konci uveď krátký seznam zdrojů [WS1..] a [soubor.txt], pokud byly použity.")
    parts.append(f"Uživatelův dotaz:\n{question}")
    return "\n\n".join(parts)

def append_memory(model: str, question: str, answer: str, meta: dict | None = None):
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": model,
        "question": question,
        "answer": answer,
        "meta": meta or {},
    }
    with open(MEM_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG + WebSearch Chat (Ollama + GPU)")
        self.geometry("900x640")
        self.minsize(840, 600)

        # defaulty
        self.REQUEST_TIMEOUT = 240
        self.DEFAULT_NUM_CTX = 896
        self.RAG_TOPK = 1
        self.WEB_TOPK = 1

        self._watchdog_after_id = None
        self._cancel = False
        self.chat_history: list[dict[str, str]] = []

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Backend:").pack(side="left")
        self.backend_var = tk.StringVar(value="auto")  # auto|cpu|gpu
        ttk.Radiobutton(top, text="Auto", variable=self.backend_var, value="auto").pack(side="left")
        ttk.Radiobutton(top, text="CPU (Ollama)", variable=self.backend_var, value="cpu").pack(side="left")
        ttk.Radiobutton(top, text="GPU (API)", variable=self.backend_var, value="gpu").pack(side="left", padx=(0, 12))

        ttk.Label(top, text="Model:").pack(side="left", padx=(8, 0))
        self.models_cpu = list_local_models()
        self.models_gpu = [
            "gpu:gpt-oss:latest",
            "gpu:starcoder:7b",
            "gpu:mistral:7b",
            "gpu:codellama:7b-instruct",
            "gpu:nous-hermes2:latest",
            "gpu:command-r:latest",
            "gpu:llama3:8b",
        ]
        all_models = self.models_cpu + self.models_gpu
        self.model_var = tk.StringVar(value="")
        self.model_cb = ttk.Combobox(top, textvariable=self.model_var, values=all_models, state="readonly", width=38)
        self.model_cb.pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Refresh", command=self.refresh_models).pack(side="left")

        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Auto-výběr modelu", variable=self.auto_var).pack(side="left", padx=(12, 0))

        self.rag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="RAG (knowledge/)", variable=self.rag_var).pack(side="left", padx=(12, 0))
        self.web_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="WebSearch", variable=self.web_var).pack(side="left", padx=(12, 0))
        self.quick_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Rychlý mód (★)", variable=self.quick_var).pack(side="left", padx=(12, 0))

        ttk.Label(top, text="Max odp.:").pack(side="left", padx=(12, 0))
        self.max_tokens = tk.IntVar(value=256)
        ttk.Spinbox(top, from_=64, to=2048, increment=64, textvariable=self.max_tokens, width=6).pack(side="left")

        frm_q = ttk.LabelFrame(self, text="Dotaz")
        frm_q.pack(fill="both", expand=False, padx=8, pady=(6, 6))
        self.q_txt = tk.Text(frm_q, height=5, wrap="word")
        self.q_txt.pack(fill="both", expand=True, padx=6, pady=6)
        self.q_txt.bind("<Return>", self.on_enter)
        self.q_txt.bind("<Shift-Return>", lambda e: None)

        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=8, pady=4)
        self.btn_send = ttk.Button(mid, text="Odeslat (Enter)", command=self.on_send)
        self.btn_send.pack(side="left")
        self.btn_stop = ttk.Button(mid, text="Zastavit", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))
        ttk.Button(mid, text="Otevřít knowledge/", command=self.open_knowledge_tip).pack(side="left", padx=8)
        ttk.Button(mid, text="Zobraz paměť", command=self.show_memory_tip).pack(side="left", padx=8)

        self.prog = ttk.Progressbar(mid, mode="indeterminate", length=200)
        self.prog.pack(side="right")
        self.timeout_label = ttk.Label(mid, text="", foreground="#b55")
        self.timeout_label.pack(side="right", padx=(0, 12))

        frm_a = ttk.LabelFrame(self, text="Odpověď")
        frm_a.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.a_txt = tk.Text(frm_a, height=18, wrap="word")
        self.a_txt.pack(fill="both", expand=True, padx=6, pady=6)
        self.a_txt.config(state="disabled")

        insecure_flag = "ON" if GPU_INSECURE else "OFF"
        h1_flag = "ON" if GPU_FORCE_H1 else "OFF"
        self.status = tk.StringVar(
            value=f"Knowledge: {KNOW_DIR} | Memory: {MEM_PATH} | GPU_BASE_URL={GPU_BASE_URL} (raw={GPU_BASE_URL_RAW}, INSECURE={insecure_flag}, FORCE_H1={h1_flag})"
        )
        ttk.Label(self, textvariable=self.status, foreground="#666").pack(fill="x", padx=8, pady=(0, 6))

        # úvodní zpráva + GPU preflight
        def _show_gpu_preflight():
            info = gpu_preflight_models()
            self.set_answer(f"[start] Připraveno.\nGPU preflight: {info}\n\nNapiš dotaz a stiskni Odeslat.")
        self.after(200, _show_gpu_preflight)

    # ---- helpers ----
    def refresh_models(self):
        self.models_cpu = list_local_models()
        all_models = self.models_cpu + self.models_gpu
        self.model_cb["values"] = all_models
        if not self.model_var.get() and all_models:
            self.model_var.set(all_models[0])

    def on_enter(self, event):
        if not (event.state & 0x0001):  # Enter bez Shiftu => odeslat
            self.on_send()
            return "break"

    def open_knowledge_tip(self):
        messagebox.showinfo(
            "Knowledge",
            f"Vlož své .txt soubory do:\n{KNOW_DIR}\n\nPři dotazu se vybere {self.RAG_TOPK} nejrelevantnější blok (keyword rank).",
        )

    def show_memory_tip(self):
        try:
            n = sum(1 for _ in open(MEM_PATH, "r", encoding="utf-8"))
        except Exception:
            n = 0
        messagebox.showinfo("Paměť", f"Log JSONL:\n{MEM_PATH}\nZáznamů: {n}\nFormát: {{ts, model, question, answer, meta}}")

    def set_answer(self, text: str):
        self.a_txt.config(state="normal")
        self.a_txt.delete("1.0", "end")
        self.a_txt.insert("1.0", text)
        self.a_txt.config(state="disabled")

    def append_answer(self, text: str):
        self.a_txt.config(state="normal")
        self.a_txt.insert("end", text)
        self.a_txt.see("end")
        self.a_txt.config(state="disabled")

    def on_stop(self):
        self._cancel = True

    def _start_progress(self):
        self._cancel = False
        self.timeout_label.config(text="")
        self.prog.start(12)
        self.btn_send.config(state="disabled", text="Pracuji…")
        self.btn_stop.config(state="normal")
        self.configure(cursor="watch")
        if self._watchdog_after_id:
            self.after_cancel(self._watchdog_after_id)
        self._watchdog_after_id = self.after(60000, lambda: self.timeout_label.config(text="…počkej, CPU/GPU maká"))

    def _stop_progress(self):
        self.prog.stop()
        self.btn_send.config(state="normal", text="Odeslat (Enter)")
        self.btn_stop.config(state="disabled")
        self.configure(cursor="")
        if self._watchdog_after_id:
            self.after_cancel(self._watchdog_after_id)
            self._watchdog_after_id = None
        self.timeout_label.config(text="")

    # ---- hlavní akce ----
    def on_send(self):
        question = self.q_txt.get("1.0", "end").strip()
        if not question:
            return
        self.q_txt.delete("1.0", "end")
        self.chat_history.append({"role": "user", "content": question})
        self.append_answer(f"\nTy: {question}\n")
        self._start_progress()

        def task():
            try:
                available_cpu = list_local_models()
                chosen_backend = self.backend_var.get()  # auto|cpu|gpu

                model = self.model_var.get()
                if self.auto_var.get() or not model:
                    model = pick_model(question, available_cpu)

                num_ctx = self.DEFAULT_NUM_CTX
                num_predict = max(64, min(self.max_tokens.get(), 512))
                if self.quick_var.get():
                    num_ctx = min(num_ctx, 768)
                    num_predict = min(num_predict, 256)

                rag_ctx = rag_retrieve(question, k=self.RAG_TOPK) if self.rag_var.get() else []
                web_ctx_str, web_meta = "", []
                if self.web_var.get():
                    ws = duckduckgo_search(question, k=self.WEB_TOPK, timeout=10)
                    web_ctx_str = build_web_context(ws)
                    web_meta = ws
                sys_prompt = (
                    "Jsi stručný a přesný český asistent. Pokud máš kontext ze znalostí/webu, prioritizuj fakta z něj. "
                    "Když něco nevíš, řekni to krátce."
                )
                history_txt = "".join(
                    [f"{'U' if m['role']=='user' else 'A'}: {m['content']}\n" for m in self.chat_history]
                )
                prompt = build_prompt(sys_prompt, history_txt.rstrip(), rag_ctx, web_ctx_str)

                # Rozhodnutí backendu
                use_gpu = False
                if chosen_backend == "gpu":
                    use_gpu = True
                elif chosen_backend == "cpu":
                    use_gpu = False
                else:
                    if model.startswith("gpu:"):
                        use_gpu = True
                    elif not available_cpu and GPU_CLIENT is not None:
                        use_gpu = True

                gpu_model = model.split("gpu:", 1)[1] if model.startswith("gpu:") else None
                used_label = f"GPU:{gpu_model}" if (use_gpu and gpu_model) else ("GPU:auto" if use_gpu else model)

                def _header():
                    self.append_answer(
                        f"AI [backend: {'GPU' if use_gpu else 'CPU'}] [model: {used_label}] (ctx={num_ctx}, max={num_predict})\n"
                    )
                self.after(0, _header)

                if use_gpu:
                    real_gpu_model = gpu_model or "mistral:7b"
                    answer = generate_gpu(
                        real_gpu_model,
                        prompt,
                        max_tokens=min(1024, max(128, num_predict * 2)),
                        temperature=0.4,
                        timeout=self.REQUEST_TIMEOUT,
                    )
                    self.after(0, lambda text=answer: self.append_answer(text + "\n"))
                else:
                    answer = stream_ollama(
                        model,
                        prompt,
                        temperature=0.35,
                        num_ctx=num_ctx,
                        timeout=self.REQUEST_TIMEOUT,
                        num_predict=num_predict,
                        on_chunk=lambda ch: self.after(0, lambda text=ch: self.append_answer(text)),
                        is_cancelled=lambda: self._cancel,
                    )
                    self.after(0, lambda: self.append_answer("\n"))

                meta = {
                    "websearch": web_meta,
                    "rag_files": list({c["file"] for c in rag_ctx}),
                    "backend": "gpu" if use_gpu else "cpu",
                    "gpu_model": gpu_model,
                }
                append_memory(used_label, question, answer if answer else "", meta)
                self.chat_history.append({"role": "assistant", "content": answer if answer else ""})

                def add_sources():
                    src_lines = []
                    if web_meta:
                        src_lines.append("\n\nZdroje (WebSearch):")
                        for i, r in enumerate(web_meta, 1):
                            src_lines.append(f"WS{i}: {r['title']} — {r['url']}")
                    if rag_ctx:
                        src_lines.append("Zdroje (Knowledge): " + ", ".join(sorted({c['file'] for c in rag_ctx})))
                    if src_lines:
                        self.append_answer("\n" + "\n".join(src_lines))
                self.after(0, add_sources)

            except requests.exceptions.Timeout:
                self.after(0, lambda: self.append_answer("\n\n[⚠ timeout] Zkrať dotaz nebo zapni Rychlý mód (★)."))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                extra = f"\nGPU_BASE_URL={GPU_BASE_URL} (raw={GPU_BASE_URL_RAW}, INSECURE={'ON' if GPU_INSECURE else 'OFF'}, FORCE_H1={'ON' if GPU_FORCE_H1 else 'OFF'})"
                self.after(0, lambda err=err, extra=extra: self.append_answer(f"Chyba: {err}{extra}\n"))
            finally:
                self.after(0, self._stop_progress)

        threading.Thread(target=task, daemon=True).start()

if __name__ == "__main__":
    try:
        import requests as _r  # noqa: F401
        import bs4 as _b  # noqa: F401
        from openai import OpenAI as _OpenAI  # noqa: F401
        import httpx as _httpx  # noqa: F401
    except Exception:
        print("Chybí knihovny. Spusť ve venv a nainstaluj: pip install -r requirements.txt")
        raise SystemExit(1)

    print("Starting GUI... DISPLAY =", os.environ.get("DISPLAY"))
    app = App()
    app.mainloop()
    print("GUI closed cleanly")
