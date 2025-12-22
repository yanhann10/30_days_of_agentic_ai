import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OLMO_API_KEY")
HTTP_REFERER = os.getenv("HTTP_REFERER", "https://example.com")
X_TITLE = os.getenv("X_TITLE", "llm_util")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PREFERRED_MODEL = os.getenv("LLM_MODEL")

MODELS_TO_TRY = []
if PREFERRED_MODEL:
    MODELS_TO_TRY.append(PREFERRED_MODEL)
MODELS_TO_TRY += ["allenai/olmo-3-7b-instruct", "meta-llama/llama-3.1-8b-instruct"]

def _post(messages, model, timeout=60):
    r = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": X_TITLE,
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
        },
        timeout=timeout,
    )
    return r

def call_llm(prompt, max_retries=None):
    if not OPENROUTER_API_KEY:
        return None

    models = MODELS_TO_TRY[:max_retries] if max_retries else MODELS_TO_TRY
    messages = [{"role": "user", "content": prompt}]

    for model in models:
        r = _post(messages, model)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 404:
            continue
        if r.status_code == 401:
            return None
        if r.status_code == 429:
            continue

    return None

def call_llm_json(prompt, max_retries=None):
    resp = call_llm(prompt, max_retries)
    if not resp:
        return None

    if "```" in resp:
        parts = resp.split("```")
        if len(parts) >= 3:
            resp = parts[1].strip()
            if resp.startswith("json"):
                resp = resp[4:].strip()

    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        return None
