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
# Note: allenai/olmo models require credits on OpenRouter
MODELS_TO_TRY += ["meta-llama/llama-3.1-8b-instruct"]

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
        print("Error: OPENROUTER_API_KEY or OLMO_API_KEY not set")
        return None

    models = MODELS_TO_TRY[:max_retries] if max_retries else MODELS_TO_TRY
    messages = [{"role": "user", "content": prompt}]

    last_error = None
    for model in models:
        try:
            r = _post(messages, model)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 404:
                print(f"Model {model} not found (404), trying next model...")
                last_error = f"Model not found: {model}"
                continue
            elif r.status_code == 401:
                print(f"Authentication failed (401) for model {model}")
                last_error = "Authentication failed"
                return None
            elif r.status_code == 429:
                print(f"Rate limit hit (429) for model {model}, trying next model...")
                last_error = f"Rate limited: {model}"
                continue
            else:
                print(f"Unexpected status {r.status_code} for model {model}: {r.text[:200]}")
                last_error = f"HTTP {r.status_code}: {r.text[:100]}"
                continue
        except requests.exceptions.Timeout:
            print(f"Timeout calling model {model}, trying next model...")
            last_error = f"Timeout: {model}"
            continue
        except Exception as e:
            print(f"Error calling model {model}: {e}")
            last_error = str(e)
            continue

    if last_error:
        print(f"All models failed. Last error: {last_error}")
    return None

def call_llm_json(prompt, max_retries=None):
    resp = call_llm(prompt, max_retries)
    if not resp:
        return None

    original_resp = resp
    if "```" in resp:
        parts = resp.split("```")
        if len(parts) >= 3:
            resp = parts[1].strip()
            if resp.startswith("json"):
                resp = resp[4:].strip()

    try:
        return json.loads(resp)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response preview: {original_resp[:300]}...")
        return None
