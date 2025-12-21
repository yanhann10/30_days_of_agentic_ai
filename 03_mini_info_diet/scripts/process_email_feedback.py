#!/usr/bin/env python3
import os
import re
import json
import base64
import pickle
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(__file__))
TOKEN_FILE = os.path.join(ROOT, "gmail_token.pickle")
CREDENTIALS_FILE = os.path.join(ROOT, "gcp_cred.json")

PREFS_FILE = os.path.join(ROOT, "prefs.jsonl")
PENDING_REPLIES_FILE = os.path.join(ROOT, "pending_replies.jsonl")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

TARGET_SUBJECT = "Re: Mini info dessert"
QUOTE_MARKER = "gmail.com> wrote:"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]


def gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)

    return build("gmail", "v1", credentials=creds)


def header(headers, name, default=""):
    name = name.lower()
    for h in headers or []:
        if h.get("name", "").lower() == name:
            return h.get("value", default)
    return default


def decode_b64(data: str) -> str:
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def extract_body(message) -> str:
    payload = message.get("payload", {}) or {}

    def walk(node):
        out = []
        for p in node.get("parts") or []:
            out.append(p)
            out.extend(walk(p))
        return out

    nodes = [payload] + walk(payload)

    for mime in ("text/plain", "text/html"):
        for n in nodes:
            if n.get("mimeType") == mime:
                data = (n.get("body") or {}).get("data")
                if data:
                    return decode_b64(data)

    data = (payload.get("body") or {}).get("data")
    return decode_b64(data) if data else ""


def iter_message_ids(service, query: str):
    token = None
    while True:
        resp = service.users().messages().list(
            userId="me", q=query, pageToken=token, maxResults=100
        ).execute()

        for m in resp.get("messages") or []:
            yield m["id"]

        token = resp.get("nextPageToken")
        if not token:
            break


def mark_read(service, msg_id: str):
    service.users().messages().modify(
        userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()


def split_feedback_and_original(body: str):
    idx = body.find(QUOTE_MARKER)
    if idx == -1:
        return body.strip(), ""
    return body[:idx].strip(), body[idx + len(QUOTE_MARKER) :].strip()


def extract_titles(original: str):
    titles = []
    pat = re.compile(r"^(?:\[\s*)?([0-2])(?:\s*\])?\s*[\.\:\-\)]\s*(.+)$")
    for line in original.splitlines():
        m = pat.match(line.strip())
        if m:
            titles.append(m.group(2).strip())
        if len(titles) == 3:
            break
    return titles


def parse_with_llm(user_feedback: str) -> dict | None:
    prompt = f"""
You analyze short user feedback about recommended research papers.

Return ONLY valid JSON matching this exact schema:

{{
  "topics_to_avoid": [],
  "topics_to_increase": [],
  "prereq_concepts": []
}}

Definitions:
- topics_to_avoid: topics the user explicitly does NOT want more of
- topics_to_increase: topics the user explicitly wants more of
- prereq_concepts: topics the user asks for prerequisites/background/explanation for

Rules:
- Fill ONLY what is explicitly stated
- Leave others as empty lists
- Do NOT mention indices, emails, or replies

User feedback:
{user_feedback}
""".strip()

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com",
            "X-Title": "email-feedback-parser",
        },
        json={
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )

    if r.status_code != 200:
        print("LLM error:", r.status_code, r.text[:200])
        return None

    content = r.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1].strip()
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        parsed = json.loads(content)
    except Exception:
        print("LLM JSON parse error. Raw:", content[:200])
        return None

    for k in ("topics_to_avoid", "topics_to_increase", "prereq_concepts"):
        if k not in parsed or not isinstance(parsed[k], list):
            parsed[k] = []
    return parsed


def append_event(event: dict):
    with open(PREFS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def append_pending_run(update: dict, meta: dict):
    if not (update.get("topics_to_avoid") or update.get("topics_to_increase") or update.get("prereq_concepts")):
        return

    record = {
        "type": "prefs_update",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": meta,
        "update": {
            "topics_to_avoid": update.get("topics_to_avoid", []),
            "topics_to_increase": update.get("topics_to_increase", []),
            "prereq_concepts": update.get("prereq_concepts", []),
        },
        "status": "pending",
        "next_action_placeholder": "TODO: daily_picks.py consumes prefs.jsonl; optionally send prereq explanations",
    }

    with open(PENDING_REPLIES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _union_extend(dst: dict, src: dict):
    for k in ("topics_to_avoid", "topics_to_increase", "prereq_concepts"):
        if src.get(k):
            dst[k].extend(src[k])


def main():
    if not OPENROUTER_API_KEY:
        raise SystemExit("Missing OPENROUTER_API_KEY")
    if not os.path.exists(PREFS_FILE):
        raise SystemExit("prefs.jsonl not found. Restore artifact before running.")

    service = gmail_service()
    me = service.users().getProfile(userId="me").execute().get("emailAddress", "")

    query = f'newer_than:1d subject:"{TARGET_SUBJECT}"'
    print("Authorized Gmail:", me)
    print("Gmail query:", query)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)

    processed = 0
    run_update = {"topics_to_avoid": [], "topics_to_increase": [], "prereq_concepts": []}

    for msg_id in iter_message_ids(service, query):
        meta = service.users().messages().get(
            userId="me",
            id=msg_id,
            format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()

        headers = meta.get("payload", {}).get("headers", [])
        subject = header(headers, "Subject", "")
        sender = header(headers, "From", "")

        received = datetime.fromtimestamp(int(meta["internalDate"]) / 1000, tz=timezone.utc)
        if subject != TARGET_SUBJECT or received < cutoff:
            continue

        full = service.users().messages().get(userId="me", id=msg_id).execute()
        body = extract_body(full).strip()
        if not body:
            continue

        feedback, original = split_feedback_and_original(body)
        if not feedback:
            continue

        titles = extract_titles(original)
        parsed = parse_with_llm(feedback)
        if not parsed:
            continue

        event = {
            "ts_utc": now.isoformat(),
            "received_utc": received.isoformat(),
            "topics_to_avoid": parsed.get("topics_to_avoid", []),
            "topics_to_increase": parsed.get("topics_to_increase", []),
            "prereq_concepts": parsed.get("prereq_concepts", []),
        }

        append_event(event)
        _union_extend(run_update, parsed)
        mark_read(service, msg_id)

        processed += 1
        print("Processed email:", msg_id)

    run_meta = {
        "authorized_gmail": me,
        "target_subject": TARGET_SUBJECT,
        "window_hours": 24,
        "processed_emails": processed,
    }
    append_pending_run(run_update, run_meta)

    print("Done. Processed:", processed)


if __name__ == "__main__":
    main()
