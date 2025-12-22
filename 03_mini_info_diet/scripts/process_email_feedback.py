#!/usr/bin/env python3
import os
import re
import json
import base64
import pickle
import tempfile
import stat
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(__file__))
TOKEN_FILE = os.path.join(ROOT, "gmail_token.pickle")
# CREDENTIALS_FILE only used for local dev; GitHub Actions uses GCP_OAUTH_CREDENTIALS_JSON env var
CREDENTIALS_FILE = os.path.join(ROOT, "gcp_cred.json")
READING = os.path.join(ROOT, "reading_progress.md")

# Get credentials from env var (required for GitHub Actions, optional for local dev)
gcp_cred_json = os.environ.get("GCP_OAUTH_CREDENTIALS_JSON")


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
    
    # Support GitHub Actions: restore token from env var if provided and file doesn't exist
    if not os.path.exists(TOKEN_FILE):
        gmail_token_b64 = os.environ.get("GMAIL_TOKEN_PICKLE_B64")
        if gmail_token_b64:
            with open(TOKEN_FILE, "wb") as f:
                f.write(base64.b64decode(gmail_token_b64))
        else:
            gmail_token_json = os.environ.get("GMAIL_TOKEN_JSON")
            if gmail_token_json:
                # Convert JSON token to pickle format for compatibility
                from google.oauth2.credentials import Credentials
                creds_from_json = Credentials.from_authorized_user_info(json.loads(gmail_token_json), SCOPES)
                with open(TOKEN_FILE, "wb") as f:
                    pickle.dump(creds_from_json, f)
    
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Token exists but expired - refresh it (works in both local and CI)
            creds.refresh(Request())
        else:
            # No valid token - need to do OAuth flow (only works locally, not in CI)
            is_ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
            if is_ci:
                # In CI/GitHub Actions: run_local_server() cannot work (no browser)
                # Token must be provided via GMAIL_TOKEN_PICKLE_B64 or GMAIL_TOKEN_JSON secret
                raise SystemExit(
                    "No valid Gmail token found in CI. "
                    "Please run locally first to generate gmail_token.pickle, "
                    "then store it as GMAIL_TOKEN_PICKLE_B64 secret for CI use."
                )
            
            # Local development only: interactive OAuth flow with browser
            # This code path is NEVER executed in CI/GitHub Actions
            temp_cred_file = None
            try:
                if gcp_cred_json:
                    # Create secure temporary file for credentials
                    temp_fd, temp_cred_file = tempfile.mkstemp(suffix='.json', dir=os.path.dirname(CREDENTIALS_FILE) or None)
                    # Write credentials with restricted permissions (owner read/write only)
                    with os.fdopen(temp_fd, 'w') as f:
                        f.write(gcp_cred_json)
                    os.chmod(temp_cred_file, stat.S_IRUSR | stat.S_IWUSR)  # 0600 - owner read/write only
                    flow = InstalledAppFlow.from_client_secrets_file(temp_cred_file, SCOPES)
                elif os.path.exists(CREDENTIALS_FILE):
                    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                else:
                    raise SystemExit(
                        "Credentials not found. Set GCP_OAUTH_CREDENTIALS_JSON environment variable "
                        f"or create {CREDENTIALS_FILE} for local development."
                    )
                
                # run_local_server() opens a browser - only works locally, never in CI
                creds = flow.run_local_server(port=0)
            finally:
                # Securely delete temp file after OAuth flow completes (or fails)
                # This ensures credentials are never left on disk
                if temp_cred_file and os.path.exists(temp_cred_file):
                    try:
                        os.chmod(temp_cred_file, stat.S_IWUSR)  # Make writable for deletion
                        os.remove(temp_cred_file)
                    except Exception:
                        pass  # Best effort cleanup

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


def extract_paper_analysis_from_feedback(feedback: str, titles: list) -> dict | None:
    """Extract paper-specific analysis from user feedback using LLM"""
    if not titles or not feedback:
        return None
    
    prompt = f"""
You analyze user feedback about specific research papers and extract structured analysis.

The user provided feedback about these papers:
{chr(10).join(f"{i+1}. {title}" for i, title in enumerate(titles))}

User feedback:
{feedback}

Extract paper-specific insights and return ONLY valid JSON matching this schema:
{{
  "papers": [
    {{
      "title": "exact paper title from list above",
      "key_challenge": "main problem addressed (if mentioned)",
      "methods": "key techniques/approaches (if mentioned)",
      "evaluation": "evaluation approach/results (if mentioned)",
      "contribution": "main contribution (if mentioned)",
      "innovation": "novel aspects (if mentioned)",
      "limitation": "limitations or concerns (if mentioned)"
    }}
  ]
}}

Rules:
- Only include papers that have specific feedback
- Leave fields empty string if not mentioned
- Use exact paper titles from the list above
- Be concise (1-2 sentences per field)

Return ONLY the JSON, no other text.
""".strip()

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://example.com",
                "X-Title": "paper-analysis-extractor",
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=30,
        )

        if r.status_code != 200:
            print("LLM error extracting paper analysis:", r.status_code, r.text[:200])
            return None

        content = r.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()

        parsed = json.loads(content)
        return parsed
    except Exception as e:
        print(f"Error extracting paper analysis: {e}")
        return None


def update_reading_progress_with_analysis(reading_file: str, analysis_data: dict):
    """Update reading_progress.md with paper analysis from feedback"""
    if not analysis_data or "papers" not in analysis_data:
        return False
    
    try:
        with open(reading_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the latest "Today's top 3 picks:" section
        sections = content.rsplit("Today's top 3 picks:", 1)
        if len(sections) < 2:
            print("Could not find 'Today's top 3 picks' section")
            return False
        
        latest_section = sections[1]
        updated_section = latest_section
        
        for paper_analysis in analysis_data["papers"]:
            title = paper_analysis.get("title", "").strip()
            if not title:
                continue
            
            # Build analysis text
            analysis_lines = []
            if paper_analysis.get("key_challenge"):
                analysis_lines.append(f"• Key Challenge: {paper_analysis['key_challenge']}")
            if paper_analysis.get("methods"):
                analysis_lines.append(f"• Methods: {paper_analysis['methods']}")
            if paper_analysis.get("evaluation"):
                analysis_lines.append(f"• Evaluation: {paper_analysis['evaluation']}")
            if paper_analysis.get("contribution"):
                analysis_lines.append(f"• Contribution: {paper_analysis['contribution']}")
            if paper_analysis.get("innovation"):
                analysis_lines.append(f"• Innovation: {paper_analysis['innovation']}")
            if paper_analysis.get("limitation"):
                analysis_lines.append(f"• Limitation: {paper_analysis['limitation']}")
            
            if not analysis_lines:
                continue
            
            # Find the paper by title (fuzzy match)
            title_escaped = re.escape(title)
            # Match from paper number to end of paper section (before next paper or end)
            # Try exact match first
            pattern = rf'(\d+\.\s+{title_escaped}.*?)(\n\n|\n\d+\.\s+|$)'
            match = re.search(pattern, updated_section, re.DOTALL | re.IGNORECASE)
            
            if not match:
                # Try partial match (title might be slightly different)
                title_words = title.split()
                if len(title_words) > 3:
                    partial_title = ' '.join(title_words[:3])
                    partial_escaped = re.escape(partial_title)
                    pattern = rf'(\d+\.\s+.*?{partial_escaped}.*?)(\n\n|\n\d+\.\s+|$)'
                    match = re.search(pattern, updated_section, re.DOTALL | re.IGNORECASE)
            
            if match:
                paper_section = match.group(1)
                # Check if Paper Analysis already exists
                if 'Paper Analysis:' in paper_section or '• Key Challenge:' in paper_section:
                    print(f"Paper Analysis already exists for {title}, skipping")
                    continue
                
                # Add Paper Analysis section
                analysis_text = '\n   - Paper Analysis:\n'
                for line in analysis_lines:
                    analysis_text += f'     {line}\n'
                
                # Insert before the closing newlines
                replacement = paper_section.rstrip() + analysis_text + '\n'
                updated_section = updated_section.replace(match.group(0), replacement + match.group(2))
            else:
                print(f"Could not find paper '{title}' in reading_progress.md")
        
        # Reconstruct content
        updated_content = sections[0] + "Today's top 3 picks:" + updated_section
        
        with open(reading_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return True
    except Exception as e:
        print(f"Error updating reading_progress.md: {e}")
        return False


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

        try:
            from update_read_status import update_read_status
            update_read_status(body)
        except Exception as e:
            print(f"Error updating read status: {e}")

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
        
        # Extract and add paper-specific analysis if titles are found
        if titles:
            paper_analysis = extract_paper_analysis_from_feedback(feedback, titles)
            if paper_analysis:
                update_reading_progress_with_analysis(READING, paper_analysis)
                print(f"Added paper analysis for {len(paper_analysis.get('papers', []))} paper(s)")
        
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
