"""Build and email a fresh daily digest of three AI-agent research directions."""

import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from send_email import send_email


ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "agent_directions_history.json"
PACIFIC = ZoneInfo("America/Los_Angeles")
load_dotenv()


def should_run_now() -> bool:
    """Choose the DST-correct one of the two UTC schedules.

    We use the scheduled expression rather than wall-clock hour because GitHub
    Actions cron jobs can start late.
    """
    if os.getenv("GITHUB_EVENT_NAME") != "schedule":
        return True
    scheduled = os.getenv("SCHEDULE_EXPRESSION", "")
    utc_hour_for_8am = 8 - int(dt.datetime.now(PACIFIC).utcoffset().total_seconds() // 3600)
    return scheduled == f"0 {utc_hour_for_8am} * * *"


def load_history():
    if not HISTORY.exists():
        return []
    try:
        return json.loads(HISTORY.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def serper_search(query: str, source: str, limit: int = 8):
    key = os.getenv("SERPER_API_KEY")
    if not key:
        return []
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "num": limit, "tbs": "qdr:w"},
        timeout=30,
    )
    response.raise_for_status()
    items = []
    for result in response.json().get("organic", []):
        url = result.get("link", "")
        if not url:
            continue
        items.append({
            "source": source,
            "title": result.get("title", "").strip(),
            "url": url,
            "published": result.get("date", ""),
            "summary": result.get("snippet", "").strip(),
        })
    return items


def latest_arxiv(limit: int = 15):
    params = {
        "search_query": 'all:"AI agent" OR all:"language agent" OR all:"LLM agent"',
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(
        "https://export.arxiv.org/api/query",
        params=params,
        headers={"User-Agent": "daily-agent-directions/1.0 yanhann10@gmail.com"},
        timeout=45,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = []
    for entry in root.findall("atom:entry", ns):
        get = lambda tag: " ".join((entry.findtext(f"atom:{tag}", default="", namespaces=ns)).split())
        items.append({
            "source": "arXiv",
            "title": get("title"),
            "url": get("id"),
            "published": get("published")[:10],
            "summary": get("summary")[:900],
        })
    return items


def collect_sources():
    searches = [
        ("News", 'AI agents research reliability memory security evaluation after:2026-01-01'),
        ("Blogs", '(AI agents OR agentic AI) research engineering blog reliability memory'),
        ("Reddit", 'site:reddit.com (AI agents OR agentic AI) memory reliability evaluation'),
        ("X", 'site:x.com (AI agents OR agentic AI) research benchmark memory reliability'),
    ]
    items = latest_arxiv()
    for source, query in searches:
        try:
            items.extend(serper_search(query, source))
        except requests.RequestException as exc:
            print(f"Warning: {source} search failed: {exc}")

    unique = {}
    for item in items:
        key = re.sub(r"[?#].*$", "", item["url"]).rstrip("/")
        unique.setdefault(key, item)
    return list(unique.values())


def call_gemini_json(prompt: str):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.35, "responseMimeType": "application/json"},
        },
        timeout=120,
    )
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def rank_directions(items, history):
    recent_entries = history[-30:]
    previous_titles = [d["title"] for e in recent_entries for d in e.get("directions", [])]
    used_urls = {u for e in recent_entries for u in e.get("source_urls", [])}
    fresh = [item for item in items if item["url"] not in used_urls]
    evidence = fresh if len(fresh) >= 12 else items
    evidence = evidence[:45]

    prompt = f"""You are a rigorous AI research advisor. Today is {dt.datetime.now(PACIFIC).date()}.
Using ONLY the evidence records below, identify and rank the 3 strongest actionable research directions in the AI-agent space.

Optimize for: open research gap, importance, feasibility of a 3-6 month project, and support from multiple source types. Do not merely summarize products. Each direction must be materially different from the previous directions and from the other two today. Prefer fresh evidence, but never invent a claim or URL. Treat Reddit and X as practitioner signals, not verified facts. X coverage may be incomplete.

Previous direction titles to avoid repeating or lightly rewording:
{json.dumps(previous_titles[-60:], ensure_ascii=False)}

Evidence records:
{json.dumps(evidence, ensure_ascii=False)}

Return one JSON object with a `directions` array of exactly 3 objects, ranked best first. Each object must contain:
- title: specific, compact research direction
- thesis: 2-3 sentences explaining the gap and why now
- research_question: one falsifiable question
- experiment: a concrete first experiment
- metrics: array of 3-5 measurable metrics
- source_urls: array of 2-4 URLs copied exactly from evidence, ideally spanning at least 2 source types
- source_types: array corresponding to the URLs
"""
    result = call_gemini_json(prompt)
    directions = result.get("directions", [])
    if len(directions) != 3:
        raise RuntimeError(f"Expected exactly 3 directions, received {len(directions)}")
    allowed_urls = {item["url"] for item in evidence}
    source_by_url = {item["url"]: item["source"] for item in evidence}
    for direction in directions:
        direction["source_urls"] = [u for u in direction.get("source_urls", []) if u in allowed_urls][:4]
        if not direction["source_urls"]:
            raise RuntimeError(f"Direction has no valid evidence URL: {direction.get('title')}")
        direction["source_types"] = [source_by_url[u] for u in direction["source_urls"]]
    return directions


def build_email(directions):
    date = dt.datetime.now(PACIFIC).strftime("%B %-d, %Y")
    subject = f"AI Agent Research Radar — {date}"
    blocks = [f"<h2>Top 3 AI-agent research directions — {html.escape(date)}</h2>"]
    text = [f"Top 3 AI-agent research directions — {date}", ""]
    for rank, direction in enumerate(directions, 1):
        title = html.escape(direction["title"])
        blocks.append(f"<h3>{rank}. {title}</h3>")
        blocks.append(f"<p>{html.escape(direction['thesis'])}</p>")
        blocks.append(f"<p><strong>Research question:</strong> {html.escape(direction['research_question'])}</p>")
        blocks.append(f"<p><strong>First experiment:</strong> {html.escape(direction['experiment'])}</p>")
        blocks.append("<p><strong>Metrics:</strong> " + html.escape(", ".join(direction.get("metrics", []))) + "</p>")
        links = " · ".join(
            f'<a href="{html.escape(url, quote=True)}">{html.escape(kind)}</a>'
            for url, kind in zip(direction["source_urls"], direction.get("source_types", []))
        )
        blocks.append(f"<p><strong>Evidence:</strong> {links}</p>")

        text.extend([
            f"{rank}. {direction['title']}",
            direction["thesis"],
            f"Research question: {direction['research_question']}",
            f"First experiment: {direction['experiment']}",
            f"Metrics: {', '.join(direction.get('metrics', []))}",
            "Evidence: " + ", ".join(direction["source_urls"]),
            "",
        ])
    blocks.append("<hr><p><small>Fresh scan of arXiv, news, research/engineering blogs, Reddit, and publicly indexed X results. Social sources are treated as signals, not verified evidence.</small></p>")
    return subject, "".join(blocks), "\n".join(text)


def main():
    if not should_run_now():
        print("Skipping the non-Pacific-8-AM DST schedule.")
        return 0
    history = load_history()
    items = collect_sources()
    print(f"Collected {len(items)} unique source records")
    if len(items) < 6:
        raise RuntimeError("Too few sources collected to create a defensible digest")
    directions = rank_directions(items, history)
    subject, body_html, body_text = build_email(directions)
    if os.getenv("DRY_RUN", "").lower() in {"1", "true", "yes"}:
        print(subject)
        print(body_text)
        return 0
    if not send_email(subject, body_html, body_text):
        raise RuntimeError("Email delivery failed")

    entry = {
        "date": dt.datetime.now(PACIFIC).isoformat(),
        "directions": [{"title": d["title"]} for d in directions],
        "source_urls": sorted({u for d in directions for u in d["source_urls"]}),
        "digest_id": hashlib.sha256(body_text.encode()).hexdigest()[:12],
    }
    history.append(entry)
    HISTORY.write_text(json.dumps(history[-365:], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Sent {entry['digest_id']} to {os.getenv('EMAIL_TO')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
