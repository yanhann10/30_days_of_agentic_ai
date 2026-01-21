#!/usr/bin/env python3
"""Refresh document sections using Gemini API."""

import os
import sys
from pathlib import Path
from datetime import datetime

import google.generativeai as genai

SECTIONS = [
    "RLHF Evaluation Frameworks",
    "Agentic Evaluation Research",
    "LLM-as-a-Judge",
    "Multimodal and Audio Evaluation",
    "Long-Context Evaluation"
]

def refresh_document(doc_path: str) -> bool:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set")
        return False

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    content = Path(doc_path).read_text()

    prompt = f"""Review this LLM evaluation document for outdated information.
Current date: {datetime.now().strftime("%B %Y")}

DOCUMENT (truncated):
{content[:12000]}

Check for:
1. New benchmarks or frameworks released
2. Outdated statistics or claims
3. Missing recent developments

If updates needed, respond with specific additions in markdown.
If no updates needed, respond: NO_UPDATES_NEEDED

Be conservative - only suggest significant changes."""

    try:
        response = model.generate_content(prompt)
        result = response.text

        if "NO_UPDATES_NEEDED" in result:
            print("No updates needed")
            return False

        # Append updates section
        update_section = f"\n\n### Updates ({datetime.now().strftime('%B %Y')})\n\n{result}\n"
        Path(doc_path).write_text(content + update_section)
        print("Document updated")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    doc = sys.argv[1] if len(sys.argv) > 1 else "agentic_eval.md"
    refresh_document(doc)
