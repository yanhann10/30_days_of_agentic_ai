# Multi-Agent Debates

Expand perspectives through structured adversarial reasoning.

## Overview

Two debate modes for document refinement:

- **Multi-Agent**: Two agents exchange arguments across N rounds; Moderator synthesizes both into final recommendations.

- **Self-Debate**: Single agent argues proponent and opponent positions for N rounds, then synthesizes. Useful when you want dialectic reasoning without multiple agent overhead.

Both modes output novelty scores, factuality scores, and actionable document improvements.

## Installation

```bash
uv add langchain-anthropic langchain-core langgraph langsmith python-dotenv
export ANTHROPIC_API_KEY=your_key
```

## Usage

```python
from debate_graph import run_debate

with open("input.md", "r", encoding="utf-8") as f:
    document_content = f.read()

result = run_debate(
    document_content=document_content,
    topic="What's missing from this analysis?",
    output_path="outputs/input_enriched.md"
)

print(result["moderator_decision"])
print(result["novelty_score"])      # 0-1
print(result["factuality_score"])   # 0-1
```

Notes:
- `document_content` is the full text of your input document (read from a file like `input.md`).
- For multi-round self-debate, use `run_self_debate(..., num_rounds=3)` in `agents.py`.
- Output is saved as `outputs/<input>_enriched.md`, containing the original document expanded/refined with more robust perspectives.

CLI example (generates `outputs/<input>_enriched.md`):
```bash
uv run python main.py --document input.md
```
