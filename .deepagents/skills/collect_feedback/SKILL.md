# collect_feedback skill

Purpose
- Interactive CLI + JSON tools to collect structured feedback on sets of research papers (e.g., daily '10 papers' lists).
- Collects: relevance (per-item scoring / marking), difficulty, concepts needing explanation, prerequisite suggestions, and free-text notes.
- Maintains a project glossary of concepts (alphabetically sorted) and a project feedback memory file used to bias future recommendations.

Files
- collect_feedback.py — Python CLI implementation

Behavior & Memory
- Saves each feedback session as timestamped JSON in .deepagents/collect_feedback_outputs/
- Adds any new concepts to .deepagents/collect_feedback_glossary.md (alphabetically sorted)
- Updates .deepagents/collect_feedback_memory.json with lightweight preference signals derived from feedback (relevance counts, disliked topics, difficulty signals) so other skills or agents can read and personalize future paper selections.

CLI usage
- Interactive mode (recommended):
  python collect_feedback.py --interactive --session-name "day-2025-12-01" --papers papers.json

- Non-interactive: read feedback from a structured JSON file:
  python collect_feedback.py --input feedback_input.json --session-name day1

- Minimal run (reads papers from stdin JSON array):
  cat papers.json | python collect_feedback.py --interactive

Input formats
- papers.json: JSON array of paper objects. Example:
  [
    {"id": "paper1", "title": "Title 1", "url": "https://...", "abstract": "..."},
    {"id": "paper2", "title": "Title 2", "url": "...", "abstract": "..."}
  ]

- feedback_input.json (non-interactive): JSON object ``{"session_name": "...", "feedback": [{"id": "paper1", "relevance": 0, "difficulty": "hard", "concepts": ["X","Y"], "prereq": "Read A"}, ...]}``

Outputs
- Session JSON stored at: .deepagents/collect_feedback_outputs/<session_name>-<timestamp>.json
- Glossary at: .deepagents/collect_feedback_glossary.md (alphabetical, cumulative)
- Memory at: .deepagents/collect_feedback_memory.json (machine-readable summary of signals)

Notes
- The skill is intended to be reproducible: use JSON inputs to record the exact schema. The interactive prompts are just a convenience for quick sessions.
- Other project skills can read .deepagents/collect_feedback_memory.json to personalize next-day recommendations.

License: MIT
