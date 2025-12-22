A personalized research paper recommendation system that delivers daily curated picks and learns from email feedback.

**Core Functionality:**

**Daily Paper Selection** :

- Samples 12 papers from a curated CSV list (`to_read.csv`), excluding previously selected papers
- Uses LLM (OLMO-3-7B) to rank and select the top 3 most applicable/impactful papers based on:
  - Paper titles and abstracts
  - User preferences extracted from past email feedback
- Fetches ArXiv abstracts and generates AI-powered 1-2 sentence digests
- Updates `reading_progress.md` with daily picks (title, summary, digest, ArXiv link)
- Sends formatted email with top 3 recommendations via EmailJS or SendGrid
- Maintains selection history to prevent duplicate picks

**Email Feedback Processing** :

- Monitors Gmail inbox for replies to daily pick emails
- Extracts user feedback from email reply text
- Uses LLM (Llama-3.1-8B-instruct) to parse natural language feedback into structured preferences:
  - `topics_to_avoid`: Topics user explicitly wants less of
  - `topics_to_increase`: Topics user explicitly wants more of
  - `prereq_concepts`: Background concepts user needs explained
- Preferences in `prefs.jsonl` are incorporated into future paper rankings
