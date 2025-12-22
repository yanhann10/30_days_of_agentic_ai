# Mini Info Diet

A personalized research paper recommendation system that delivers daily curated picks from a curated reading list and learns from your email feedback to continuously refine recommendations.

## Overview

Mini Info Diet helps you stay on top of research papers by:

- **Curating daily picks**: Automatically selects the top 3 most relevant papers from your reading list
- **Learning from feedback**: Processes your email replies to understand preferences and adjust future recommendations
- **Maintaining context**: Tracks reading progress and selection history to avoid duplicates

## Architecture

```
┌──────────────────┐      ┌──────────────┐
│  daily_picks.py   │◄─────│ prefs.jsonl  │
│  (LLM ranking)    │      │ (preferences)│
└──────┬───────────┘      └──────────────┘
       │
       ▼
┌──────────────────┐
│  Email Delivery   │
│  (Top 3 picks)    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│process_email_    │
│feedback.py       │
└──────┬───────────┘
       │
       └──────────────┐
                      ▼
              ┌──────────────┐
              │ prefs.jsonl  │
              └──────────────┘
```

## How It Works

### Daily Paper Selection

The system runs daily (via GitHub Actions) to:

1. **Sample candidates**: Selects 12 papers from `to_read.csv`, excluding previously picked papers
2. **AI-powered ranking**: Uses LLM (OLMO-3-7B) to rank papers based on:
   - Paper titles and ArXiv abstracts
   - Your past preferences extracted from email feedback
   - Relevance, impact, and innovation criteria
3. **Enrichment**: Fetches ArXiv abstracts and generates concise 1-2 sentence digests
4. **Delivery**: Sends formatted email with top 3 picks via EmailJS or SendGrid
5. **Tracking**: Updates `reading_progress.md` and maintains selection history in `picks_history.json`

### Email Feedback Processing

The system learns from your preferences through email replies:

1. **Receive daily picks**: You get an email with the top 3 paper recommendations
2. **Reply with feedback**: Respond naturally (e.g., "prefer more papers on tool use", "less interested in surveys")
3. **Automatic parsing**: The system monitors your Gmail inbox and uses LLM (Llama-3.1-8B-instruct) to extract structured preferences:
   - `topics_to_avoid`: Topics you explicitly want less of
   - `topics_to_increase`: Topics you explicitly want more of
   - `prereq_concepts`: Background concepts you need explained
4. **Continuous improvement**: Preferences are stored in `prefs.jsonl` and automatically incorporated into future paper rankings

The feedback loop ensures recommendations become more aligned with your interests over time, without requiring explicit thumbs up/down interactions.

## Setup

See [setup.md](setup.md) for detailed setup instructions, including:

- GitHub Actions configuration
- Email provider setup (SendGrid/EmailJS)
- Gmail OAuth setup for feedback processing
- Required API keys and secrets

## Files

- `to_read.csv`: Curated list of papers (title, ArXiv link)
- `reading_progress.md`: Daily picks and reading status
- `picks_history.json`: History of selected papers to prevent duplicates
- `prefs.jsonl`: User preferences extracted from email feedback
- `scripts/daily_picks.py`: Daily selection and email sending
- `scripts/process_email_feedback.py`: Gmail monitoring and preference extraction
