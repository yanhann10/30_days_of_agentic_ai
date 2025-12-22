# Mini Info Diet

A personalized research paper recommendation system that delivers daily curated picks from a curated reading list and learns from your email feedback to continuously refine recommendations.

## Overview

Mini Info Diet helps you stay on top of research papers by:

- **Curating daily picks**: Automatically selects the top 3 most relevant papers from your reading list
- **Learning from feedback**: Processes your email replies to understand preferences and adjust future recommendations

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

The workflow runs daily via GitHub Actions and contains these functionalities:

- **LLM based selection**: Uses LLM (OLMO-3-7B) to shortlist to-reads based on relevance
- **Maintaining context**: Tracks reading progress and selection history to avoid duplicates
- **Multi-tier Search**: ArXiv API / Serper / openreview
- **Content Enrichment**: PDF extraction + structured insight generation
- **Preference-aware**: Email feedback integration
- **Stateful Progress Tracking**: Automatic reading status updates from email replies

### Email Feedback Processing

The system learns from user preferences through email replies by monitoring the Gmail inbox and extract structured preferences:

- `topics_to_avoid`: Topics you explicitly want less of
- `topics_to_increase`: Topics you explicitly want more of
- `prereq_concepts`: Background concepts you need explained
  Preferences are stored in `prefs.jsonl` and automatically incorporated into future paper selection.

## Setup

See [setup.md](setup.md) for detailed setup instructions, including:

- GitHub Actions configuration
- Email provider setup (SendGrid/EmailJS)
- Gmail OAuth setup for feedback processing
- Required API keys and secrets
