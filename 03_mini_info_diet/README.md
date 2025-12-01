### Mini Info Diet

This mini-project pulls three papers each day from `to_read.csv`, emails them via the `daily_picks` GitHub Action, and records outcomes in `reading_progress.md`. You can give feedback on each email to refine the ranking logic so future picks stay relevant.

#### How Feedback Works

1. **Receive daily picks**: You'll get an email with the top 3 paper recommendations
2. **Reply with feedback**: Respond to the email with your preferences (e.g., "prefer more papers on tool use", "less interested in surveys")
3. **Add feedback manually**: Run `python scripts/add_feedback.py "your feedback text"` to store your preferences
4. **Refined picks**: The system uses your last 5 feedback entries to personalize future recommendations

The feedback is stored in `feedback.json` and automatically influences the next day's picks by adjusting the AI's ranking criteria.
