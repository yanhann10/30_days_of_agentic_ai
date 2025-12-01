#!/usr/bin/env python3
"""
Script to manually add feedback from email replies.
Usage: python add_feedback.py "Your feedback text here"

The feedback will be stored and used to refine future paper picks.
"""
import sys, json, os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
FEEDBACK = os.path.join(ROOT, 'feedback.json')

if len(sys.argv) < 2:
    print("Usage: python add_feedback.py 'your feedback text'")
    sys.exit(1)

feedback_text = sys.argv[1]

# Load existing feedback
if os.path.exists(FEEDBACK):
    with open(FEEDBACK) as f:
        feedback_data = json.load(f)
else:
    feedback_data = []

# Add new feedback entry
entry = {
    'date': datetime.utcnow().isoformat(),
    'comment': feedback_text
}

# Optional: parse for preferred topics (simple keyword extraction)
topics = []
if 'more' in feedback_text.lower():
    # Extract what comes after 'more'
    parts = feedback_text.lower().split('more')
    if len(parts) > 1:
        topics.append(parts[1].strip()[:100])
if topics:
    entry['preferred_topics'] = topics

feedback_data.append(entry)

# Save feedback
with open(FEEDBACK, 'w') as f:
    json.dump(feedback_data, f, indent=2)

print(f"Feedback added successfully! Total feedback entries: {len(feedback_data)}")
