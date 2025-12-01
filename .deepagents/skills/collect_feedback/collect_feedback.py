#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_DIR = '/Users/hanyan/git_repo/30_days_of_agentic_ai'
DEEP_DIR = os.path.join(PROJECT_DIR, '.deepagents')
OUTPUT_DIR = os.path.join(DEEP_DIR, 'collect_feedback_outputs')
GLOSSARY_PATH = os.path.join(DEEP_DIR, 'collect_feedback_glossary.md')
MEMORY_PATH = os.path.join(DEEP_DIR, 'collect_feedback_memory.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(DEEP_DIR):
    os.makedirs(DEEP_DIR, exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def update_glossary(concepts):
    existing = []
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH, 'r', encoding='utf-8') as f:
            existing = [line.strip() for line in f if line.strip()]
    s = set(existing)
    s.update(concepts)
    sorted_list = sorted(s, key=lambda x: x.lower())
    with open(GLOSSARY_PATH, 'w', encoding='utf-8') as f:
        for c in sorted_list:
            f.write(c + '\n')


def update_memory(feedback_list):
    # lightweight signals: relevance_count, total_seen, difficulty_counts per paper/topic
    mem = load_json(MEMORY_PATH) or {'by_paper': {}, 'by_concept': {}}
    for fb in feedback_list:
        pid = fb.get('id')
        rel = fb.get('relevance')
        difficulty = fb.get('difficulty')
        concepts = fb.get('concepts', [])
        if pid not in mem['by_paper']:
            mem['by_paper'][pid] = {'seen': 0, 'relevance_sum': 0, 'difficulty': {}}
        p = mem['by_paper'][pid]
        p['seen'] += 1
        if isinstance(rel, (int, float)):
            p['relevance_sum'] += rel
        if difficulty:
            p['difficulty'][difficulty] = p['difficulty'].get(difficulty, 0) + 1
        for c in concepts:
            if c not in mem['by_concept']:
                mem['by_concept'][c] = {'count': 0, 'difficulty': {}}
            mem['by_concept'][c]['count'] += 1
            if difficulty:
                mem['by_concept'][c]['difficulty'][difficulty] = mem['by_concept'][c]['difficulty'].get(difficulty, 0) + 1
    save_json(MEMORY_PATH, mem)


def interactive_session(papers, session_name):
    feedbacks = []
    print(f"Starting interactive feedback session: {session_name}")
    print("For relevance you can enter 0-10 or mark 'irrelevant'. For difficulty: easy/medium/hard/unknown.")
    for i, p in enumerate(papers, start=1):
        print('\n')
        print(f"[{i}] {p.get('title')} (id: {p.get('id')})")
        print(p.get('url', ''))
        print('\nAbstract:')
        print(p.get('abstract', '')[:800])
        rel = input('Relevance (0-10 or irrelevant): ').strip()
        if rel.lower() == 'irrelevant':
            rel_val = 0
        else:
            try:
                rel_val = float(rel)
            except:
                rel_val = None
        diff = input('Difficulty (easy/medium/hard/unknown): ').strip().lower()
        concepts = input('Concepts needing explanation (comma-separated): ').strip()
        concepts_list = [c.strip() for c in concepts.split(',') if c.strip()]
        prereq = input('If difficult, suggested prerequisite (optional): ').strip()
        notes = input('Free-text notes (optional): ').strip()
        fb = {'id': p.get('id'), 'title': p.get('title'), 'relevance': rel_val, 'difficulty': diff or None, 'concepts': concepts_list, 'prereq': prereq or None, 'notes': notes or None}
        feedbacks.append(fb)
    return feedbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--papers', type=str, help='Path to JSON array of papers')
    parser.add_argument('--input', type=str, help='Path to feedback input JSON (non-interactive)')
    parser.add_argument('--session-name', type=str, default=None)
    args = parser.parse_args()

    papers = []
    if args.papers:
        with open(args.papers, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    else:
        if not sys.stdin.isatty():
            papers = json.load(sys.stdin)

    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
            session_name = data.get('session_name') or args.session_name or datetime.utcnow().isoformat()
            feedbacks = data.get('feedback', [])
    else:
        session_name = args.session_name or datetime.utcnow().strftime('session-%Y%m%dT%H%M%SZ')
        if args.interactive:
            feedbacks = interactive_session(papers, session_name)
        else:
            print('No input provided. Use --interactive with --papers or provide --input feedback file.', file=sys.stderr)
            sys.exit(2)

    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_name = f"{session_name}-{timestamp}.json"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    save_json(out_path, {'session': session_name, 'timestamp': timestamp, 'feedback': feedbacks})

    # update glossary and memory
    all_concepts = []
    for fb in feedbacks:
        all_concepts.extend(fb.get('concepts', []))
    update_glossary(all_concepts)
    update_memory(feedbacks)

    print(f"Saved session to {out_path}")
    print(f"Updated glossary at {GLOSSARY_PATH}")
    print(f"Updated memory at {MEMORY_PATH}")

if __name__ == '__main__':
    main()
