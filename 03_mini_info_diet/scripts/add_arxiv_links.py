#!/usr/bin/env python3
import os, requests, time
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.environ.get('SERPER_API_KEY')
ROOT = os.path.dirname(os.path.dirname(__file__))
TO_READ = os.path.join(ROOT, 'to_read.csv')

papers = []
with open(TO_READ) as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        papers.append({
            'title': parts[0],
            'link': parts[1] if len(parts) > 1 else ''
        })

missing_count = 0
for i, paper in enumerate(papers):
    if paper['link']:
        continue

    missing_count += 1
    print(f"Searching for: {paper['title']}")

    query = f"{paper['title']} site:arxiv.org"
    resp = requests.post(
        'https://google.serper.dev/search',
        headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
        json={'q': query, 'num': 3}
    )

    if resp.status_code != 200:
        print(f"  Search failed: {resp.status_code}")
        continue

    results = resp.json().get('organic', [])
    arxiv_link = None

    for result in results:
        url = result.get('link', '')
        if 'arxiv.org/abs/' in url:
            arxiv_link = url
            break

    if arxiv_link:
        papers[i]['link'] = arxiv_link
        print(f"  Found: {arxiv_link}")
    else:
        print(f"  Not found")

    time.sleep(1)

with open(TO_READ, 'w') as f:
    for paper in papers:
        f.write(f"{paper['title']}\t{paper['link']}\n")

print(f"\nDone! Added {missing_count} arXiv links")
