#!/usr/bin/env python3
import os, time, requests, xml.etree.ElementTree as ET

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

def fetch_arxiv_by_title(title):
    try:
        params = {
            "search_query": f'all:"{title}"',
            "start": 0,
            "max_results": 1
        }
        headers = {"User-Agent": "arxiv-fetcher/0.1"}
        r = requests.get(ARXIV_API_URL, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return None, None

        root = ET.fromstring(r.text)
        entry = root.find("atom:entry", NS)
        if entry is None:
            return None, None

        url_elem = entry.find("atom:id", NS)
        abstract_elem = entry.find("atom:summary", NS)

        url = url_elem.text.strip() if url_elem is not None else None
        abstract = abstract_elem.text.strip() if abstract_elem is not None else None

        return url, abstract
    except Exception as e:
        print(f"  Error fetching: {e}")
        return None, None

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
found_count = 0

for i, paper in enumerate(papers):
    if paper['link']:
        continue

    missing_count += 1
    print(f"Searching for: {paper['title']}")

    url, abstract = fetch_arxiv_by_title(paper['title'])

    if url:
        papers[i]['link'] = url
        found_count += 1
        print(f"  Found: {url}")
    else:
        print(f"  Not found")

    time.sleep(3)

with open(TO_READ, 'w') as f:
    for paper in papers:
        f.write(f"{paper['title']}\t{paper['link']}\n")

print(f"\nDone! Found {found_count}/{missing_count} arXiv links")
