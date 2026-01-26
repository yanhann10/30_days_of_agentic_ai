#!/usr/bin/env python3
"""
Download papers from arXiv for parsing.
Selects 2-3 papers per subcategory for balanced coverage.
"""

import json
import os
import time
from pathlib import Path
from collections import defaultdict

try:
    import httpx
    USE_HTTPX = True
except ImportError:
    import urllib.request
    USE_HTTPX = False

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PDF_DIR = BASE_DIR / "data" / "pdfs"


def download_pdf(arxiv_id: str, output_path: Path) -> bool:
    """Download PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        if USE_HTTPX:
            with httpx.Client(follow_redirects=True, timeout=60) as client:
                response = client.get(url)
                response.raise_for_status()
                output_path.write_bytes(response.content)
        else:
            req = urllib.request.Request(url, headers={'User-Agent': 'KG-Builder/1.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                output_path.write_bytes(response.read())
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def select_papers_by_subcategory(papers: list[dict], per_subcategory: int = 2) -> list[dict]:
    """Select N papers per subcategory for balanced coverage."""
    by_subcategory = defaultdict(list)

    for paper in papers:
        if paper.get('arxiv_id'):
            subcat = paper.get('subcategory', 'Other') or 'Other'
            by_subcategory[subcat].append(paper)

    selected = []
    for subcat, subcat_papers in by_subcategory.items():
        # Take first N papers (they're typically most cited/important)
        selected.extend(subcat_papers[:per_subcategory])

    return selected


def main(limit: int = 10, per_subcategory: int = 2):
    """Download papers for parsing."""
    print("=" * 60)
    print("Downloading papers from arXiv")
    print("=" * 60)

    # Load papers
    papers_path = PROCESSED_DIR / "papers.json"
    with open(papers_path) as f:
        papers = json.load(f)

    print(f"Total papers: {len(papers)}")

    # Select by subcategory
    selected = select_papers_by_subcategory(papers, per_subcategory)
    if limit:
        selected = selected[:limit]

    print(f"Selected {len(selected)} papers for download")

    # Create output directory
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Download papers
    downloaded = []
    for i, paper in enumerate(selected):
        arxiv_id = paper['arxiv_id']
        title = paper['title'][:50]
        output_path = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"

        print(f"\n[{i+1}/{len(selected)}] {arxiv_id}: {title}...")

        if output_path.exists():
            print("  Already downloaded")
            downloaded.append(paper)
            continue

        if download_pdf(arxiv_id, output_path):
            print(f"  Downloaded to {output_path.name}")
            downloaded.append(paper)
            time.sleep(1)  # Rate limiting
        else:
            print("  Skipping")

    # Save download manifest
    manifest_path = PROCESSED_DIR / "downloaded_papers.json"
    with open(manifest_path, 'w') as f:
        json.dump(downloaded, f, indent=2)
    print(f"\nSaved manifest to {manifest_path}")

    print(f"\nDownloaded {len(downloaded)}/{len(selected)} papers")

    return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Max papers to download")
    parser.add_argument("--per-subcat", type=int, default=2, help="Papers per subcategory")
    args = parser.parse_args()

    main(limit=args.limit, per_subcategory=args.per_subcat)
