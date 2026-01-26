#!/usr/bin/env python3
"""
Parse the Awesome-Self-Evolving-Agents README to extract papers and taxonomy.

Output:
- data/raw/awesome_readme.md (cached README)
- data/processed/papers.json (parsed paper entries)
- data/processed/categories.json (taxonomy structure)
"""

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
try:
    import httpx
    USE_HTTPX = True
except ImportError:
    import urllib.request
    USE_HTTPX = False

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

GITHUB_RAW_URL = "https://raw.githubusercontent.com/EvoAgentX/Awesome-Self-Evolving-Agents/main/README.md"


@dataclass
class Paper:
    title: str
    url: str
    description: str
    category: str
    subcategory: str
    arxiv_id: Optional[str] = None
    year: Optional[int] = None


@dataclass
class Category:
    name: str
    level: int  # 1 = top level, 2 = subcategory, etc.
    parent: Optional[str] = None
    description: Optional[str] = None


def fetch_readme() -> str:
    """Fetch README from GitHub and cache locally."""
    cache_path = RAW_DIR / "awesome_readme.md"

    # Use cache if exists and recent (< 1 day old)
    if cache_path.exists():
        import time
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"Using cached README ({age_hours:.1f} hours old)")
            return cache_path.read_text()

    print("Fetching README from GitHub...")
    if USE_HTTPX:
        response = httpx.get(GITHUB_RAW_URL, timeout=30)
        response.raise_for_status()
        content = response.text
    else:
        with urllib.request.urlopen(GITHUB_RAW_URL, timeout=30) as response:
            content = response.read().decode('utf-8')

    # Cache it
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content)
    print(f"Cached README to {cache_path}")

    return content


def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from URL."""
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'arxiv\.org/abs/(\w+-\w+/\d+)',  # old format
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_year_from_title(title: str) -> Optional[int]:
    """Extract year from paper title if present."""
    # Common patterns: (2024), [2024], '24, etc.
    patterns = [
        r"\((\d{4})\)",
        r"\[(\d{4})\]",
        r"'(\d{2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            year = match.group(1)
            if len(year) == 2:
                year = "20" + year
            return int(year)
    return None


def parse_paper_line(line: str, category: str, subcategory: str) -> Optional[Paper]:
    """Parse a markdown line containing a paper entry."""
    line = line.strip()

    # Skip non-paper lines
    if not line.startswith('- '):
        return None

    # Pattern 1: - (*Venue'YY*) Title [[Paper](url)] [[Code](url)]
    # Match venue part first, then capture everything up to [[
    venue_match = re.match(r'-\s*\(\*([^)]+)\*\)\s*', line)

    if venue_match:
        venue = venue_match.group(1).strip()
        rest_of_line = line[venue_match.end():]

        # Find title - everything up to [[
        title_end = rest_of_line.find('[[')
        if title_end == -1:
            title_end = rest_of_line.find('[')
        if title_end == -1:
            return None

        title = rest_of_line[:title_end].strip().rstrip(':').strip()
        links_part = rest_of_line[title_end:]

        # Extract paper URL from [[Paper](url)] or [Paper](url)
        paper_url_match = re.search(r'\[+[^\]]*(?:Paper|paper)[^\]]*\]+\(([^)]+)\)', links_part)

        if paper_url_match:
            url = paper_url_match.group(1)
        else:
            # Try simpler pattern - first link
            simple_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', links_part)
            if simple_match:
                url = simple_match.group(2)
            else:
                return None

        # No description in this format - title is the description
        description = ""

        # Extract year from venue (e.g., "ICLR'24" -> 2024)
        year_match = re.search(r"'(\d{2})", venue)
        year = int("20" + year_match.group(1)) if year_match else None

        arxiv_id = extract_arxiv_id(url)

        return Paper(
            title=title,
            url=url,
            description=description,
            category=category,
            subcategory=subcategory,
            arxiv_id=arxiv_id,
            year=year
        )

    # Pattern 2: Standard markdown - [Title](URL) - Description
    pattern = r'-\s*\*?\*?\[([^\]]+)\]\(([^)]+)\)\*?\*?\s*[-:]?\s*(.*)'
    match = re.match(pattern, line)

    if not match:
        return None

    title = match.group(1).strip()
    url = match.group(2).strip()
    description = match.group(3).strip()

    # Skip if it's a badge or image
    if 'img.shields.io' in url or url.endswith(('.png', '.jpg', '.svg')):
        return None

    arxiv_id = extract_arxiv_id(url)
    year = extract_year_from_title(title)

    return Paper(
        title=title,
        url=url,
        description=description,
        category=category,
        subcategory=subcategory,
        arxiv_id=arxiv_id,
        year=year
    )


def parse_readme(content: str) -> tuple[list[Paper], list[Category]]:
    """Parse README content to extract papers and categories."""
    papers = []
    categories = []

    current_category = ""
    current_subcategory = ""
    current_subsubcategory = ""

    lines = content.split('\n')

    for line in lines:
        line = line.rstrip()

        # Skip empty lines
        if not line:
            continue

        # Detect headers (categories)
        if line.startswith('# '):
            # Top level - usually just title, skip
            continue
        elif line.startswith('## '):
            current_category = line[3:].strip()
            current_subcategory = ""
            current_subsubcategory = ""
            categories.append(Category(
                name=current_category,
                level=1,
                parent=None
            ))
        elif line.startswith('### '):
            current_subcategory = line[4:].strip()
            current_subsubcategory = ""
            categories.append(Category(
                name=current_subcategory,
                level=2,
                parent=current_category
            ))
        elif line.startswith('#### '):
            current_subsubcategory = line[5:].strip()
            categories.append(Category(
                name=current_subsubcategory,
                level=3,
                parent=current_subcategory
            ))

        # Detect paper entries (lines starting with -)
        elif line.strip().startswith('- '):
            # Use most specific category available
            effective_subcategory = current_subsubcategory or current_subcategory
            paper = parse_paper_line(line, current_category, effective_subcategory)
            if paper:
                papers.append(paper)

    return papers, categories


def main():
    """Main entry point."""
    print("=" * 60)
    print("Parsing Awesome-Self-Evolving-Agents README")
    print("=" * 60)

    # Fetch README
    content = fetch_readme()
    print(f"README length: {len(content):,} characters")

    # Parse content
    papers, categories = parse_readme(content)

    print(f"\nExtracted:")
    print(f"  - {len(papers)} papers")
    print(f"  - {len(categories)} categories")

    # Save papers
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    papers_path = PROCESSED_DIR / "papers.json"
    with open(papers_path, 'w') as f:
        json.dump([asdict(p) for p in papers], f, indent=2)
    print(f"\nSaved papers to {papers_path}")

    # Save categories
    categories_path = PROCESSED_DIR / "categories.json"
    with open(categories_path, 'w') as f:
        json.dump([asdict(c) for c in categories], f, indent=2)
    print(f"Saved categories to {categories_path}")

    # Print category summary
    print("\nCategory breakdown:")
    category_counts = {}
    for paper in papers:
        cat = paper.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Print sample papers
    print("\nSample papers:")
    for paper in papers[:5]:
        print(f"  - {paper.title[:60]}...")
        print(f"    Category: {paper.category} > {paper.subcategory}")
        if paper.arxiv_id:
            print(f"    arXiv: {paper.arxiv_id}")

    return papers, categories


if __name__ == "__main__":
    main()
