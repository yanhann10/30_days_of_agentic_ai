#!/usr/bin/env python3
"""
Enrich paper data with abstracts, citations, and metadata from external APIs.

Uses:
- arXiv API for abstracts and metadata
- Semantic Scholar API for citations
- Multi-agent debate framework for relationship inference (optional)

Output:
- data/processed/enriched_papers.json
"""

import json
import re
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from xml.etree import ElementTree as ET

try:
    import httpx
    USE_HTTPX = True
except ImportError:
    import urllib.request
    import urllib.error
    USE_HTTPX = False

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# API endpoints
ARXIV_API = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"

# Rate limiting
ARXIV_DELAY = 3.0  # seconds between arXiv requests
SEMANTIC_SCHOLAR_DELAY = 1.0  # seconds between S2 requests (100/5min = 1.2s avg)


@dataclass
class EnrichedPaper:
    """Paper with enriched metadata."""
    title: str
    url: str
    description: str
    category: str
    subcategory: str
    arxiv_id: Optional[str] = None
    year: Optional[int] = None
    # Enriched fields
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    references: list[str] = field(default_factory=list)  # arXiv IDs this paper cites
    cited_by: list[str] = field(default_factory=list)  # arXiv IDs that cite this paper
    semantic_scholar_id: Optional[str] = None
    code_url: Optional[str] = None
    enrichment_status: str = "pending"  # pending, success, partial, failed


def fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch URL content with httpx or urllib fallback."""
    if USE_HTTPX:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return response.text
    else:
        req = urllib.request.Request(url, headers={'User-Agent': 'KG-Builder/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode('utf-8')


def fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch JSON from URL."""
    if USE_HTTPX:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return response.json()
    else:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'KG-Builder/1.0',
            'Accept': 'application/json'
        })
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))


def enrich_from_arxiv(arxiv_id: str) -> dict:
    """Fetch paper metadata from arXiv API."""
    url = f"{ARXIV_API}?id_list={arxiv_id}"

    try:
        xml_content = fetch_url(url)
        root = ET.fromstring(xml_content)

        # arXiv API uses Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entry = root.find('atom:entry', ns)
        if entry is None:
            return {"error": "No entry found"}

        # Extract fields
        title_elem = entry.find('atom:title', ns)
        abstract_elem = entry.find('atom:summary', ns)

        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)

        # Get published date for venue/year
        published = entry.find('atom:published', ns)
        year = None
        if published is not None and published.text:
            year = int(published.text[:4])

        return {
            "title": title_elem.text.strip().replace('\n', ' ') if title_elem is not None else None,
            "abstract": abstract_elem.text.strip().replace('\n', ' ') if abstract_elem is not None else None,
            "authors": authors,
            "year": year,
        }
    except Exception as e:
        return {"error": str(e)}


def enrich_from_semantic_scholar(arxiv_id: str) -> dict:
    """Fetch citation data from Semantic Scholar API."""
    # Semantic Scholar uses ARXIV: prefix for arXiv papers
    paper_id = f"ARXIV:{arxiv_id}"
    fields = "citationCount,references.externalIds,citations.externalIds,paperId"
    url = f"{SEMANTIC_SCHOLAR_API}/{paper_id}?fields={fields}"

    try:
        data = fetch_json(url)

        # Extract arXiv IDs from references
        references = []
        for ref in data.get('references', []) or []:
            ext_ids = ref.get('externalIds', {}) or {}
            if 'ArXiv' in ext_ids:
                references.append(ext_ids['ArXiv'])

        # Extract arXiv IDs from citations
        cited_by = []
        for cit in data.get('citations', []) or []:
            ext_ids = cit.get('externalIds', {}) or {}
            if 'ArXiv' in ext_ids:
                cited_by.append(ext_ids['ArXiv'])

        return {
            "citation_count": data.get('citationCount'),
            "references": references,
            "cited_by": cited_by,
            "semantic_scholar_id": data.get('paperId'),
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": "Paper not found in Semantic Scholar"}
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def enrich_paper(paper: dict, use_semantic_scholar: bool = True) -> EnrichedPaper:
    """Enrich a single paper with external data."""
    enriched = EnrichedPaper(
        title=paper['title'],
        url=paper['url'],
        description=paper['description'],
        category=paper['category'],
        subcategory=paper['subcategory'],
        arxiv_id=paper.get('arxiv_id'),
        year=paper.get('year'),
    )

    if not enriched.arxiv_id:
        enriched.enrichment_status = "failed"
        return enriched

    # Fetch from arXiv
    arxiv_data = enrich_from_arxiv(enriched.arxiv_id)

    if "error" not in arxiv_data:
        enriched.abstract = arxiv_data.get('abstract')
        enriched.authors = arxiv_data.get('authors', [])
        if arxiv_data.get('year'):
            enriched.year = arxiv_data['year']
        enriched.enrichment_status = "partial"

    time.sleep(ARXIV_DELAY)

    # Fetch from Semantic Scholar
    if use_semantic_scholar:
        s2_data = enrich_from_semantic_scholar(enriched.arxiv_id)

        if "error" not in s2_data:
            enriched.citation_count = s2_data.get('citation_count')
            enriched.references = s2_data.get('references', [])
            enriched.cited_by = s2_data.get('cited_by', [])
            enriched.semantic_scholar_id = s2_data.get('semantic_scholar_id')
            enriched.enrichment_status = "success"

        time.sleep(SEMANTIC_SCHOLAR_DELAY)
    elif enriched.abstract:
        enriched.enrichment_status = "partial"

    return enriched


def load_papers() -> list[dict]:
    """Load parsed papers from JSON."""
    papers_path = PROCESSED_DIR / "papers.json"
    with open(papers_path) as f:
        return json.load(f)


def save_enriched_papers(papers: list[EnrichedPaper]):
    """Save enriched papers to JSON."""
    output_path = PROCESSED_DIR / "enriched_papers.json"
    with open(output_path, 'w') as f:
        json.dump([asdict(p) for p in papers], f, indent=2)
    print(f"Saved enriched papers to {output_path}")


def main(limit: int = None, skip_semantic_scholar: bool = False):
    """Main entry point."""
    print("=" * 60)
    print("Enriching paper data from arXiv and Semantic Scholar")
    print("=" * 60)

    papers = load_papers()
    print(f"Loaded {len(papers)} papers")

    if limit:
        papers = papers[:limit]
        print(f"Processing first {limit} papers only")

    enriched_papers = []
    success_count = 0
    partial_count = 0
    failed_count = 0

    for i, paper in enumerate(papers):
        print(f"\n[{i+1}/{len(papers)}] {paper['title'][:50]}...")

        enriched = enrich_paper(paper, use_semantic_scholar=not skip_semantic_scholar)
        enriched_papers.append(enriched)

        if enriched.enrichment_status == "success":
            success_count += 1
            print(f"  Success - {enriched.citation_count or 0} citations")
        elif enriched.enrichment_status == "partial":
            partial_count += 1
            print(f"  Partial - got abstract only")
        else:
            failed_count += 1
            print(f"  Failed - no arXiv ID")

        # Save progress periodically
        if (i + 1) % 10 == 0:
            save_enriched_papers(enriched_papers)
            print(f"\n--- Progress saved ({i+1}/{len(papers)}) ---")

    # Final save
    save_enriched_papers(enriched_papers)

    # Summary
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Total papers: {len(papers)}")
    print(f"  Success: {success_count}")
    print(f"  Partial: {partial_count}")
    print(f"  Failed: {failed_count}")

    # Citation statistics
    citations = [p.citation_count for p in enriched_papers if p.citation_count]
    if citations:
        print(f"\nCitation statistics:")
        print(f"  Total citations: {sum(citations)}")
        print(f"  Average: {sum(citations) / len(citations):.1f}")
        print(f"  Max: {max(citations)}")

    return enriched_papers


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enrich paper data")
    parser.add_argument("--limit", type=int, help="Limit number of papers to process")
    parser.add_argument("--skip-s2", action="store_true", help="Skip Semantic Scholar API")
    args = parser.parse_args()

    main(limit=args.limit, skip_semantic_scholar=args.skip_s2)
