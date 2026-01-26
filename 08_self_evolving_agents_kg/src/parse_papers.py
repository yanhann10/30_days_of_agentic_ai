#!/usr/bin/env python3
"""
Parse downloaded PDFs to extract text and structure.
Uses PyMuPDF (free) or LlamaParse.
"""

import json
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PDF_DIR = BASE_DIR / "data" / "pdfs"


def parse_with_pymupdf(pdf_path: Path) -> dict:
    """Parse PDF using PyMuPDF (free, fast)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    doc = fitz.open(pdf_path)
    full_text = ""
    abstract = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += text + "\n"

        # Try to extract abstract from first 2 pages
        if page_num < 2 and not abstract:
            lower = text.lower()
            if "abstract" in lower:
                start = lower.find("abstract")
                # Find end of abstract (usually "introduction" or "1.")
                end_markers = ["introduction", "1.", "1 introduction"]
                end = len(text)
                for marker in end_markers:
                    idx = lower.find(marker, start + 100)
                    if idx > 0:
                        end = min(end, idx)
                abstract = text[start:end].strip()

    doc.close()

    return {
        "full_text": full_text,
        "abstract": abstract[:2000] if abstract else "",
        "char_count": len(full_text),
        "parser": "pymupdf"
    }


def parse_with_llamaparse(pdf_path: Path) -> dict:
    """Parse PDF using LlamaParse (better quality, requires API key)."""
    try:
        from llama_parse import LlamaParse
    except ImportError:
        raise ImportError("Install llama-parse: pip install llama-parse")

    # Check for API key
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not set")

    parser = LlamaParse(
        api_key=api_key,
        result_type="text",
        verbose=False
    )

    documents = parser.load_data(str(pdf_path))
    full_text = "\n".join([doc.text for doc in documents])

    # Extract abstract
    abstract = ""
    lower = full_text.lower()
    if "abstract" in lower:
        start = lower.find("abstract")
        end = lower.find("introduction", start + 100)
        if end > start:
            abstract = full_text[start:end].strip()

    return {
        "full_text": full_text,
        "abstract": abstract[:2000] if abstract else "",
        "char_count": len(full_text),
        "parser": "llamaparse"
    }


def parse_paper(pdf_path: Path, use_llamaparse: bool = False) -> dict:
    """Parse a single paper."""
    if use_llamaparse:
        try:
            return parse_with_llamaparse(pdf_path)
        except Exception as e:
            print(f"  LlamaParse failed ({e}), falling back to PyMuPDF")

    return parse_with_pymupdf(pdf_path)


def main(use_llamaparse: bool = False):
    """Parse all downloaded papers."""
    print("=" * 60)
    print("Parsing downloaded papers")
    print("=" * 60)

    # Load manifest
    manifest_path = PROCESSED_DIR / "downloaded_papers.json"
    if not manifest_path.exists():
        print("No download manifest found. Run download_papers.py first.")
        return

    with open(manifest_path) as f:
        papers = json.load(f)

    print(f"Papers to parse: {len(papers)}")

    parsed_papers = []
    for i, paper in enumerate(papers):
        arxiv_id = paper['arxiv_id']
        pdf_path = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"

        print(f"\n[{i+1}/{len(papers)}] {arxiv_id}: {paper['title'][:40]}...")

        if not pdf_path.exists():
            print("  PDF not found, skipping")
            continue

        try:
            parsed = parse_paper(pdf_path, use_llamaparse)
            paper['parsed_text'] = parsed['full_text'][:50000]  # Limit size
            paper['parsed_abstract'] = parsed['abstract']
            paper['parser_used'] = parsed['parser']
            paper['char_count'] = parsed['char_count']
            parsed_papers.append(paper)
            print(f"  Parsed: {parsed['char_count']:,} chars, abstract: {len(parsed['abstract'])} chars")
        except Exception as e:
            print(f"  Error parsing: {e}")

    # Save parsed papers
    output_path = PROCESSED_DIR / "parsed_papers.json"
    with open(output_path, 'w') as f:
        json.dump(parsed_papers, f, indent=2)
    print(f"\nSaved {len(parsed_papers)} parsed papers to {output_path}")

    return parsed_papers


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llamaparse", action="store_true", help="Use LlamaParse instead of PyMuPDF")
    args = parser.parse_args()

    main(use_llamaparse=args.llamaparse)
