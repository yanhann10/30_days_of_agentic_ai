#!/usr/bin/env python3
"""
Extract methods and build a method-connection graph.
NOT a citation graph - focuses on conceptual relationships.

Uses TF-IDF style approach for method importance and co-occurrence for connections.
"""

import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
import math

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Known method patterns - seed terms for evolving agent methods
SEED_METHODS = {
    # Training-based
    "supervised fine-tuning", "sft", "reinforcement learning", "rl", "rlhf",
    "ppo", "dpo", "direct preference optimization", "reward model",
    "self-training", "bootstrapping", "curriculum learning",

    # Inference-time
    "chain-of-thought", "cot", "react", "reflection", "self-refine",
    "tree-of-thought", "tot", "monte carlo tree search", "mcts",
    "beam search", "best-of-n", "self-consistency",

    # Prompt optimization
    "prompt tuning", "prompt optimization", "evolutionary prompt",
    "gradient-based prompt", "meta-prompt", "automatic prompt",

    # Memory
    "retrieval-augmented", "rag", "memory bank", "episodic memory",
    "working memory", "long-term memory", "context compression",

    # Tool use
    "tool learning", "tool use", "api call", "function calling",
    "code execution", "code interpreter",

    # Multi-agent
    "multi-agent", "agent collaboration", "debate", "consensus",
    "role-playing", "delegation", "coordinator",

    # Self-evolution
    "self-evolving", "self-improvement", "meta-learning",
    "neural architecture search", "automl", "self-play",
    "genetic algorithm", "evolutionary", "mutation", "selection"
}


@dataclass
class Method:
    """A method/technique extracted from papers."""
    name: str
    normalized_name: str
    papers: list[str] = field(default_factory=list)  # arxiv_ids
    frequency: int = 0
    tf_idf_score: float = 0.0
    category: str = ""


@dataclass
class MethodConnection:
    """Connection between two methods based on co-occurrence."""
    method1: str
    method2: str
    co_occurrence_count: int = 0
    papers: list[str] = field(default_factory=list)  # papers mentioning both
    strength: float = 0.0  # normalized connection strength


def normalize_method_name(name: str) -> str:
    """Normalize method name to canonical form."""
    name = name.lower().strip()
    # Common normalizations
    normalizations = {
        "cot": "chain-of-thought",
        "tot": "tree-of-thought",
        "rag": "retrieval-augmented generation",
        "rl": "reinforcement learning",
        "sft": "supervised fine-tuning",
        "ppo": "proximal policy optimization",
        "dpo": "direct preference optimization",
        "mcts": "monte carlo tree search",
        "llm": "large language model",
    }
    return normalizations.get(name, name)


def extract_methods_from_text(text: str, title: str = "") -> list[str]:
    """Extract method mentions from paper text."""
    text_lower = text.lower()
    title_lower = title.lower()
    found_methods = []

    # Check seed methods
    for method in SEED_METHODS:
        # Check in title (higher weight) and text
        if method in title_lower or method in text_lower:
            # Count occurrences
            count = text_lower.count(method)
            if count >= 2:  # At least 2 mentions to be significant
                found_methods.append(normalize_method_name(method))

    # Also extract method-like patterns
    # Pattern: "X method", "X approach", "X algorithm", "X technique"
    patterns = [
        r'\b(\w+(?:-\w+)*)\s+(?:method|approach|algorithm|technique|framework|model)\b',
        r'\b(?:using|via|with|through)\s+(\w+(?:-\w+)*)\b',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match) > 3 and match not in ['the', 'our', 'this', 'new', 'proposed']:
                found_methods.append(normalize_method_name(match))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for m in found_methods:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    return unique


def compute_tf_idf(method_docs: dict[str, list[str]], total_docs: int) -> dict[str, float]:
    """Compute TF-IDF scores for methods."""
    scores = {}
    for method, docs in method_docs.items():
        tf = len(docs)  # Term frequency (number of docs containing method)
        idf = math.log(total_docs / (1 + tf))  # Inverse document frequency
        scores[method] = tf * idf
    return scores


def build_method_graph(papers: list[dict]) -> tuple[list[Method], list[MethodConnection]]:
    """Build method-connection graph from parsed papers."""
    # Extract methods from each paper
    paper_methods = {}  # arxiv_id -> list of methods
    method_papers = defaultdict(list)  # method -> list of arxiv_ids

    for paper in papers:
        arxiv_id = paper.get('arxiv_id', '')
        text = paper.get('parsed_text', '') or paper.get('abstract', '')
        title = paper.get('title', '')

        if not text:
            continue

        methods = extract_methods_from_text(text, title)
        paper_methods[arxiv_id] = methods

        for method in methods:
            method_papers[method].append(arxiv_id)

    # Compute TF-IDF scores
    tf_idf_scores = compute_tf_idf(method_papers, len(papers))

    # Create Method objects
    methods = []
    for method_name, paper_list in method_papers.items():
        if len(paper_list) >= 1:  # Method appears in at least 1 paper
            m = Method(
                name=method_name,
                normalized_name=normalize_method_name(method_name),
                papers=paper_list,
                frequency=len(paper_list),
                tf_idf_score=tf_idf_scores.get(method_name, 0),
                category=categorize_method(method_name)
            )
            methods.append(m)

    # Sort by frequency
    methods.sort(key=lambda m: m.frequency, reverse=True)

    # Build connections based on co-occurrence
    connections = []
    method_names = list(method_papers.keys())

    for i, m1 in enumerate(method_names):
        for m2 in method_names[i+1:]:
            # Find papers that mention both methods
            common_papers = set(method_papers[m1]) & set(method_papers[m2])
            if common_papers:
                # Calculate connection strength (Jaccard similarity)
                union = set(method_papers[m1]) | set(method_papers[m2])
                strength = len(common_papers) / len(union) if union else 0

                conn = MethodConnection(
                    method1=m1,
                    method2=m2,
                    co_occurrence_count=len(common_papers),
                    papers=list(common_papers),
                    strength=strength
                )
                connections.append(conn)

    # Sort connections by strength
    connections.sort(key=lambda c: c.strength, reverse=True)

    return methods, connections


def categorize_method(method: str) -> str:
    """Categorize method into high-level categories."""
    method_lower = method.lower()

    training_keywords = ['training', 'fine-tuning', 'sft', 'rl', 'reward', 'ppo', 'dpo']
    inference_keywords = ['chain', 'tree', 'search', 'beam', 'sampling', 'inference']
    prompt_keywords = ['prompt', 'instruction', 'template']
    memory_keywords = ['memory', 'retrieval', 'rag', 'context']
    tool_keywords = ['tool', 'api', 'function', 'code', 'execute']
    multi_keywords = ['multi-agent', 'debate', 'collaboration', 'consensus']
    evolution_keywords = ['evolving', 'evolution', 'genetic', 'mutation', 'meta']

    for kw in training_keywords:
        if kw in method_lower:
            return "training"
    for kw in inference_keywords:
        if kw in method_lower:
            return "inference"
    for kw in prompt_keywords:
        if kw in method_lower:
            return "prompt"
    for kw in memory_keywords:
        if kw in method_lower:
            return "memory"
    for kw in tool_keywords:
        if kw in method_lower:
            return "tool"
    for kw in multi_keywords:
        if kw in method_lower:
            return "multi-agent"
    for kw in evolution_keywords:
        if kw in method_lower:
            return "evolution"

    return "other"


def main():
    """Extract methods and build connection graph."""
    print("=" * 60)
    print("Extracting methods and building connection graph")
    print("=" * 60)

    # Load parsed papers
    parsed_path = PROCESSED_DIR / "parsed_papers.json"
    if not parsed_path.exists():
        print("No parsed papers found. Run parse_papers.py first.")
        return

    with open(parsed_path) as f:
        papers = json.load(f)

    print(f"Papers loaded: {len(papers)}")

    # Build method graph
    methods, connections = build_method_graph(papers)

    print(f"\nExtracted {len(methods)} methods")
    print(f"Found {len(connections)} method connections")

    # Save methods
    methods_path = PROCESSED_DIR / "methods.json"
    with open(methods_path, 'w') as f:
        json.dump([asdict(m) for m in methods], f, indent=2)
    print(f"Saved methods to {methods_path}")

    # Save connections
    connections_path = PROCESSED_DIR / "method_connections.json"
    with open(connections_path, 'w') as f:
        json.dump([asdict(c) for c in connections], f, indent=2)
    print(f"Saved connections to {connections_path}")

    # Print top methods
    print("\nTop 15 methods by frequency:")
    for m in methods[:15]:
        print(f"  {m.name}: {m.frequency} papers, category={m.category}")

    # Print top connections
    print("\nTop 10 method connections:")
    for c in connections[:10]:
        print(f"  {c.method1} <-> {c.method2}: strength={c.strength:.2f}, papers={c.co_occurrence_count}")

    return methods, connections


if __name__ == "__main__":
    main()
