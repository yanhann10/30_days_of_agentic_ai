#!/usr/bin/env python3
"""
Extract concepts, mechanisms, and benchmarks from paper abstracts using LLM.

Optionally integrates multi-agent debate for concept validation.

Output:
- data/processed/concepts.json
- data/processed/paper_concepts.json
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Add debate framework to path if available
DEBATE_FRAMEWORK = Path(__file__).parent.parent.parent / "07_multi_agent_debates"
if DEBATE_FRAMEWORK.exists():
    sys.path.insert(0, str(DEBATE_FRAMEWORK))
    DEBATE_AVAILABLE = True
else:
    DEBATE_AVAILABLE = False


@dataclass
class Concept:
    """A technical concept extracted from papers."""
    name: str
    normalized_name: str  # Canonical form
    category: str  # mechanism, architecture, capability, benchmark, evaluation
    definition: Optional[str] = None
    first_paper_id: Optional[str] = None
    mention_count: int = 0
    aliases: list[str] = field(default_factory=list)


@dataclass
class PaperConcepts:
    """Concepts associated with a paper."""
    arxiv_id: str
    title: str
    concepts: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)
    introduces: list[str] = field(default_factory=list)  # Concepts this paper introduces


# Concept normalization mapping
CONCEPT_ALIASES = {
    "cot": "Chain-of-Thought",
    "chain of thought": "Chain-of-Thought",
    "chain-of-thought": "Chain-of-Thought",
    "rag": "Retrieval-Augmented Generation",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "rlhf": "RLHF",
    "reinforcement learning from human feedback": "RLHF",
    "sft": "Supervised Fine-Tuning",
    "supervised fine-tuning": "Supervised Fine-Tuning",
    "dpo": "Direct Preference Optimization",
    "direct preference optimization": "Direct Preference Optimization",
    "ppo": "PPO",
    "proximal policy optimization": "PPO",
    "llm": "Large Language Model",
    "large language model": "Large Language Model",
    "mcts": "Monte Carlo Tree Search",
    "monte carlo tree search": "Monte Carlo Tree Search",
    "mas": "Multi-Agent System",
    "multi-agent system": "Multi-Agent System",
}


def normalize_concept(name: str) -> str:
    """Normalize concept name to canonical form."""
    lower = name.lower().strip()
    if lower in CONCEPT_ALIASES:
        return CONCEPT_ALIASES[lower]
    # Title case for new concepts
    return name.strip().title()


def get_anthropic_client():
    """Get Anthropic client for Claude API."""
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        print("anthropic package not installed. Install with: pip install anthropic")
        return None


def extract_concepts_with_claude(abstracts: list[dict], batch_size: int = 5) -> list[dict]:
    """Extract concepts from abstracts using Claude API."""
    client = get_anthropic_client()
    if not client:
        return []

    all_results = []

    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1}/{(len(abstracts) + batch_size - 1) // batch_size}")

        # Format batch for prompt
        batch_text = ""
        for j, paper in enumerate(batch):
            batch_text += f"\n---\nPaper {j + 1}: {paper['title']}\n"
            batch_text += f"arXiv ID: {paper['arxiv_id']}\n"
            batch_text += f"Abstract: {paper.get('abstract', 'N/A')}\n"

        prompt = f"""Analyze these research papers about AI agents and self-evolving systems.
For each paper, extract:

1. **Key Concepts**: Technical concepts, methods, or frameworks mentioned (e.g., "Chain-of-Thought", "Self-Reflection", "Tool Learning")
2. **Mechanisms**: Specific techniques or algorithms used (e.g., "reinforcement learning", "evolutionary optimization", "prompt mutation")
3. **Benchmarks**: Evaluation benchmarks mentioned (e.g., "AgentBench", "SWE-bench", "HumanEval")
4. **Introduces**: New concepts or methods this paper introduces (if any)

Papers to analyze:
{batch_text}

Respond in JSON format:
```json
[
  {{
    "arxiv_id": "...",
    "concepts": ["concept1", "concept2"],
    "mechanisms": ["mechanism1"],
    "benchmarks": ["benchmark1"],
    "introduces": ["new_concept"]
  }},
  ...
]
```

Be specific and use standard terminology. Only include concepts actually mentioned in the abstract."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON from response
            text = response.content[0].text
            # Extract JSON from markdown code block if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            results = json.loads(text)
            all_results.extend(results)

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Return empty results for failed batch
            for paper in batch:
                all_results.append({
                    "arxiv_id": paper['arxiv_id'],
                    "concepts": [],
                    "mechanisms": [],
                    "benchmarks": [],
                    "introduces": [],
                    "error": str(e)
                })

    return all_results


def run_debate_validation(concepts: list[Concept], top_n: int = 10) -> dict:
    """Use multi-agent debate to validate concept categorization."""
    if not DEBATE_AVAILABLE:
        print("Multi-agent debate framework not available")
        return {}

    try:
        from debate_graph import run_debate

        # Focus on most frequent concepts for validation
        top_concepts = sorted(concepts, key=lambda c: c.mention_count, reverse=True)[:top_n]

        concept_summary = "\n".join([
            f"- {c.name} ({c.category}): mentioned {c.mention_count} times"
            for c in top_concepts
        ])

        topic = "How should we categorize and organize these technical concepts for a knowledge graph?"
        doc_content = f"""
        Extracted concepts from self-evolving agents research:

        {concept_summary}

        Categories used:
        - mechanism: Techniques or algorithms
        - architecture: System designs
        - capability: Agent abilities
        - benchmark: Evaluation datasets
        - evaluation: Assessment methods
        """

        result = run_debate(doc_content, topic)
        return result

    except Exception as e:
        print(f"Debate validation failed: {e}")
        return {}


def build_concept_graph(extraction_results: list[dict]) -> tuple[list[Concept], list[PaperConcepts]]:
    """Build concept entities from extraction results."""
    concept_map = {}  # normalized_name -> Concept
    paper_concepts = []

    for result in extraction_results:
        arxiv_id = result.get('arxiv_id')
        if not arxiv_id:
            continue

        pc = PaperConcepts(
            arxiv_id=arxiv_id,
            title=result.get('title', ''),
            concepts=[],
            mechanisms=[],
            benchmarks=[],
            introduces=[]
        )

        # Process concepts
        for concept_name in result.get('concepts', []):
            normalized = normalize_concept(concept_name)
            pc.concepts.append(normalized)

            if normalized not in concept_map:
                concept_map[normalized] = Concept(
                    name=concept_name,
                    normalized_name=normalized,
                    category="concept",
                    first_paper_id=arxiv_id,
                    mention_count=1
                )
            else:
                concept_map[normalized].mention_count += 1
                if concept_name.lower() != normalized.lower():
                    concept_map[normalized].aliases.append(concept_name)

        # Process mechanisms
        for mech_name in result.get('mechanisms', []):
            normalized = normalize_concept(mech_name)
            pc.mechanisms.append(normalized)

            if normalized not in concept_map:
                concept_map[normalized] = Concept(
                    name=mech_name,
                    normalized_name=normalized,
                    category="mechanism",
                    first_paper_id=arxiv_id,
                    mention_count=1
                )
            else:
                concept_map[normalized].mention_count += 1

        # Process benchmarks
        for bench_name in result.get('benchmarks', []):
            normalized = normalize_concept(bench_name)
            pc.benchmarks.append(normalized)

            if normalized not in concept_map:
                concept_map[normalized] = Concept(
                    name=bench_name,
                    normalized_name=normalized,
                    category="benchmark",
                    first_paper_id=arxiv_id,
                    mention_count=1
                )
            else:
                concept_map[normalized].mention_count += 1

        # Process introduces
        for intro_name in result.get('introduces', []):
            normalized = normalize_concept(intro_name)
            pc.introduces.append(normalized)

            if normalized in concept_map:
                concept_map[normalized].first_paper_id = arxiv_id

        paper_concepts.append(pc)

    return list(concept_map.values()), paper_concepts


def load_enriched_papers() -> list[dict]:
    """Load enriched papers with abstracts."""
    path = PROCESSED_DIR / "enriched_papers.json"
    if not path.exists():
        print(f"Enriched papers not found at {path}")
        print("Run enrich_papers.py first")
        return []
    with open(path) as f:
        return json.load(f)


def save_concepts(concepts: list[Concept], paper_concepts: list[PaperConcepts]):
    """Save extracted concepts to JSON."""
    concepts_path = PROCESSED_DIR / "concepts.json"
    with open(concepts_path, 'w') as f:
        json.dump([asdict(c) for c in concepts], f, indent=2)
    print(f"Saved {len(concepts)} concepts to {concepts_path}")

    paper_concepts_path = PROCESSED_DIR / "paper_concepts.json"
    with open(paper_concepts_path, 'w') as f:
        json.dump([asdict(pc) for pc in paper_concepts], f, indent=2)
    print(f"Saved paper-concept mappings to {paper_concepts_path}")


def main(use_debate: bool = False, limit: int = None):
    """Main entry point."""
    print("=" * 60)
    print("Extracting concepts from paper abstracts")
    print("=" * 60)

    papers = load_enriched_papers()
    if not papers:
        return

    # Filter to papers with abstracts
    papers_with_abstracts = [p for p in papers if p.get('abstract')]
    print(f"Papers with abstracts: {len(papers_with_abstracts)}/{len(papers)}")

    if limit:
        papers_with_abstracts = papers_with_abstracts[:limit]
        print(f"Processing first {limit} papers only")

    # Extract concepts using Claude
    print("\nExtracting concepts with Claude API...")
    extraction_results = extract_concepts_with_claude(papers_with_abstracts)

    # Build concept graph
    concepts, paper_concepts = build_concept_graph(extraction_results)

    print(f"\nExtracted {len(concepts)} unique concepts")
    print(f"Mapped concepts for {len(paper_concepts)} papers")

    # Optional: validate with multi-agent debate
    if use_debate and DEBATE_AVAILABLE:
        print("\nRunning multi-agent debate for concept validation...")
        debate_result = run_debate_validation(concepts)
        if debate_result:
            print(f"Debate novelty score: {debate_result.get('novelty_score', 'N/A')}")

    # Save results
    save_concepts(concepts, paper_concepts)

    # Print top concepts
    print("\nTop 20 concepts by mention count:")
    sorted_concepts = sorted(concepts, key=lambda c: c.mention_count, reverse=True)
    for c in sorted_concepts[:20]:
        print(f"  {c.normalized_name}: {c.mention_count} ({c.category})")

    return concepts, paper_concepts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract concepts from papers")
    parser.add_argument("--debate", action="store_true", help="Use multi-agent debate for validation")
    parser.add_argument("--limit", type=int, help="Limit number of papers")
    args = parser.parse_args()

    main(use_debate=args.debate, limit=args.limit)
