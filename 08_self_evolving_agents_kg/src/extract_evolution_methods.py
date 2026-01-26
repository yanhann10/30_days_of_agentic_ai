#!/usr/bin/env python3
"""
Extract EVOLUTION methods from papers.
Focuses specifically on HOW agents evolve, not general capabilities.

Distinguishes:
- Evolution MECHANISMS (how): genetic algorithm, RLHF, self-play, etc.
- Evolution GOALS (what): self-improvement, adaptation (filtered out as not methods)
- Evaluation methods: how evolution quality is measured
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict, field

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Evolution MECHANISMS - actual methods that enable evolution
# Preserve original casing for display
EVOLUTION_MECHANISMS = {
    # Evolutionary/Genetic approaches
    "genetic algorithm": {"display": "Genetic Algorithm", "category": "evolutionary"},
    "evolutionary algorithm": {"display": "Evolutionary Algorithm", "category": "evolutionary"},
    "mutation": {"display": "Mutation", "category": "evolutionary"},
    "crossover": {"display": "Crossover", "category": "evolutionary"},
    "selection pressure": {"display": "Selection Pressure", "category": "evolutionary"},
    "fitness function": {"display": "Fitness Function", "category": "evolutionary"},
    "population-based": {"display": "Population-Based", "category": "evolutionary"},

    # RL-based evolution
    "rlhf": {"display": "RLHF", "category": "rl-evolution"},
    "reinforcement learning from human feedback": {"display": "RLHF", "category": "rl-evolution"},
    "ppo": {"display": "PPO", "category": "rl-evolution"},
    "proximal policy optimization": {"display": "PPO", "category": "rl-evolution"},
    "dpo": {"display": "DPO", "category": "rl-evolution"},
    "direct preference optimization": {"display": "DPO", "category": "rl-evolution"},
    "reward model": {"display": "Reward Model", "category": "rl-evolution"},
    "reward shaping": {"display": "Reward Shaping", "category": "rl-evolution"},
    "self-play": {"display": "Self-Play", "category": "rl-evolution"},

    # Self-improvement loops
    "self-refine": {"display": "Self-Refine", "category": "self-improvement"},
    "self-reflection": {"display": "Self-Reflection", "category": "self-improvement"},
    "iterative refinement": {"display": "Iterative Refinement", "category": "self-improvement"},
    "self-critique": {"display": "Self-Critique", "category": "self-improvement"},
    "self-debug": {"display": "Self-Debug", "category": "self-improvement"},
    "self-edit": {"display": "Self-Edit", "category": "self-improvement"},
    "self-correction": {"display": "Self-Correction", "category": "self-improvement"},

    # Bootstrapping/Self-training
    "bootstrapping": {"display": "Bootstrapping", "category": "self-training"},
    "self-training": {"display": "Self-Training", "category": "self-training"},
    "self-instruct": {"display": "Self-Instruct", "category": "self-training"},
    "synthetic data": {"display": "Synthetic Data Generation", "category": "self-training"},
    "data augmentation": {"display": "Data Augmentation", "category": "self-training"},

    # Prompt evolution
    "prompt evolution": {"display": "Prompt Evolution", "category": "prompt-evolution"},
    "automatic prompt": {"display": "Automatic Prompt Engineering", "category": "prompt-evolution"},
    "prompt optimization": {"display": "Prompt Optimization", "category": "prompt-evolution"},
    "meta-prompt": {"display": "Meta-Prompt", "category": "prompt-evolution"},
    "promptbreeder": {"display": "PromptBreeder", "category": "prompt-evolution"},
    "evoprompt": {"display": "EvoPrompt", "category": "prompt-evolution"},

    # Experience/Memory-based evolution
    "experience replay": {"display": "Experience Replay", "category": "experience-based"},
    "episodic memory": {"display": "Episodic Memory", "category": "experience-based"},
    "workflow memory": {"display": "Workflow Memory", "category": "experience-based"},
    "skill library": {"display": "Skill Library", "category": "experience-based"},
    "tool creation": {"display": "Tool Creation", "category": "experience-based"},

    # Meta-learning
    "meta-learning": {"display": "Meta-Learning", "category": "meta-learning"},
    "learning to learn": {"display": "Learning to Learn", "category": "meta-learning"},
    "few-shot adaptation": {"display": "Few-Shot Adaptation", "category": "meta-learning"},
    "curriculum learning": {"display": "Curriculum Learning", "category": "meta-learning"},

    # Multi-agent evolution
    "debate": {"display": "Multi-Agent Debate", "category": "multi-agent"},
    "agent collaboration": {"display": "Agent Collaboration", "category": "multi-agent"},
    "consensus": {"display": "Consensus Mechanism", "category": "multi-agent"},

    # Architecture evolution
    "neural architecture search": {"display": "Neural Architecture Search", "category": "architecture"},
    "automl": {"display": "AutoML", "category": "architecture"},
}

# Evaluation methods for evolution quality
EVALUATION_METHODS = {
    "pass@k": {"display": "Pass@k", "desc": "Success rate over k attempts"},
    "self-consistency": {"display": "Self-Consistency", "desc": "Agreement across multiple samples"},
    "human evaluation": {"display": "Human Evaluation", "desc": "Human judges rate quality"},
    "automatic evaluation": {"display": "Automatic Evaluation", "desc": "LLM-as-judge or metrics"},
    "benchmark": {"display": "Benchmark Score", "desc": "Standard benchmark performance"},
    "ablation": {"display": "Ablation Study", "desc": "Component contribution analysis"},
    "reward score": {"display": "Reward Score", "desc": "RL reward signal"},
    "win rate": {"display": "Win Rate", "desc": "Pairwise comparison wins"},
}

# Terms that are GOALS not methods (filter these out)
EVOLUTION_GOALS = {
    "self-evolving", "self-improvement", "self-improving", "adaptation",
    "continuous learning", "lifelong learning", "autonomous improvement"
}


@dataclass
class EvolutionMethod:
    """An evolution method extracted from papers."""
    name: str  # lowercase for matching
    display_name: str  # proper casing for display
    category: str
    papers: list = field(default_factory=list)  # list of {arxiv_id, title, year, how_enables}
    frequency: int = 0
    years: list = field(default_factory=list)  # publication years


@dataclass
class MethodConnection:
    """Connection between two evolution methods."""
    method1: str
    method2: str
    co_occurrence_count: int = 0
    papers: list = field(default_factory=list)
    strength: float = 0.0


def extract_year_from_arxiv_id(arxiv_id: str) -> int:
    """Extract year from arXiv ID (e.g., 2309.12345 -> 2023)."""
    if not arxiv_id:
        return 0
    match = re.match(r'(\d{2})(\d{2})\.', arxiv_id)
    if match:
        year_prefix = int(match.group(1))
        if year_prefix >= 90:
            return 1900 + year_prefix
        else:
            return 2000 + year_prefix
    return 0


def extract_how_enables_evolution(text: str, method_name: str) -> str:
    """Extract a sentence describing how this method enables evolution."""
    text_lower = text.lower()
    method_lower = method_name.lower()

    # Find sentences containing the method
    sentences = re.split(r'[.!?]+', text)

    for sent in sentences:
        sent_lower = sent.lower()
        if method_lower in sent_lower:
            # Look for evolution-related context
            evolution_keywords = ['evolv', 'improv', 'adapt', 'learn', 'optim', 'enhanc', 'updat']
            if any(kw in sent_lower for kw in evolution_keywords):
                # Clean and truncate
                clean = ' '.join(sent.split())[:200]
                if len(clean) > 50:  # Only if substantial
                    return clean.strip()

    return ""


def extract_evolution_methods(papers: list) -> tuple[list[EvolutionMethod], list[MethodConnection]]:
    """Extract evolution-specific methods from papers."""

    paper_methods = {}  # arxiv_id -> list of methods found
    method_data = defaultdict(lambda: {"papers": [], "years": []})

    for paper in papers:
        arxiv_id = paper.get('arxiv_id', '')
        text = paper.get('parsed_text', '') or paper.get('abstract', '')
        title = paper.get('title', '')
        year = extract_year_from_arxiv_id(arxiv_id)

        if not text:
            continue

        text_lower = text.lower()
        title_lower = title.lower()
        found_methods = []

        # Check for evolution mechanisms
        for method_key, method_info in EVOLUTION_MECHANISMS.items():
            if method_key in text_lower or method_key in title_lower:
                # Count occurrences - need at least 2 mentions to be significant
                count = text_lower.count(method_key)
                if count >= 2 or method_key in title_lower:
                    display_name = method_info["display"]

                    # Extract how it enables evolution
                    how_enables = extract_how_enables_evolution(text, method_key)

                    method_data[display_name]["papers"].append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "year": year,
                        "how_enables": how_enables
                    })
                    method_data[display_name]["years"].append(year)
                    method_data[display_name]["category"] = method_info["category"]

                    found_methods.append(display_name)

        paper_methods[arxiv_id] = list(set(found_methods))

    # Create EvolutionMethod objects
    methods = []
    for name, data in method_data.items():
        if len(data["papers"]) >= 1:  # At least 1 paper
            methods.append(EvolutionMethod(
                name=name.lower(),
                display_name=name,
                category=data.get("category", "other"),
                papers=data["papers"],
                frequency=len(data["papers"]),
                years=sorted(set(data["years"]))
            ))

    # Sort by frequency
    methods.sort(key=lambda m: m.frequency, reverse=True)

    # Build connections based on co-occurrence
    connections = []
    method_names = [m.display_name for m in methods]

    for i, m1 in enumerate(method_names):
        m1_papers = {p["arxiv_id"] for p in method_data[m1]["papers"]}
        for m2 in method_names[i+1:]:
            m2_papers = {p["arxiv_id"] for p in method_data[m2]["papers"]}
            common = m1_papers & m2_papers
            if common:
                union = m1_papers | m2_papers
                strength = len(common) / len(union) if union else 0
                connections.append(MethodConnection(
                    method1=m1,
                    method2=m2,
                    co_occurrence_count=len(common),
                    papers=list(common),
                    strength=strength
                ))

    connections.sort(key=lambda c: c.strength, reverse=True)

    return methods, connections


def main():
    """Extract evolution methods and build connection graph."""
    print("=" * 60)
    print("Extracting EVOLUTION methods (how agents improve)")
    print("=" * 60)

    # Load parsed papers
    parsed_path = PROCESSED_DIR / "parsed_papers.json"
    if not parsed_path.exists():
        print("No parsed papers found. Run parse_papers.py first.")
        return

    with open(parsed_path) as f:
        papers = json.load(f)

    print(f"Papers loaded: {len(papers)}")

    # Extract evolution methods
    methods, connections = extract_evolution_methods(papers)

    print(f"\nExtracted {len(methods)} evolution methods")
    print(f"Found {len(connections)} method connections")

    # Save methods
    methods_path = PROCESSED_DIR / "evolution_methods.json"
    with open(methods_path, 'w') as f:
        json.dump([asdict(m) for m in methods], f, indent=2)
    print(f"Saved methods to {methods_path}")

    # Save connections
    connections_path = PROCESSED_DIR / "evolution_connections.json"
    with open(connections_path, 'w') as f:
        json.dump([asdict(c) for c in connections], f, indent=2)
    print(f"Saved connections to {connections_path}")

    # Print summary by category
    print("\nMethods by category:")
    by_category = defaultdict(list)
    for m in methods:
        by_category[m.category].append(m)

    for cat, cat_methods in sorted(by_category.items()):
        print(f"\n  {cat}:")
        for m in cat_methods[:5]:
            years_str = f"{min(m.years)}-{max(m.years)}" if m.years else "?"
            print(f"    {m.display_name}: {m.frequency} papers ({years_str})")

    # Print year distribution
    print("\nTemporal distribution:")
    year_counts = defaultdict(int)
    for m in methods:
        for y in m.years:
            if y > 0:
                year_counts[y] += 1
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {'#' * year_counts[year]} ({year_counts[year]})")

    return methods, connections


if __name__ == "__main__":
    main()
