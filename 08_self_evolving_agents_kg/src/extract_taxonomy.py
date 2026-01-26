#!/usr/bin/env python3
"""
Extract evolution methods using the AUTHORITATIVE taxonomy from:
1. The landscape.png visual taxonomy
2. The Awesome-Self-Evolving-Agents README subsection headers

This ensures we use the official methodology categories, not arbitrary keywords.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict, field

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# =============================================================================
# AUTHORITATIVE TAXONOMY from landscape.png and README
# =============================================================================

TAXONOMY = {
    # -------------------------------------------------------------------------
    # 1. SINGLE-AGENT OPTIMISATION
    # -------------------------------------------------------------------------
    "Single-Agent Optimisation": {
        "1.1 LLM Behaviour Optimisation": {
            "Training-Based": {
                "Supervised Fine-Tuning (SFT)": {
                    "keywords": ["supervised fine-tuning", "sft", "instruction tuning", "fine-tuning"],
                    "desc": "Train on curated instruction-response pairs",
                    "how_evolves": "Learns from expert demonstrations to improve behavior"
                },
                "Reinforcement Learning (RL)": {
                    "keywords": ["reinforcement learning", "rlhf", "ppo", "dpo", "reward model", "self-play", "self-rewarding"],
                    "desc": "Optimize via reward signals",
                    "how_evolves": "Learns from feedback to maximize task performance"
                }
            },
            "Test-Time (Inference)": {
                "Feedback-Based": {
                    "keywords": ["feedback", "verifier", "process reward", "self-consistency"],
                    "desc": "Use verification signals at inference",
                    "how_evolves": "Iteratively improves outputs using execution or model feedback"
                },
                "Search-Based": {
                    "keywords": ["tree of thought", "beam search", "monte carlo", "mcts", "graph of thought", "buffer of thought"],
                    "desc": "Explore multiple reasoning paths",
                    "how_evolves": "Searches solution space for optimal reasoning chains"
                },
                "Reasoning-Based": {
                    "keywords": ["chain of thought", "cot", "reasoning", "self-taught"],
                    "desc": "Structured reasoning at inference",
                    "how_evolves": "Decomposes complex problems into reasoning steps"
                }
            }
        },
        "1.2 Prompt Optimisation": {
            "Edit-Based": {
                "keywords": ["edit-based", "grips", "gps prompt", "tempera"],
                "desc": "Iteratively edit prompts",
                "how_evolves": "Modifies prompts via gradient-free edits"
            },
            "Evolutionary": {
                "keywords": ["evoprompt", "promptbreeder", "evolutionary prompt", "genetic prompt", "gepa"],
                "desc": "Evolve prompts via mutation/crossover",
                "how_evolves": "Population of prompts compete and combine traits"
            },
            "Generative": {
                "keywords": ["automatic prompt engineer", "ape", "opro", "promptagent", "large language models as optimizers"],
                "desc": "LLM generates optimized prompts",
                "how_evolves": "Meta-prompt generates and refines task prompts"
            },
            "Text Gradient-Based": {
                "keywords": ["textgrad", "gradient descent prompt", "semantic backprop", "text gradient"],
                "desc": "Optimize via textual gradients",
                "how_evolves": "Uses natural language feedback as gradient signal"
            }
        },
        "1.3 Memory Optimisation": {
            "keywords": ["memory bank", "agent workflow memory", "gist memory", "long-term memory", "episodic memory", "memorybank", "mem0"],
            "desc": "Store and retrieve experiences",
            "how_evolves": "Accumulates knowledge for future task improvement"
        },
        "1.4 Tool Optimisation": {
            "Training-Based": {
                "SFT for Tools": {
                    "keywords": ["gpt4tools", "toolllm", "tool learning", "tool-use training"],
                    "desc": "Train on tool usage data",
                    "how_evolves": "Learns tool selection and usage patterns"
                },
                "RL for Tools": {
                    "keywords": ["retool", "toolrl", "tool reinforcement"],
                    "desc": "RL for tool selection",
                    "how_evolves": "Optimizes tool use via reward feedback"
                }
            },
            "Inference-Time": {
                "keywords": ["easytool", "tool instruction", "tool play"],
                "desc": "Optimize tool use at inference",
                "how_evolves": "Adapts tool usage without retraining"
            },
            "Tool Creation": {
                "keywords": ["creator", "tool creation", "clova", "tool update"],
                "desc": "Agent creates new tools",
                "how_evolves": "Expands capabilities by generating reusable tools"
            }
        },
        "1.5 Unified Optimisation": {
            "keywords": ["unified", "evoagent", "self-evolving agent", "lifelong learning"],
            "desc": "Holistic agent evolution",
            "how_evolves": "Combines multiple optimisation dimensions"
        }
    },

    # -------------------------------------------------------------------------
    # 2. MULTI-AGENT OPTIMISATION
    # -------------------------------------------------------------------------
    "Multi-Agent Optimisation": {
        "Automatic Construction": {
            "keywords": ["metaagent", "automatic multi-agent", "auto construction"],
            "desc": "Auto-build multi-agent systems",
            "how_evolves": "Discovers optimal agent configurations"
        },
        "MAS Optimisation": {
            "keywords": ["aflow", "metagpt", "autogen", "agentverse", "gptswarm", "dspy", "workflow", "mas-gpt"],
            "desc": "Optimize multi-agent coordination",
            "how_evolves": "Improves agent communication and task distribution"
        },
        "Agent Collaboration": {
            "keywords": ["collaboration", "debate", "consensus", "multi-agent"],
            "desc": "Multiple agents work together",
            "how_evolves": "Diverse perspectives improve outcomes"
        }
    },

    # -------------------------------------------------------------------------
    # 3. DOMAIN-SPECIFIC OPTIMISATION
    # -------------------------------------------------------------------------
    "Domain-Specific Optimisation": {
        "Biomedicine": {
            "keywords": ["medical", "mmedagent", "mdagents", "healthflow", "chemcrow", "chemagent", "molecular", "drug"],
            "desc": "Medical and molecular discovery",
            "how_evolves": "Specializes for healthcare and chemistry tasks"
        },
        "Programming": {
            "keywords": ["code", "agentcoder", "self-debug", "self-edit", "openhands", "swe-bench", "debugging"],
            "desc": "Code generation and debugging",
            "how_evolves": "Iteratively refines code via execution feedback"
        },
        "Scientific Research": {
            "keywords": ["scientific", "piflow", "research agent", "discovery"],
            "desc": "Automated scientific research",
            "how_evolves": "Conducts experiments and analyzes results"
        },
        "Financial & Legal": {
            "keywords": ["financial", "finrobot", "legal", "lawgpt", "fincon"],
            "desc": "Finance and legal reasoning",
            "how_evolves": "Adapts to domain regulations and patterns"
        }
    },

    # -------------------------------------------------------------------------
    # 4. EVALUATION
    # -------------------------------------------------------------------------
    "Evaluation": {
        "keywords": ["benchmark", "evaluation", "webarena", "swe-bench", "openagi", "toolqa"],
        "desc": "Benchmarks and metrics",
        "how_evolves": "Measures evolution quality and progress"
    }
}


@dataclass
class TaxonomyMethod:
    """A method from the authoritative taxonomy."""
    name: str
    category: str  # Top-level category
    subcategory: str  # Sub-category path
    desc: str
    how_evolves: str
    papers: list = field(default_factory=list)
    frequency: int = 0
    years: list = field(default_factory=list)


def extract_year(arxiv_id: str) -> int:
    """Extract year from arXiv ID."""
    if not arxiv_id:
        return 0
    match = re.match(r'(\d{2})(\d{2})\.', arxiv_id)
    if match:
        prefix = int(match.group(1))
        return 1900 + prefix if prefix >= 90 else 2000 + prefix
    return 0


def flatten_taxonomy(tax: dict, path: str = "") -> list[dict]:
    """Flatten nested taxonomy into list of method entries."""
    results = []

    for key, value in tax.items():
        current_path = f"{path} > {key}" if path else key

        if isinstance(value, dict):
            if "keywords" in value:
                # This is a leaf node (actual method)
                results.append({
                    "name": key,
                    "path": current_path,
                    "keywords": value.get("keywords", []),
                    "desc": value.get("desc", ""),
                    "how_evolves": value.get("how_evolves", "")
                })
            else:
                # Recurse into sub-categories
                results.extend(flatten_taxonomy(value, current_path))

    return results


def extract_methods_from_papers(papers: list) -> list[TaxonomyMethod]:
    """Extract methods using authoritative taxonomy."""

    # Flatten taxonomy
    flat_methods = flatten_taxonomy(TAXONOMY)

    # Track results
    method_data = {}

    for method_entry in flat_methods:
        method_name = method_entry["name"]
        keywords = method_entry["keywords"]
        path = method_entry["path"]

        method_data[method_name] = {
            "path": path,
            "desc": method_entry["desc"],
            "how_evolves": method_entry["how_evolves"],
            "papers": [],
            "years": []
        }

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            title = paper.get("title", "").lower()
            text = (paper.get("parsed_text", "") or paper.get("abstract", "")).lower()
            year = extract_year(arxiv_id)

            # Check if any keyword matches
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in title or kw_lower in text:
                    # Extract context sentence
                    context = ""
                    sentences = re.split(r'[.!?]+', text)
                    for sent in sentences:
                        if kw_lower in sent.lower() and len(sent) > 50:
                            context = sent.strip()[:200]
                            break

                    method_data[method_name]["papers"].append({
                        "arxiv_id": arxiv_id,
                        "title": paper.get("title", ""),
                        "year": year,
                        "context": context
                    })
                    method_data[method_name]["years"].append(year)
                    break  # Only count once per paper

    # Convert to TaxonomyMethod objects
    methods = []
    for name, data in method_data.items():
        if data["papers"]:  # Only include methods found in papers
            # Get top-level category from path
            parts = data["path"].split(" > ")
            category = parts[0] if parts else "Other"
            subcategory = " > ".join(parts[1:]) if len(parts) > 1 else ""

            methods.append(TaxonomyMethod(
                name=name,
                category=category,
                subcategory=subcategory,
                desc=data["desc"],
                how_evolves=data["how_evolves"],
                papers=data["papers"],
                frequency=len(data["papers"]),
                years=sorted(set(y for y in data["years"] if y > 0))
            ))

    methods.sort(key=lambda m: m.frequency, reverse=True)
    return methods


def build_connections(methods: list[TaxonomyMethod]) -> list[dict]:
    """Build connections based on co-occurrence in papers."""
    connections = []

    for i, m1 in enumerate(methods):
        m1_papers = {p["arxiv_id"] for p in m1.papers}
        for m2 in methods[i+1:]:
            m2_papers = {p["arxiv_id"] for p in m2.papers}
            common = m1_papers & m2_papers
            if common:
                union = m1_papers | m2_papers
                strength = len(common) / len(union) if union else 0
                connections.append({
                    "method1": m1.name,
                    "method2": m2.name,
                    "co_occurrence": len(common),
                    "papers": list(common),
                    "strength": strength
                })

    connections.sort(key=lambda c: c["strength"], reverse=True)
    return connections


def main():
    """Extract methods using authoritative taxonomy."""
    print("=" * 60)
    print("Extracting methods using AUTHORITATIVE TAXONOMY")
    print("(from landscape.png and README subsections)")
    print("=" * 60)

    # Load papers
    papers_path = PROCESSED_DIR / "parsed_papers.json"
    if not papers_path.exists():
        print("Run parse_papers.py first.")
        return

    with open(papers_path) as f:
        papers = json.load(f)

    print(f"Papers loaded: {len(papers)}")

    # Extract methods
    methods = extract_methods_from_papers(papers)
    connections = build_connections(methods)

    print(f"\nFound {len(methods)} taxonomy methods in papers")
    print(f"Found {len(connections)} connections")

    # Save
    methods_path = PROCESSED_DIR / "taxonomy_methods.json"
    with open(methods_path, 'w') as f:
        json.dump([asdict(m) for m in methods], f, indent=2)

    connections_path = PROCESSED_DIR / "taxonomy_connections.json"
    with open(connections_path, 'w') as f:
        json.dump(connections, f, indent=2)

    print(f"\nSaved to {methods_path}")

    # Summary by top-level category
    print("\n" + "=" * 60)
    print("METHODS BY CATEGORY")
    print("=" * 60)

    by_category = defaultdict(list)
    for m in methods:
        by_category[m.category].append(m)

    for cat in ["Single-Agent Optimisation", "Multi-Agent Optimisation", "Domain-Specific Optimisation", "Evaluation"]:
        if cat in by_category:
            print(f"\n{cat}:")
            for m in by_category[cat]:
                years = f"({min(m.years)}-{max(m.years)})" if m.years else ""
                print(f"  - {m.name}: {m.frequency} papers {years}")
                if m.subcategory:
                    print(f"    Path: {m.subcategory}")

    return methods, connections


if __name__ == "__main__":
    main()
