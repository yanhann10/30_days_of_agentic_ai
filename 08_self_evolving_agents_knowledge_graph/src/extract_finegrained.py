#!/usr/bin/env python3
"""
Extract evolution methods by METHOD TYPE (not domain).
Categories: RL-Based, Feedback-Based, Search-Based, Graph-Based, etc.
Specific methods as subnodes of broader categories.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict, field

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# =============================================================================
# Categories by EVOLUTION METHOD TYPE
# =============================================================================

FINEGRAINED_METHODS = {
    # -------------------------------------------------------------------------
    # RL-BASED EVOLUTION
    # -------------------------------------------------------------------------
    "PPO Fine-Tuning": {
        "keywords": ["ppo", "proximal policy optimization", "policy gradient"],
        "category": "RL-Based",
        "how": "Constrained policy updates based on reward signal",
        "what": "Stable RL training that prevents catastrophic policy changes"
    },
    "Direct Preference Optimization": {
        "keywords": ["dpo", "direct preference optimization", "preference learning"],
        "category": "RL-Based",
        "how": "Directly optimize on preference pairs without reward model",
        "what": "Simpler alternative to RLHF that skips reward modeling"
    },
    "Human Feedback RL": {
        "keywords": ["rlhf", "reinforcement learning from human feedback", "human feedback"],
        "category": "RL-Based",
        "how": "Train reward model on preferences, then optimize policy",
        "what": "Aligns model with human preferences through feedback"
    },
    "Self-Rewarding": {
        "keywords": ["self-rewarding", "self reward", "self-reward"],
        "category": "RL-Based",
        "how": "Model generates its own reward signal for training",
        "what": "Eliminates need for external reward model"
    },
    "Process Reward Model": {
        "keywords": ["process reward", "prm", "step reward", "math-shepherd"],
        "category": "RL-Based",
        "how": "Reward each reasoning step, not just final answer",
        "what": "Dense feedback for better credit assignment"
    },
    "Self-Play RL": {
        "keywords": ["self-play", "self play", "playing against itself"],
        "category": "RL-Based",
        "how": "Agent plays both sides, learns from game outcomes",
        "what": "Competitive self-interaction drives improvement"
    },
    "In-Context Learning": {
        "keywords": ["in-context learning", "icl", "few-shot", "in context"],
        "category": "RL-Based",
        "how": "Learn from examples in prompt without weight updates",
        "what": "Adapts to new tasks via demonstration examples"
    },

    # -------------------------------------------------------------------------
    # FEEDBACK-BASED EVOLUTION
    # -------------------------------------------------------------------------
    "Self-Refine": {
        "keywords": ["self-refine", "self refine", "iterative refinement"],
        "category": "Feedback-Based",
        "how": "Generate -> Critique -> Refine loop until quality met",
        "what": "LLM improves its own output through self-critique"
    },
    "Self-Correction": {
        "keywords": ["self-correction", "self correct", "self-correcting"],
        "category": "Feedback-Based",
        "how": "Internal verifier detects errors, triggers fixes",
        "what": "Agent recognizes and fixes its own mistakes"
    },
    "Self-Debug": {
        "keywords": ["self-debug", "self debug", "debugging itself"],
        "category": "Feedback-Based",
        "how": "Execute code -> Read errors -> Fix bugs iteratively",
        "what": "Uses execution feedback to correct code"
    },
    "Self-Edit": {
        "keywords": ["self-edit", "self edit", "fault-aware"],
        "category": "Feedback-Based",
        "how": "Identify fault locations -> Apply targeted edits",
        "what": "Locates errors and makes minimal corrections"
    },
    "Reflexion": {
        "keywords": ["reflexion", "self-reflection", "reflect on failure"],
        "category": "Feedback-Based",
        "how": "Analyze failures -> Store lessons -> Apply next time",
        "what": "Explicit reflection extracts insights from failures"
    },
    "Critique-Revise": {
        "keywords": ["critique", "constitutional", "revise"],
        "category": "Feedback-Based",
        "how": "Generate critique of output -> Revise based on critique",
        "what": "Structured feedback loop for quality improvement"
    },

    # -------------------------------------------------------------------------
    # SEARCH-BASED EVOLUTION
    # -------------------------------------------------------------------------
    "Tree-of-Thought": {
        "keywords": ["tree of thought", "tot", "thought tree"],
        "category": "Search-Based",
        "how": "Branch into paths -> Evaluate -> Prune bad branches",
        "what": "Explores reasoning as tree with backtracking"
    },
    "Monte Carlo Search": {
        "keywords": ["mcts", "monte carlo tree search", "monte carlo"],
        "category": "Search-Based",
        "how": "Random rollouts -> Backpropagate values -> Guide search",
        "what": "Balances exploration and exploitation in reasoning"
    },
    "Beam Search": {
        "keywords": ["beam search", "beam decoding"],
        "category": "Search-Based",
        "how": "Maintain top-k candidates at each step -> Expand best",
        "what": "Higher quality than greedy decoding"
    },
    "Best-of-N Sampling": {
        "keywords": ["best-of-n", "best of n", "sample and select"],
        "category": "Search-Based",
        "how": "Generate N candidates -> Score all -> Return best",
        "what": "Simple quality improvement via overgeneration"
    },
    "Self-Consistency": {
        "keywords": ["self-consistency", "self consistency", "majority voting"],
        "category": "Search-Based",
        "how": "Sample multiple solutions -> Aggregate via voting",
        "what": "Reduces variance by selecting most consistent answer"
    },
    "Graph-of-Thought": {
        "keywords": ["graph of thought", "got", "graph reasoning"],
        "category": "Search-Based",
        "how": "Reasoning as graph with arbitrary connections",
        "what": "More flexible than tree structure"
    },
    "Chain-of-Thought": {
        "keywords": ["chain of thought", "chain-of-thought", "cot", "step-by-step reasoning"],
        "category": "Search-Based",
        "how": "Decompose problem into sequential reasoning steps",
        "what": "Explicit reasoning chain improves complex task performance"
    },

    # -------------------------------------------------------------------------
    # GRAPH-BASED METHODS
    # -------------------------------------------------------------------------
    "Graph Reader": {
        "keywords": ["graphreader", "graph reader", "graph-based agent", "graph based agent"],
        "category": "Graph-Based",
        "how": "Build a graph from long context and reason over it",
        "what": "Improves long-context understanding via structured graph memory"
    },
    "Graph Planner": {
        "keywords": ["graph planner", "graph planning", "planning graph"],
        "category": "Graph-Based",
        "how": "Plan actions on a graph of states and transitions",
        "what": "Improves multi-step planning with explicit structure"
    },
    "Graph Agents": {
        "keywords": ["graph agent", "graph-based agents", "graph based agents"],
        "category": "Graph-Based",
        "how": "Agents coordinate through a shared graph representation",
        "what": "Enables structured collaboration and routing"
    },
    "Knowledge Graph Reasoning": {
        "keywords": ["knowledge graph", "kg reasoning", "graph reasoning"],
        "category": "Graph-Based",
        "how": "Use explicit entity-relation graphs for inference",
        "what": "Enables multi-hop reasoning and structured retrieval"
    },
    "Graph-Structured Memory": {
        "keywords": ["graph memory", "memory graph", "graph-structured memory"],
        "category": "Graph-Based",
        "how": "Store memories as a graph of entities and links",
        "what": "Supports relational recall and compositional updates"
    },

    # -------------------------------------------------------------------------
    # EVOLUTIONARY PROMPT OPTIMIZATION
    # -------------------------------------------------------------------------
    "Prompt Breeder": {
        "keywords": ["promptbreeder", "prompt breeder"],
        "category": "Evolutionary Prompt",
        "how": "Self-referential mutation of prompts and mutation operators",
        "what": "Meta-evolution of both prompts and evolution strategies"
    },
    "Evo Prompt": {
        "keywords": ["evoprompt", "evo prompt", "evolutionary prompt"],
        "category": "Evolutionary Prompt",
        "how": "Population of prompts -> Mutation/Crossover -> Selection",
        "what": "Genetic algorithm operators for prompt optimization"
    },
    "Genetic Prompt Search": {
        "keywords": ["gps", "genetic prompt search"],
        "category": "Evolutionary Prompt",
        "how": "Genetic search over discrete prompt space",
        "what": "Few-shot learning via evolved prompts"
    },
    "Prompt Optimization via LLM": {
        "keywords": ["opro", "optimization by prompting", "llm as optimizer"],
        "category": "Evolutionary Prompt",
        "how": "LLM generates and evaluates prompt candidates",
        "what": "Uses LLM itself to optimize prompts"
    },
    "Auto Prompt Engineer": {
        "keywords": ["ape", "automatic prompt engineer"],
        "category": "Evolutionary Prompt",
        "how": "Generate prompt candidates -> Evaluate -> Select best",
        "what": "Automated prompt engineering without humans"
    },
    "Edit-Based Prompt Search": {
        "keywords": ["grips", "gradient-free", "edit-based instruction"],
        "category": "Evolutionary Prompt",
        "how": "Iterative edits based on task feedback",
        "what": "Gradient-free prompt optimization via edits"
    },

    # -------------------------------------------------------------------------
    # GRADIENT-BASED PROMPT (Text Gradients)
    # -------------------------------------------------------------------------
    "Text Gradients": {
        "keywords": ["textgrad", "text gradient", "textual gradient"],
        "category": "Text Gradient",
        "how": "Natural language feedback as gradient signal",
        "what": "Backpropagates textual critiques to improve prompts"
    },
    "Semantic Backprop": {
        "keywords": ["semantic backprop", "semantic backpropagation"],
        "category": "Text Gradient",
        "how": "Propagate semantic feedback through computation graph",
        "what": "Gradient descent analog for language systems"
    },
    "Gradient Summaries": {
        "keywords": ["grad-sum", "gradient summarization"],
        "category": "Text Gradient",
        "how": "Summarize gradients from multiple examples",
        "what": "Efficient prompt optimization via gradient aggregation"
    },

    # -------------------------------------------------------------------------
    # BOOTSTRAPPING / SELF-TRAINING
    # -------------------------------------------------------------------------
    "Self-Taught Reasoner": {
        "keywords": ["star", "self-taught reasoner", "bootstrapping reasoning"],
        "category": "Bootstrapping",
        "how": "Generate rationales -> Filter correct -> Fine-tune on them",
        "what": "Uses own correct solutions as training data"
    },
    "Curriculum Learning": {
        "keywords": ["curriculum learning", "curriculum", "progressive training"],
        "category": "Bootstrapping",
        "how": "Order training from easy to hard tasks progressively",
        "what": "Structured learning progression improves generalization"
    },
    "Knowledge Distillation": {
        "keywords": ["distillation", "knowledge distillation", "teacher-student"],
        "category": "Bootstrapping",
        "how": "Train smaller model to mimic larger teacher model",
        "what": "Transfers capabilities to more efficient models"
    },
    "Self-Instruct": {
        "keywords": ["self-instruct", "self instruct", "instruction generation"],
        "category": "Bootstrapping",
        "how": "Bootstrap from seeds -> Generate instructions -> Train",
        "what": "Creates instruction data from minimal examples"
    },
    "Synthetic Data Gen": {
        "keywords": ["synthetic data", "self-generated data"],
        "category": "Bootstrapping",
        "how": "LLM generates examples -> Filter -> Train on them",
        "what": "Creates training corpus without human annotation"
    },
    "Reward-Filtered Self-Training": {
        "keywords": ["rest", "reinforced self-training"],
        "category": "Bootstrapping",
        "how": "Generate -> Filter by reward -> Fine-tune iteratively",
        "what": "Combines self-training with reward filtering"
    },

    # -------------------------------------------------------------------------
    # TOOL EVOLUTION
    # -------------------------------------------------------------------------
    "Tool Creation": {
        "keywords": ["tool creation", "create tool", "creator"],
        "category": "Tool Evolution",
        "how": "Agent writes code to create new tools",
        "what": "Expands capability by generating tool functions"
    },
    "Tool Learning": {
        "keywords": ["tool learning", "gpt4tools", "toolllm"],
        "category": "Tool Evolution",
        "how": "Fine-tune on tool-use demonstrations",
        "what": "Learns tool selection and invocation"
    },
    "Tool-Use RL": {
        "keywords": ["toolrl", "tool reinforcement", "retool"],
        "category": "Tool Evolution",
        "how": "Reward for successful tool use -> Learn when/how",
        "what": "RL-based tool selection optimization"
    },
    "Tool Discovery": {
        "keywords": ["tool discovery", "toolchain", "tool-planner"],
        "category": "Tool Evolution",
        "how": "Search available tools -> Plan sequence -> Execute",
        "what": "Dynamically finds and chains tools"
    },

    # -------------------------------------------------------------------------
    # MEMORY EVOLUTION
    # -------------------------------------------------------------------------
    "Workflow Memory": {
        "keywords": ["workflow memory", "agent workflow", "awm"],
        "category": "Memory Evolution",
        "how": "Store successful action sequences -> Retrieve for similar tasks",
        "what": "Remembers proven procedures"
    },
    "Memory Bank": {
        "keywords": ["memory bank", "memorybank"],
        "category": "Memory Evolution",
        "how": "Store and retrieve long-term experiences",
        "what": "Persistent memory for personalized agents"
    },
    "Vector Memory": {
        "keywords": ["vector memory", "vector database", "embedding store"],
        "category": "Memory Evolution",
        "how": "Store memories as vector embeddings for similarity search",
        "what": "Enables semantic recall at scale"
    },
    "Episodic Memory": {
        "keywords": ["episodic memory"],
        "category": "Memory Evolution",
        "how": "Store episodes with outcomes -> Query and apply lessons",
        "what": "Long-term storage of experiences"
    },
    "Skill Library": {
        "keywords": ["skill library", "skill repository", "voyager"],
        "category": "Memory Evolution",
        "how": "Extract skills from successes -> Store -> Compose for new tasks",
        "what": "Accumulates modular capabilities"
    },
    "Gist Memory": {
        "keywords": ["gist memory", "gist"],
        "category": "Memory Evolution",
        "how": "Store compressed summaries of long context",
        "what": "Keeps key facts while shrinking memory footprint"
    },
    "Compressive Memory": {
        "keywords": ["compressive memory", "compress"],
        "category": "Memory Evolution",
        "how": "Compress long context into retrievable representations",
        "what": "Enables long-context understanding"
    },
    "Retrieval Memory": {
        "keywords": ["retrieval memory", "retrieval-augmented", "rag"],
        "category": "Memory Evolution",
        "how": "Retrieve relevant memories on demand for each task",
        "what": "Improves accuracy with context grounding"
    },
    "Long-Term Memory": {
        "keywords": ["long-term memory", "long term memory", "ltm"],
        "category": "Memory Evolution",
        "how": "Persist knowledge across sessions and tasks",
        "what": "Supports personalization and continuity"
    },
    "Working Memory": {
        "keywords": ["working memory", "short-term memory", "short term memory"],
        "category": "Memory Evolution",
        "how": "Keep task-relevant facts in a small active buffer",
        "what": "Improves focus and reduces context overload"
    },
    "Agent Memory": {
        "keywords": ["agent memory", "memory module"],
        "category": "Memory Evolution",
        "how": "Dedicated memory module for agentic workflows",
        "what": "Improves consistency across long-running tasks"
    },

    # -------------------------------------------------------------------------
    # MULTI-AGENT EVOLUTION
    # -------------------------------------------------------------------------
    "Multi-Agent Debate": {
        "keywords": ["debate", "multi-agent debate", "agents debate"],
        "category": "Multi-Agent",
        "how": "Agents argue positions -> Refine -> Converge",
        "what": "Adversarial discussion improves reasoning"
    },
    "Agent Collaboration": {
        "keywords": ["collaboration", "collaborative agents", "agentverse"],
        "category": "Multi-Agent",
        "how": "Specialized agents handle subtasks -> Combine outputs",
        "what": "Division of labor for complex tasks"
    },
    "Workflow Optimization": {
        "keywords": ["aflow", "metagpt", "autogen", "workflow"],
        "category": "Multi-Agent",
        "how": "Evolve agent communication patterns and routing",
        "what": "Optimizes how agents coordinate"
    },
    "Multi-Agent Architecture Search": {
        "keywords": ["adas", "architecture search", "mas design"],
        "category": "Multi-Agent",
        "how": "Automatically design multi-agent system topology",
        "what": "Learns optimal agent configurations"
    },
    "Symbolic Learning": {
        "keywords": ["symbolic learning", "symbolic rules"],
        "category": "Multi-Agent",
        "how": "Agents learn symbolic rules from interactions",
        "what": "Enables evolution beyond training distribution"
    },
}


@dataclass
class FineGrainedMethod:
    """A fine-grained evolution method."""
    name: str
    category: str
    how: str
    what: str
    papers: list = field(default_factory=list)
    frequency: int = 0
    years: list = field(default_factory=list)


def extract_year(arxiv_id: str) -> int:
    if not arxiv_id:
        return 0
    match = re.match(r'(\d{2})(\d{2})\.', arxiv_id)
    if match:
        prefix = int(match.group(1))
        return 1900 + prefix if prefix >= 90 else 2000 + prefix
    return 0


def extract_context(text: str, keywords: list) -> str:
    """Extract a sentence showing how the method is used."""
    sentences = re.split(r'[.!?]+', text)
    for kw in keywords:
        for sent in sentences:
            if kw.lower() in sent.lower() and len(sent) > 40:
                return sent.strip()[:180]
    return ""


def extract_methods(papers: list) -> list[FineGrainedMethod]:
    """Extract fine-grained methods from papers."""
    method_data = {name: {"papers": [], "years": []} for name in FINEGRAINED_METHODS}

    for paper in papers:
        arxiv_id = paper.get("arxiv_id", "")
        title = paper.get("title", "").lower()
        text = (paper.get("parsed_text", "") or paper.get("abstract", "") or "").lower()
        year = extract_year(arxiv_id)

        for method_name, info in FINEGRAINED_METHODS.items():
            keywords = info["keywords"]
            for kw in keywords:
                if kw.lower() in title or kw.lower() in text:
                    count = text.count(kw.lower()) + title.count(kw.lower())
                    if count >= 1:  # Lower threshold for title matches
                        context = extract_context(
                            paper.get("parsed_text", "") or paper.get("abstract", "") or title,
                            keywords
                        )
                        method_data[method_name]["papers"].append({
                            "arxiv_id": arxiv_id,
                            "title": paper.get("title", ""),
                            "year": year,
                            "context": context
                        })
                        method_data[method_name]["years"].append(year)
                        break

    methods = []
    for name, data in method_data.items():
        if data["papers"]:
            info = FINEGRAINED_METHODS[name]
            methods.append(FineGrainedMethod(
                name=name,
                category=info["category"],
                how=info["how"],
                what=info["what"],
                papers=data["papers"],
                frequency=len(data["papers"]),
                years=sorted(set(y for y in data["years"] if y > 0))
            ))

    methods.sort(key=lambda m: m.frequency, reverse=True)
    return methods


def build_connections(methods: list[FineGrainedMethod]) -> list[dict]:
    """Build connections based on co-occurrence."""
    connections = []
    for i, m1 in enumerate(methods):
        m1_papers = {
            (p.get("arxiv_id") or p.get("title", "")).strip().lower()
            for p in m1.papers
            if (p.get("arxiv_id") or p.get("title"))
        }
        for m2 in methods[i+1:]:
            m2_papers = {
                (p.get("arxiv_id") or p.get("title", "")).strip().lower()
                for p in m2.papers
                if (p.get("arxiv_id") or p.get("title"))
            }
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
    print("=" * 60)
    print("Extracting evolution methods by METHOD TYPE")
    print("(RL-Based, Feedback-Based, Search-Based, Evolutionary Prompt, etc.)")
    print("=" * 60)

    # Try parsed_papers.json first, fall back to papers.json
    papers_path = PROCESSED_DIR / "parsed_papers.json"
    fallback_path = PROCESSED_DIR / "papers.json"

    papers = []
    if papers_path.exists():
        with open(papers_path) as f:
            papers = json.load(f)

    if not papers and fallback_path.exists():
        print("No parsed papers, using papers.json metadata")
        with open(fallback_path) as f:
            papers = json.load(f)

    if not papers:
        print("No papers found.")
        return

    print(f"Papers loaded: {len(papers)}")

    methods = extract_methods(papers)
    connections = build_connections(methods)

    print(f"\nFound {len(methods)} methods")
    print(f"Found {len(connections)} connections")

    methods_path = PROCESSED_DIR / "finegrained_methods.json"
    with open(methods_path, 'w') as f:
        json.dump([asdict(m) for m in methods], f, indent=2)

    connections_path = PROCESSED_DIR / "finegrained_connections.json"
    with open(connections_path, 'w') as f:
        json.dump(connections, f, indent=2)

    print(f"\nSaved to {methods_path}")

    print("\n" + "=" * 60)
    print("METHODS BY CATEGORY")
    print("=" * 60)

    by_cat = defaultdict(list)
    for m in methods:
        by_cat[m.category].append(m)

    for cat in sorted(by_cat.keys()):
        print(f"\n{cat}:")
        for m in sorted(by_cat[cat], key=lambda x: x.frequency, reverse=True):
            years = f"({min(m.years)}-{max(m.years)})" if m.years else ""
            print(f"  - {m.name}: {m.frequency} papers {years}")

    return methods, connections


if __name__ == "__main__":
    main()
