#!/usr/bin/env python3
"""
Extract FINE-GRAINED evolution methods with specific details.
Categories based on GitHub repo section subheaders.
Domain-Specific methods include actual technique subnodes.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict, field

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# =============================================================================
# Categories from GitHub repo section subheaders
# =============================================================================

FINEGRAINED_METHODS = {
    # -------------------------------------------------------------------------
    # 1.1 LLM BEHAVIOUR OPTIMISATION
    # -------------------------------------------------------------------------
    # 1.1.1 Training-Based
    "Supervised Fine-Tuning": {
        "keywords": ["supervised fine-tuning", "sft", "fine-tune on", "instruction tuning"],
        "category": "LLM Behaviour Optimisation",
        "how": "Train model on curated (input, output) pairs to learn desired behavior",
        "what": "Updates model weights to follow instructions and produce better outputs"
    },
    "Reinforcement Learning from Feedback": {
        "keywords": ["rlhf", "reinforcement learning from human feedback", "ppo", "dpo", "self-rewarding"],
        "category": "LLM Behaviour Optimisation",
        "how": "Train reward model on preferences, then optimize policy via RL (PPO/DPO)",
        "what": "Aligns model outputs with human preferences through reward signal"
    },
    "Self-Play Learning": {
        "keywords": ["self-play", "self play", "playing against itself", "competitive self"],
        "category": "LLM Behaviour Optimisation",
        "how": "Agent plays both sides of interaction, learns from game outcomes",
        "what": "Improves through competitive self-interaction without external data"
    },
    "Bootstrapping from Rationales": {
        "keywords": ["star", "bootstrapping reasoning", "bootstrap from rationale", "self-taught reasoner"],
        "category": "LLM Behaviour Optimisation",
        "how": "Generate rationales -> Filter correct ones -> Fine-tune on them",
        "what": "STaR: Uses correctly-solved examples as training data for next iteration"
    },
    # 1.1.2 Test-Time
    "Feedback-Based Refinement": {
        "keywords": ["feedback-based", "self-refine", "iterative refinement", "critique"],
        "category": "LLM Behaviour Optimisation",
        "how": "Generate -> Receive feedback -> Refine until quality threshold met",
        "what": "LLM critiques its own output and produces improved version"
    },
    "Tree-of-Thought Search": {
        "keywords": ["tree of thought", "tot", "thought tree", "deliberate problem"],
        "category": "LLM Behaviour Optimisation",
        "how": "Branch into multiple reasoning paths -> Evaluate -> Prune bad branches",
        "what": "Explores reasoning space as a tree with backtracking"
    },
    "Self-Consistency Decoding": {
        "keywords": ["self-consistency", "self consistency", "multiple reasoning paths", "majority voting"],
        "category": "LLM Behaviour Optimisation",
        "how": "Sample multiple solutions, aggregate via voting or ranking",
        "what": "Reduces variance by selecting most consistent answer across samples"
    },
    "Monte Carlo Tree Search": {
        "keywords": ["mcts", "monte carlo tree", "monte carlo search"],
        "category": "LLM Behaviour Optimisation",
        "how": "Random rollouts -> Backpropagate values -> Guide exploration",
        "what": "Balances exploration and exploitation in reasoning"
    },
    "Process Reward Models": {
        "keywords": ["process reward", "step-by-step reward", "prm", "math-shepherd"],
        "category": "LLM Behaviour Optimisation",
        "how": "Reward each reasoning step, not just final answer",
        "what": "Provides dense feedback for better credit assignment"
    },

    # -------------------------------------------------------------------------
    # 1.2 PROMPT OPTIMISATION
    # -------------------------------------------------------------------------
    "Edit-Based Prompt Search": {
        "keywords": ["gps prompt", "grips", "edit-based instruction", "tempera"],
        "category": "Prompt Optimisation",
        "how": "Iteratively edit prompts based on task performance feedback",
        "what": "Optimizes prompts via discrete edits without gradient access"
    },
    "Evolutionary Prompt Optimization": {
        "keywords": ["evoprompt", "promptbreeder", "genetic prompt", "evolutionary prompt"],
        "category": "Prompt Optimisation",
        "how": "Population of prompts -> Mutation/Crossover -> Selection by fitness",
        "what": "Evolves prompts using genetic algorithm operators"
    },
    "Generative Prompt Engineering": {
        "keywords": ["automatic prompt engineer", "ape", "promptagent", "opro", "large language models as optimizers"],
        "category": "Prompt Optimisation",
        "how": "LLM proposes prompt candidates, evaluates on examples, selects best",
        "what": "Removes human from prompt engineering loop"
    },
    "Text Gradient Optimization": {
        "keywords": ["textgrad", "text gradient", "semantic backprop", "grad-sum"],
        "category": "Prompt Optimisation",
        "how": "Natural language feedback serves as gradient signal for updates",
        "what": "Propagates textual critiques backward to improve prompts"
    },

    # -------------------------------------------------------------------------
    # 1.3 MEMORY OPTIMIZATION
    # -------------------------------------------------------------------------
    "Agent Workflow Memory": {
        "keywords": ["workflow memory", "agent workflow", "action sequence memory"],
        "category": "Memory Optimization",
        "how": "Store successful action sequences -> Retrieve for similar tasks",
        "what": "Remembers proven procedures for task types"
    },
    "Long-Term Memory Systems": {
        "keywords": ["memorybank", "long-term memory", "episodic memory", "memory bank"],
        "category": "Memory Optimization",
        "how": "Store task episodes with outcomes -> Query relevant episodes -> Apply lessons",
        "what": "Persistent storage of experiences for future reference"
    },
    "Compressive Memory": {
        "keywords": ["compress to impress", "compressive memory", "gist memory"],
        "category": "Memory Optimization",
        "how": "Compress long context into retrievable gist representations",
        "what": "Enables long-context understanding without full storage"
    },
    "Skill Library": {
        "keywords": ["skill library", "skill repository", "learned skills"],
        "category": "Memory Optimization",
        "how": "Extract reusable skills from successes -> Store in library -> Compose for new tasks",
        "what": "Accumulates modular capabilities over time"
    },

    # -------------------------------------------------------------------------
    # 1.4 TOOL OPTIMIZATION
    # -------------------------------------------------------------------------
    # Training-Based Tool
    "Tool Instruction Fine-Tuning": {
        "keywords": ["gpt4tools", "toolllm", "tool learning", "toolbench"],
        "category": "Tool Optimization",
        "how": "Fine-tune on tool-use demonstrations -> Generalize to new tools",
        "what": "Learns tool selection and invocation from examples"
    },
    "RL for Tool Selection": {
        "keywords": ["toolrl", "retool", "tool reinforcement", "tool-star"],
        "category": "Tool Optimization",
        "how": "Reward signal for successful tool use -> Policy learns when/how to use tools",
        "what": "Optimizes tool selection through trial and error"
    },
    # Inference-Time Tool
    "Tool Discovery and Retrieval": {
        "keywords": ["toolchain", "tool-planner", "mcp-zero", "tool discovery"],
        "category": "Tool Optimization",
        "how": "Search available tools -> Plan sequence -> Execute and verify",
        "what": "Dynamically finds and chains tools for complex tasks"
    },
    # Tool Creation
    "Tool Creation": {
        "keywords": ["creator", "tool creation", "create tool", "generate tool", "clova"],
        "category": "Tool Optimization",
        "how": "Agent writes code to create new tools when existing ones insufficient",
        "what": "Expands capability by generating reusable tool functions"
    },

    # -------------------------------------------------------------------------
    # 2. MULTI-AGENT OPTIMISATION
    # -------------------------------------------------------------------------
    "Automatic MAS Construction": {
        "keywords": ["metaagent", "adas", "automated design", "automatic construction"],
        "category": "Multi-Agent Optimisation",
        "how": "Automatically construct multi-agent systems based on task requirements",
        "what": "Designs agent teams without manual architecture specification"
    },
    "Workflow Optimization": {
        "keywords": ["aflow", "workflowllm", "metagpt", "autogen", "scoreflow"],
        "category": "Multi-Agent Optimisation",
        "how": "Search/evolve agent communication patterns and task routing",
        "what": "Optimizes how agents coordinate and share information"
    },
    "Multi-Agent Debate": {
        "keywords": ["debate", "multi-agent debate", "agents debate"],
        "category": "Multi-Agent Optimisation",
        "how": "Agents argue opposing positions -> Refine through rounds -> Converge on answer",
        "what": "Adversarial discussion surfaces errors and improves reasoning"
    },
    "Symbolic Learning for Agents": {
        "keywords": ["symbolic learning", "self-evolving agents", "agentverse"],
        "category": "Multi-Agent Optimisation",
        "how": "Agents learn symbolic rules from interactions, update behavior",
        "what": "Enables agents to evolve beyond training distribution"
    },

    # -------------------------------------------------------------------------
    # 3. DOMAIN-SPECIFIC OPTIMISATION
    # -------------------------------------------------------------------------
    # 3.1 Biomedicine - Medical Diagnosis
    "Medical Multi-Agent Systems": {
        "keywords": ["mdagents", "medagentsim", "mdteamgpt", "mmedagent"],
        "category": "Domain: Medical Diagnosis",
        "how": "Multiple specialized medical agents collaborate on diagnosis",
        "what": "Adaptive agent teams for complex medical decision-making"
    },
    "Self-Evolving Biomedical Agents": {
        "keywords": ["stella", "healthflow", "medagent-pro"],
        "category": "Domain: Medical Diagnosis",
        "how": "Agent autonomously improves medical reasoning through meta-planning",
        "what": "Self-improving agents for healthcare research"
    },
    # 3.1 Biomedicine - Molecular Discovery
    "Chemistry Tool Agents": {
        "keywords": ["chemcrow", "cactus", "chemagent"],
        "category": "Domain: Molecular Discovery",
        "how": "Augment LLM with chemistry-specific tools for molecule manipulation",
        "what": "Enables molecular reasoning through specialized tool use"
    },
    "Drug Discovery Agents": {
        "keywords": ["drugagent", "liddia", "genomas"],
        "category": "Domain: Molecular Discovery",
        "how": "Multi-agent collaboration for drug design and analysis",
        "what": "Automates drug discovery pipeline stages"
    },
    # 3.2 Programming - Code Refinement
    "Iterative Code Refinement": {
        "keywords": ["self-refine", "agentcoder", "codecor", "code review"],
        "category": "Domain: Code Refinement",
        "how": "Generate code -> Test -> Refine based on results",
        "what": "Iterative improvement of code through self-feedback"
    },
    "Multi-Agent Code Generation": {
        "keywords": ["openhands", "self-evolving multi-agent", "alphaevolve"],
        "category": "Domain: Code Refinement",
        "how": "Specialized agents for writing, reviewing, testing code",
        "what": "Division of labor for complex software development"
    },
    # 3.2 Programming - Code Debugging
    "Self-Debug": {
        "keywords": ["self-debug", "self debug", "teaching llm to self-debug"],
        "category": "Domain: Code Debugging",
        "how": "Execute code -> Read error messages -> Fix bugs iteratively",
        "what": "Uses execution feedback to identify and correct code errors"
    },
    "Fault-Aware Code Editing": {
        "keywords": ["self-edit", "fault-aware", "rgd debugger"],
        "category": "Domain: Code Debugging",
        "how": "Identify fault locations -> Apply targeted minimal edits",
        "what": "Locates specific errors and makes minimal corrections"
    },
    # 3.3 Scientific Research
    "Scientific Discovery Agents": {
        "keywords": ["piflow", "earthlink", "scientific discovery", "research agent"],
        "category": "Domain: Scientific Research",
        "how": "Multi-agent collaboration guided by scientific principles",
        "what": "Automates hypothesis generation and experimental design"
    },
    # 3.4 Finance/Legal
    "Financial Decision Agents": {
        "keywords": ["finrobot", "fincon", "r&d-agent-quant"],
        "category": "Domain: Finance",
        "how": "Multi-agent systems with conceptual verbal reinforcement",
        "what": "Optimizes trading and financial analysis decisions"
    },
    "Legal Reasoning Agents": {
        "keywords": ["lawluo", "legalgpt", "agentcourt"],
        "category": "Domain: Legal",
        "how": "Chain of legal thought with multi-agent consultation",
        "what": "Structured legal reasoning through agent collaboration"
    },

    # -------------------------------------------------------------------------
    # 4. EVALUATION METHODS
    # -------------------------------------------------------------------------
    "Benchmark-Based Evaluation": {
        "keywords": ["swe-bench", "agentbench", "webarena", "osworld"],
        "category": "Evaluation",
        "how": "Standardized benchmarks measure agent capabilities",
        "what": "Quantitative assessment of agent performance on real tasks"
    },
    "LLM-as-a-Judge": {
        "keywords": ["llm-as-a-judge", "llm judge", "auto-arena"],
        "category": "Evaluation",
        "how": "LLM evaluates agent outputs for quality/correctness",
        "what": "Scalable evaluation without human annotation"
    },
    "Agent Safety Benchmarks": {
        "keywords": ["agentharm", "redcode", "mobilesafetybench", "safelawbench"],
        "category": "Evaluation",
        "how": "Test agents for harmful behaviors and safety violations",
        "what": "Measures alignment and robustness of self-evolving agents"
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
    text_lower = text.lower()
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
        text = (paper.get("parsed_text", "") or paper.get("abstract", "")).lower()
        year = extract_year(arxiv_id)

        for method_name, info in FINEGRAINED_METHODS.items():
            keywords = info["keywords"]

            for kw in keywords:
                if kw.lower() in title or kw.lower() in text:
                    count = text.count(kw.lower())
                    if count >= 2 or kw.lower() in title:
                        context = extract_context(
                            paper.get("parsed_text", "") or paper.get("abstract", ""),
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
    print("=" * 60)
    print("Extracting FINE-GRAINED evolution methods")
    print("(Categories from GitHub repo section subheaders)")
    print("=" * 60)

    # Try parsed_papers.json first, fall back to papers.json
    papers_path = PROCESSED_DIR / "parsed_papers.json"
    fallback_path = PROCESSED_DIR / "papers.json"

    papers = []
    if papers_path.exists():
        with open(papers_path) as f:
            papers = json.load(f)

    if not papers and fallback_path.exists():
        print("No parsed papers, using papers.json metadata (title-based matching)")
        with open(fallback_path) as f:
            papers = json.load(f)

    if not papers:
        print("No papers found. Run parse_awesome_readme.py first.")
        return

    print(f"Papers loaded: {len(papers)}")

    methods = extract_methods(papers)
    connections = build_connections(methods)

    print(f"\nFound {len(methods)} fine-grained methods")
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
        for m in by_cat[cat]:
            years = f"({min(m.years)}-{max(m.years)})" if m.years else ""
            print(f"  - {m.name}: {m.frequency} papers {years}")
            print(f"    HOW: {m.how[:60]}...")

    return methods, connections


if __name__ == "__main__":
    main()
