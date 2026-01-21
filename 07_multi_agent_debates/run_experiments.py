#!/usr/bin/env python3
"""
Experiment Runner: Compare debate configurations.

Usage:
    python run_experiments.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
from langsmith import traceable, Client
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "agent-eval-debate-experiments")

client = Client()

MODELS = {
    "opus": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-3-5-haiku-20241022",
}

ACADEMIA_PROMPT = """You are Dr. Elena Chen, a Principal Research Scientist.
Focus on: theoretical foundations, research gaps, reproducibility, standardized benchmarks.
Cite papers when possible. Be rigorous but constructive."""

INDUSTRY_PROMPT = """You are Marcus Rodriguez, VP of AI Platform.
Focus on: production deployment, cost/latency tradeoffs, real-world constraints, practical solutions.
Share production experience. Push back on ivory tower solutions."""

STARTUP_PROMPT = """You are Alex Kim, CTO of an AI startup.
Focus on: rapid iteration, minimum viable evaluation, resource constraints, competitive edge.
Balance quality with speed-to-market. Be pragmatic and scrappy."""

MODERATOR_PROMPT = """You are the moderator synthesizing this debate.
Find synthesis between perspectives. Score proposals on novelty (0-1) and usefulness (0-1).
Be diplomatic but decisive. Make final recommendations.

OUTPUT FORMAT (exactly):
## Synthesis
[summary]

## Scores
- Novelty Score: X.X
- Usefulness Score: X.X

## Final Recommendations
[actionable items]
"""

DEBATER_PROMPTS = {
    "academia": ACADEMIA_PROMPT,
    "industry": INDUSTRY_PROMPT,
    "startup": STARTUP_PROMPT,
}

DEBATE_TOPIC = "What evaluation metrics and frameworks are missing for production LLM agents?"

SAMPLE_DOC = """# LLM Agent Evaluation Guide
This document covers evaluation methods for LLM-based agents including:
- Task completion metrics
- Safety evaluations
- User satisfaction measures
Current challenges include multi-turn credit assignment and balancing rigor with speed.
"""


def create_agent(model_key: str, temperature: float = 0.7) -> ChatAnthropic:
    return ChatAnthropic(
        model=MODELS[model_key],
        temperature=temperature,
        max_tokens=2048,
    )


@traceable(name="single_debater_response")
def get_debater_response(
    agent: ChatAnthropic,
    persona_prompt: str,
    document: str,
    topic: str,
    context: str = ""
) -> str:
    messages = [SystemMessage(content=persona_prompt)]

    prompt = f"""Improve this document on LLM evaluation.

DOCUMENT:
{document}

TOPIC: {topic}

{"PREVIOUS DISCUSSION:" + context if context else "You are opening the discussion."}

Provide your perspective with specific recommendations. Keep it concise (300 words max)."""

    messages.append(HumanMessage(content=prompt))
    response = agent.invoke(messages)
    return response.content


@traceable(name="moderator_synthesis")
def get_moderator_synthesis(
    agent: ChatAnthropic,
    document: str,
    topic: str,
    debate_transcript: str
) -> dict:
    messages = [SystemMessage(content=MODERATOR_PROMPT)]

    prompt = f"""Synthesize this debate on improving an LLM evaluation document.

DOCUMENT:
{document}

TOPIC: {topic}

DEBATE TRANSCRIPT:
{debate_transcript}

Provide synthesis, scores (Novelty 0-1, Usefulness 0-1), and recommendations."""

    messages.append(HumanMessage(content=prompt))
    response = agent.invoke(messages)

    content = response.content
    novelty_score = 0.5
    usefulness_score = 0.5

    try:
        for line in content.split('\n'):
            if 'Novelty Score:' in line:
                novelty_score = float(line.split(':')[1].strip().split()[0])
            elif 'Usefulness Score:' in line:
                usefulness_score = float(line.split(':')[1].strip().split()[0])
    except:
        pass

    return {
        "content": content,
        "novelty_score": novelty_score,
        "usefulness_score": usefulness_score,
    }


@traceable(name="run_debate_config")
def run_debate(
    num_debaters: int,
    debater_model: str,
    moderator_model: str,
    document: str = SAMPLE_DOC,
    topic: str = DEBATE_TOPIC
) -> dict:
    debater_keys = list(DEBATER_PROMPTS.keys())[:num_debaters]
    transcript_parts = []

    # Round 1: Initial positions
    for i, key in enumerate(debater_keys):
        agent = create_agent(debater_model)
        response = get_debater_response(
            agent=agent,
            persona_prompt=DEBATER_PROMPTS[key],
            document=document,
            topic=topic,
            context=""
        )
        transcript_parts.append(f"=== {key.upper()} (Round 1) ===\n{response}")

    if num_debaters > 1:
        context = "\n\n".join(transcript_parts)
        for i, key in enumerate(debater_keys):
            agent = create_agent(debater_model)
            response = get_debater_response(
                agent=agent,
                persona_prompt=DEBATER_PROMPTS[key],
                document=document,
                topic=topic,
                context=context
            )
            transcript_parts.append(f"=== {key.upper()} (Round 2) ===\n{response}")

    full_transcript = "\n\n".join(transcript_parts)
    moderator = create_agent(moderator_model, temperature=0.5)
    synthesis = get_moderator_synthesis(
        agent=moderator,
        document=document,
        topic=topic,
        debate_transcript=full_transcript
    )

    return {
        "config": {
            "num_debaters": num_debaters,
            "debater_model": debater_model,
            "moderator_model": moderator_model,
        },
        "transcript": full_transcript,
        "synthesis": synthesis["content"],
        "novelty_score": synthesis["novelty_score"],
        "usefulness_score": synthesis["usefulness_score"],
    }


def run_all_experiments():
    experiments = []

    print("\n" + "="*60)
    print("EXPERIMENT 1: Varying number of debaters (sonnet for all)")
    print("="*60)

    for num_debaters in [1, 2, 3]:
        print(f"\nRunning with {num_debaters} debater(s)...")
        result = run_debate(
            num_debaters=num_debaters,
            debater_model="sonnet",
            moderator_model="sonnet"
        )
        result["experiment"] = f"{num_debaters}_debaters_sonnet"
        experiments.append(result)
        print(f"  Novelty: {result['novelty_score']:.2f}, Usefulness: {result['usefulness_score']:.2f}")

    print("\n" + "="*60)
    print("EXPERIMENT 2: Model combinations (2 debaters)")
    print("="*60)

    model_configs = [
        ("haiku", "haiku", "haiku_all"),
        ("haiku", "sonnet", "haiku_debaters_sonnet_mod"),
        ("sonnet", "sonnet", "sonnet_all"),
        ("sonnet", "opus", "sonnet_debaters_opus_mod"),
    ]

    for debater_model, moderator_model, label in model_configs:
        print(f"\nRunning: debaters={debater_model}, moderator={moderator_model}...")
        try:
            result = run_debate(
                num_debaters=2,
                debater_model=debater_model,
                moderator_model=moderator_model
            )
            result["experiment"] = label
            experiments.append(result)
            print(f"  Novelty: {result['novelty_score']:.2f}, Usefulness: {result['usefulness_score']:.2f}")
        except Exception as e:
            print(f"  Error: {e}")
            experiments.append({
                "experiment": label,
                "config": {"debater_model": debater_model, "moderator_model": moderator_model, "num_debaters": 2},
                "novelty_score": 0.0,
                "usefulness_score": 0.0,
                "error": str(e)
            })

    return experiments


def plot_results(experiments: list, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    debater_exps = [e for e in experiments if e["experiment"].endswith("_debaters_sonnet")]
    if debater_exps:
        x = [e["config"]["num_debaters"] for e in debater_exps]
        novelty = [e["novelty_score"] for e in debater_exps]
        usefulness = [e["usefulness_score"] for e in debater_exps]

        axes[0].plot(x, novelty, 'b-o', label='Novelty', linewidth=2, markersize=10)
        axes[0].plot(x, usefulness, 'g-s', label='Usefulness', linewidth=2, markersize=10)
        axes[0].set_xlabel('Number of Debaters', fontsize=12)
        axes[0].set_ylabel('Score (0-1)', fontsize=12)
        axes[0].set_title('Impact of Number of Debaters\n(All Sonnet)', fontsize=14)
        axes[0].legend()
        axes[0].set_xticks([1, 2, 3])
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)

    model_exps = [e for e in experiments if not e["experiment"].endswith("_debaters_sonnet")]
    if model_exps:
        labels = [e["experiment"].replace("_", "\n") for e in model_exps]
        novelty = [e["novelty_score"] for e in model_exps]
        usefulness = [e["usefulness_score"] for e in model_exps]

        x = np.arange(len(labels))
        width = 0.35

        axes[1].bar(x - width/2, novelty, width, label='Novelty', color='steelblue')
        axes[1].bar(x + width/2, usefulness, width, label='Usefulness', color='seagreen')
        axes[1].set_xlabel('Configuration', fontsize=12)
        axes[1].set_ylabel('Score (0-1)', fontsize=12)
        axes[1].set_title('Model Combinations Comparison\n(2 Debaters)', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=9)
        axes[1].legend()
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "experiment_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    return plot_path


def generate_findings_report(experiments: list, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)

    report = []
    report.append("# Debate Agent Evaluation Findings")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**LangSmith Project:** {os.getenv('LANGSMITH_PROJECT', 'agent-eval-debate-experiments')}")

    # Summary table
    report.append("\n## Summary Results")
    report.append("\n| Experiment | Num Debaters | Debater Model | Moderator Model | Novelty | Usefulness |")
    report.append("|------------|--------------|---------------|-----------------|---------|------------|")
    for exp in experiments:
        config = exp.get("config", {})
        report.append(f"| {exp['experiment']} | {config.get('num_debaters', 'N/A')} | {config.get('debater_model', 'N/A')} | {config.get('moderator_model', 'N/A')} | {exp['novelty_score']:.2f} | {exp['usefulness_score']:.2f} |")

    report.append("\n## Key Findings")

    debater_exps = [e for e in experiments if e["experiment"].endswith("_debaters_sonnet")]
    if debater_exps:
        report.append("\n### 1. Impact of Number of Debaters")
        for exp in debater_exps:
            report.append(f"- **{exp['config']['num_debaters']} debater(s):** Novelty={exp['novelty_score']:.2f}, Usefulness={exp['usefulness_score']:.2f}")

        # Calculate trends
        if len(debater_exps) > 1:
            novelty_trend = debater_exps[-1]["novelty_score"] - debater_exps[0]["novelty_score"]
            useful_trend = debater_exps[-1]["usefulness_score"] - debater_exps[0]["usefulness_score"]
            report.append(f"\n**Trend (1->3 debaters):** Novelty {'+' if novelty_trend >= 0 else ''}{novelty_trend:.2f}, Usefulness {'+' if useful_trend >= 0 else ''}{useful_trend:.2f}")

    model_exps = [e for e in experiments if not e["experiment"].endswith("_debaters_sonnet")]
    if model_exps:
        report.append("\n### 2. Model Configuration Impact")

        best_novelty = max(model_exps, key=lambda x: x["novelty_score"])
        best_useful = max(model_exps, key=lambda x: x["usefulness_score"])

        report.append(f"- **Best for Novelty:** {best_novelty['experiment']} (score: {best_novelty['novelty_score']:.2f})")
        report.append(f"- **Best for Usefulness:** {best_useful['experiment']} (score: {best_useful['usefulness_score']:.2f})")

        # Bigger vs smaller moderator
        report.append("\n**Moderator Model Effect:**")
        for exp in model_exps:
            mod = exp['config'].get('moderator_model', 'unknown')
            deb = exp['config'].get('debater_model', 'unknown')
            if mod != deb:
                report.append(f"- {deb} debaters + {mod} moderator: Novelty={exp['novelty_score']:.2f}, Usefulness={exp['usefulness_score']:.2f}")

    report.append("\n### 3. Recommendations")
    report.append("Based on the experiments:")
    report.append("- [ ] Optimal debater count for quality vs. cost tradeoff")
    report.append("- [ ] Whether bigger moderator improves synthesis")
    report.append("- [ ] Cost-effective configuration for production")

    report.append("\n## LangSmith Traces")
    report.append(f"\nAll traces are available at: https://smith.langchain.com")
    report.append(f"\nProject: `{os.getenv('LANGSMITH_PROJECT', 'agent-eval-debate-experiments')}`")

    report.append("\n## Raw Data")
    report.append("\n```json")
    report.append(json.dumps([{k: v for k, v in e.items() if k != 'transcript' and k != 'synthesis'} for e in experiments], indent=2))
    report.append("```")

    # Save report
    report_content = "\n".join(report)
    report_path = os.path.join(output_dir, "experiment_findings.md")
    with open(report_path, 'w') as f:
        f.write(report_content)

    print(f"Report saved to: {report_path}")
    return report_content


def save_full_results(experiments: list, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"experiment_results_{timestamp}.json")

    with open(results_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "experiments": experiments,
        }, f, indent=2)

    print(f"Full results saved to: {results_path}")
    return results_path


if __name__ == "__main__":
    print("="*60)
    print("DEBATE AGENT EXPERIMENTS")
    print("Testing configurations for optimal novelty & usefulness")
    print("="*60)

    # Run all experiments
    experiments = run_all_experiments()

    # Save and visualize results
    print("\n" + "="*60)
    print("GENERATING OUTPUTS")
    print("="*60)

    save_full_results(experiments)
    plot_results(experiments)
    report = generate_findings_report(experiments)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nCheck outputs/ for:")
    print("  - experiment_findings.md (summary report)")
    print("  - experiment_results.png (comparison charts)")
    print("  - experiment_results_*.json (full data)")
    print(f"\nLangSmith traces: https://smith.langchain.com")
