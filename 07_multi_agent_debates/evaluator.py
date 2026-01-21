"""Evaluation system for debate agents."""

import json
import os
import numpy as np
from datetime import datetime
from typing import Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import Client, traceable
from langsmith.schemas import Run, Example
from pydantic import BaseModel, Field
from openai import OpenAI

# DeepEval imports
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

# Initialize clients
langsmith_client = Client()
openai_client = OpenAI()


class EvaluationRubric(BaseModel):
    factuality: float = Field(description="Score 0-1 for factual accuracy (via deepeval)")
    novelty: float = Field(description="Score 0-1 for novel contributions (via embedding similarity)")
    coherence: float = Field(description="Score 0-1 for argument coherence")
    actionability: float = Field(description="Score 0-1 for actionable recommendations")
    balance: float = Field(description="Score 0-1 for balanced perspective")
    reasoning: str = Field(description="Explanation of scores")
    method: str = Field(default="hybrid", description="Evaluation method used")


class DebateRoundMetrics(BaseModel):
    round_number: int
    speaker: str
    factuality: float
    novelty: float
    key_claims: list[str]
    new_content_added: list[str]


@traceable(name="evaluate_factuality_deepeval")
def evaluate_factuality_deepeval(
    claims: str,
    context: str,
    threshold: float = 0.7
) -> tuple[float, str]:
    try:
        # Create test case - FaithfulnessMetric checks if output is grounded in context
        test_case = LLMTestCase(
            input="Evaluate the factual accuracy of the following claims",
            actual_output=claims,
            retrieval_context=[context]
        )

        # Use FaithfulnessMetric - checks claims are grounded in context
        metric = FaithfulnessMetric(
            threshold=threshold,
            model="gpt-4o-mini",
            include_reason=True
        )

        metric.measure(test_case)

        return metric.score, metric.reason or "No detailed reasoning provided"

    except Exception as e:
        print(f"DeepEval error: {e}")
        return 0.7, f"Fallback score due to error: {str(e)}"


@traceable(name="evaluate_hallucination_deepeval")
def evaluate_hallucination_deepeval(
    generated_text: str,
    context: str
) -> tuple[float, str]:
    try:
        test_case = LLMTestCase(
            input="Check for hallucinations",
            actual_output=generated_text,
            context=[context]
        )

        metric = HallucinationMetric(
            threshold=0.5,
            model="gpt-4o-mini",
            include_reason=True
        )

        metric.measure(test_case)
        factuality_score = 1.0 - metric.score
        return factuality_score, metric.reason or "No hallucinations detected"

    except Exception as e:
        print(f"Hallucination check error: {e}")
        return 0.7, f"Fallback: {str(e)}"


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


@traceable(name="evaluate_novelty_embeddings")
def evaluate_novelty_embeddings(
    new_content: str,
    original_document: str,
    chunk_size: int = 1000
) -> tuple[float, str, list[str]]:
    try:
        new_chunks = [new_content[i:i+chunk_size] for i in range(0, len(new_content), chunk_size)]

        if len(original_document) > 8000:
            doc_sample = original_document[:3000] + original_document[-3000:]
        else:
            doc_sample = original_document

        original_embedding = get_embedding(doc_sample)

        novelty_scores = []
        novel_segments = []

        for chunk in new_chunks:
            if len(chunk.strip()) < 50:
                continue

            chunk_embedding = get_embedding(chunk)
            similarity = cosine_similarity(chunk_embedding, original_embedding)
            chunk_novelty = 1.0 - similarity
            novelty_scores.append(chunk_novelty)

            if chunk_novelty > 0.3:
                novel_segments.append(chunk[:200] + "...")

        if not novelty_scores:
            return 0.5, "No substantial content to evaluate", []

        avg_novelty = sum(novelty_scores) / len(novelty_scores)

        # Interpret score
        if avg_novelty > 0.4:
            interpretation = "High novelty - significant new content"
        elif avg_novelty > 0.25:
            interpretation = "Moderate novelty - some new perspectives"
        else:
            interpretation = "Low novelty - mostly restatements"

        reasoning = f"{interpretation}. Avg embedding distance: {avg_novelty:.3f}. Found {len(novel_segments)} highly novel segments."

        return avg_novelty, reasoning, novel_segments[:5]

    except Exception as e:
        print(f"Novelty evaluation error: {e}")
        return 0.5, f"Error: {str(e)}", []


@traceable(name="evaluate_coherence_llm")
def evaluate_coherence_llm(debate_transcript: str) -> tuple[float, str]:
    agent = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.2, max_tokens=1024)

    prompt = f"""Evaluate the coherence of this debate transcript.

TRANSCRIPT:
{debate_transcript[:4000]}

Score the coherence from 0 to 1:
- 1.0: Crystal clear arguments, logical flow, no contradictions
- 0.7: Generally coherent with minor gaps
- 0.4: Some confusing or contradictory points
- 0.0: Incoherent, contradictory arguments

Respond ONLY with JSON:
{{"score": 0.X, "reasoning": "brief explanation"}}
"""

    response = agent.invoke([HumanMessage(content=prompt)])

    try:
        # Parse JSON from response
        content = response.content
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        data = json.loads(content)
        return data["score"], data["reasoning"]
    except:
        return 0.7, "Default coherence score"


@traceable(name="evaluate_debate_round_hybrid")
def evaluate_debate_round(
    debate_transcript: str,
    original_document: str,
    proposed_improvements: str,
    round_number: int = 1
) -> EvaluationRubric:
    print(f"  [Round {round_number}] Running hybrid evaluation...")

    # 1. Factuality via DeepEval
    print(f"    - Evaluating factuality (deepeval)...")
    factuality_score, factuality_reason = evaluate_factuality_deepeval(
        claims=proposed_improvements,
        context=original_document[:6000]
    )

    # 2. Novelty via embeddings
    print(f"    - Evaluating novelty (embeddings)...")
    novelty_score, novelty_reason, novel_segments = evaluate_novelty_embeddings(
        new_content=proposed_improvements,
        original_document=original_document
    )

    # 3. Coherence via LLM
    print(f"    - Evaluating coherence (LLM)...")
    coherence_score, coherence_reason = evaluate_coherence_llm(debate_transcript)

    print(f"    - Evaluating actionability & balance (LLM)...")
    agent = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.2, max_tokens=1024)

    ab_prompt = f"""Evaluate these two dimensions for the debate improvements:

PROPOSED IMPROVEMENTS:
{proposed_improvements[:3000]}

1. ACTIONABILITY (0-1): Can recommendations be implemented?
   - 1.0: Specific, concrete, immediately actionable
   - 0.5: Somewhat vague
   - 0.0: Abstract without practical guidance

2. BALANCE (0-1): Does it fairly consider multiple perspectives?
   - 1.0: Excellent synthesis
   - 0.5: Noticeable bias
   - 0.0: Completely one-sided

Respond ONLY with JSON:
{{"actionability": 0.X, "balance": 0.X, "reasoning": "brief explanation"}}
"""

    ab_response = agent.invoke([HumanMessage(content=ab_prompt)])

    try:
        content = ab_response.content
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        ab_data = json.loads(content)
        actionability_score = ab_data["actionability"]
        balance_score = ab_data["balance"]
        ab_reason = ab_data["reasoning"]
    except:
        actionability_score = 0.7
        balance_score = 0.7
        ab_reason = "Default scores"

    full_reasoning = f"""
**Factuality (deepeval)**: {factuality_score:.2f} - {factuality_reason}
**Novelty (embeddings)**: {novelty_score:.2f} - {novelty_reason}
**Coherence (LLM)**: {coherence_score:.2f} - {coherence_reason}
**Actionability/Balance (LLM)**: {actionability_score:.2f}/{balance_score:.2f} - {ab_reason}
"""

    return EvaluationRubric(
        factuality=factuality_score,
        novelty=novelty_score,
        coherence=coherence_score,
        actionability=actionability_score,
        balance=balance_score,
        reasoning=full_reasoning.strip(),
        method="hybrid (deepeval + embeddings + LLM)"
    )


@traceable(name="evaluate_all_debates")
def evaluate_all_debates(debate_results: list[dict], original_document: str) -> dict:
    evaluations = []
    round_metrics = []

    for i, debate in enumerate(debate_results):
        print(f"\nEvaluating debate {i+1}: {debate['topic'][:50]}...")

        transcript = "\n".join(debate["result"]["debate_transcript"])
        improvements = debate["result"]["moderator_decision"]

        eval_result = evaluate_debate_round(
            debate_transcript=transcript,
            original_document=original_document,
            proposed_improvements=improvements,
            round_number=i+1
        )

        evaluations.append({
            "topic": debate["topic"],
            "round": i + 1,
            "mode": debate.get("mode", "multi-agent"),
            "num_rounds": debate.get("num_rounds", 4),
            "evaluation": eval_result.model_dump(),
            "moderator_novelty": debate["result"]["novelty_score"],
            "moderator_factuality": debate["result"]["factuality_score"]
        })

        # Track per-round metrics
        round_metrics.append({
            "round": i + 1,
            "topic": debate["topic"][:40] + "...",
            "factuality": eval_result.factuality,
            "novelty": eval_result.novelty,
            "coherence": eval_result.coherence,
            "actionability": eval_result.actionability,
            "balance": eval_result.balance
        })

    # Aggregate scores
    avg_scores = {
        "factuality": sum(e["evaluation"]["factuality"] for e in evaluations) / len(evaluations),
        "novelty": sum(e["evaluation"]["novelty"] for e in evaluations) / len(evaluations),
        "coherence": sum(e["evaluation"]["coherence"] for e in evaluations) / len(evaluations),
        "actionability": sum(e["evaluation"]["actionability"] for e in evaluations) / len(evaluations),
        "balance": sum(e["evaluation"]["balance"] for e in evaluations) / len(evaluations),
    }

    return {
        "individual_evaluations": evaluations,
        "round_metrics": round_metrics,
        "aggregate_scores": avg_scores,
        "timestamp": datetime.now().isoformat(),
        "methods_used": {
            "factuality": "deepeval FaithfulnessMetric",
            "novelty": "OpenAI text-embedding-3-small + cosine similarity",
            "coherence": "Claude LLM judge",
            "actionability": "Claude LLM judge",
            "balance": "Claude LLM judge"
        }
    }


def generate_evaluation_report(evaluation_results: dict) -> str:
    report = []
    report.append("=" * 80)
    report.append("DEBATE EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {evaluation_results['timestamp']}")

    # Methods used
    report.append("\n" + "-" * 50)
    report.append("EVALUATION METHODS")
    report.append("-" * 50)
    for metric, method in evaluation_results.get("methods_used", {}).items():
        report.append(f"  {metric}: {method}")

    # Aggregate scores
    report.append("\n" + "-" * 50)
    report.append("AGGREGATE SCORES")
    report.append("-" * 50)

    for metric, score in evaluation_results["aggregate_scores"].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        report.append(f"{metric.capitalize():15} [{bar}] {score:.2f}")

    # Round-by-round table
    report.append("\n" + "-" * 50)
    report.append("ROUND-BY-ROUND METRICS")
    report.append("-" * 50)

    # Table header
    report.append(f"{'Round':<6} {'Factuality':<12} {'Novelty':<10} {'Coherence':<11} {'Action.':<10} {'Balance':<10}")
    report.append("-" * 70)

    for rm in evaluation_results.get("round_metrics", []):
        report.append(
            f"{rm['round']:<6} "
            f"{rm['factuality']:<12.2f} "
            f"{rm['novelty']:<10.2f} "
            f"{rm['coherence']:<11.2f} "
            f"{rm['actionability']:<10.2f} "
            f"{rm['balance']:<10.2f}"
        )

    # Individual evaluations detail
    report.append("\n" + "-" * 50)
    report.append("DETAILED REASONING")
    report.append("-" * 50)

    for eval_data in evaluation_results["individual_evaluations"]:
        report.append(f"\n[Round {eval_data['round']}] {eval_data['topic'][:60]}...")
        report.append(eval_data['evaluation']['reasoning'][:500])

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def generate_markdown_report(evaluation_results: dict, output_path: str = "outputs/evaluation_results.md"):
    md = []
    md.append("# Debate Evaluation Report\n")
    md.append(f"**Generated**: {evaluation_results['timestamp']}\n")

    # Methods table
    md.append("## Evaluation Methods Used\n")
    md.append("| Metric | Method |")
    md.append("|--------|--------|")
    for metric, method in evaluation_results.get("methods_used", {}).items():
        md.append(f"| {metric.capitalize()} | {method} |")
    md.append("")

    # Aggregate scores
    md.append("## Aggregate Scores\n")
    md.append("| Metric | Score | Visual |")
    md.append("|--------|-------|--------|")
    for metric, score in evaluation_results["aggregate_scores"].items():
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        md.append(f"| **{metric.capitalize()}** | {score:.2f} | `{bar}` |")
    md.append("")

    # Round-by-round table
    md.append("## Round-by-Round Metrics\n")
    md.append("| Debate | Topic | Mode | Rounds | Factuality | Novelty | Coherence | Actionability | Balance |")
    md.append("|--------|-------|------|--------|------------|---------|-----------|---------------|---------|")

    for i, rm in enumerate(evaluation_results.get("round_metrics", [])):
        eval_data = evaluation_results["individual_evaluations"][i]
        topic = eval_data["topic"][:25] + "..."
        mode = eval_data.get("mode", "multi-agent")
        num_rounds = eval_data.get("num_rounds", 4)
        md.append(
            f"| {rm['round']} | {topic} | {mode} | {num_rounds} | "
            f"{rm['factuality']:.2f} | {rm['novelty']:.2f} | {rm['coherence']:.2f} | "
            f"{rm['actionability']:.2f} | {rm['balance']:.2f} |"
        )
    md.append("")

    # Detailed reasoning
    md.append("## Detailed Reasoning\n")
    for eval_data in evaluation_results["individual_evaluations"]:
        md.append(f"### Round {eval_data['round']}: {eval_data['topic'][:50]}...\n")
        md.append(eval_data['evaluation']['reasoning'])
        md.append("")

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(md))

    print(f"Markdown report saved to: {output_path}")
    return "\n".join(md)


def create_langsmith_dataset(debate_results: list[dict], dataset_name: str = "agent-eval-debates"):
    try:
        try:
            dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
            print(f"Using existing dataset: {dataset_name}")
        except:
            dataset = langsmith_client.create_dataset(
                dataset_name=dataset_name,
                description="Debate results for LLM/Agent evaluation document improvement"
            )
            print(f"Created new dataset: {dataset_name}")

        for debate in debate_results:
            langsmith_client.create_example(
                inputs={
                    "topic": debate["topic"],
                    "document_content": debate["result"]["document_content"][:2000]
                },
                outputs={
                    "moderator_decision": debate["result"]["moderator_decision"],
                    "novelty_score": debate["result"]["novelty_score"],
                    "factuality_score": debate["result"]["factuality_score"]
                },
                dataset_id=dataset.id
            )

        print(f"Added {len(debate_results)} examples to dataset")
        return dataset

    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None


if __name__ == "__main__":
    # Test the evaluation functions
    print("Testing evaluation methods...")

    # Test factuality
    test_claims = "LLM evaluation requires careful consideration of bias. The HELM benchmark from Stanford covers 42 tasks."
    test_context = "HELM is Stanford's holistic evaluation framework covering 42 different tasks across multiple dimensions."

    score, reason = evaluate_factuality_deepeval(test_claims, test_context)
    print(f"Factuality test: {score:.2f} - {reason}")

    # Test novelty
    new_content = "A completely new approach to evaluation using quantum computing and blockchain integration."
    original = "Traditional LLM evaluation uses metrics like BLEU, ROUGE, and human judgment."

    novelty, reason, segments = evaluate_novelty_embeddings(new_content, original)
    print(f"Novelty test: {novelty:.2f} - {reason}")
