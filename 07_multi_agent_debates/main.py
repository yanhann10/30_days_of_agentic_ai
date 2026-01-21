#!/usr/bin/env python3
"""
Multi-Agent Debate System for Document Improvement
Academia vs Industry perspectives with Visionary Moderator

Usage:
    python main.py                    # Run full debate and evaluation
    python main.py --debate-only      # Run debates without evaluation
    python main.py --eval-only        # Run evaluation on existing results
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables
load_dotenv()

# Verify LangSmith configuration
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "agent-eval-debate")

from debate_graph import run_all_debates, DEBATE_TOPICS
from evaluator import (
    evaluate_all_debates,
    generate_evaluation_report,
    generate_markdown_report,
    create_langsmith_dataset
)


def load_document(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def save_results(results: dict, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results as JSON
    results_file = os.path.join(output_dir, f"debate_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        # Convert to serializable format
        serializable_results = []
        for r in results["debate_results"]:
            serializable_results.append({
                "topic": r["topic"],
                "transcript": r["result"]["debate_transcript"],
                "moderator_decision": r["result"]["moderator_decision"],
                "novelty_score": r["result"]["novelty_score"],
                "factuality_score": r["result"]["factuality_score"]
            })
        json.dump({
            "debate_results": serializable_results,
            "evaluation": results.get("evaluation"),
            "timestamp": timestamp
        }, f, indent=2)

    print(f"Results saved to: {results_file}")

    if "evaluation" in results:
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        report = generate_evaluation_report(results["evaluation"])
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Evaluation report saved to: {report_file}")

    transcript_file = os.path.join(output_dir, f"debate_transcript_{timestamp}.md")
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write("# Multi-Agent Debate Transcript\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        for r in results["debate_results"]:
            f.write(f"## Topic: {r['topic']}\n\n")
            f.write("\n".join(r["result"]["debate_transcript"]))
            f.write("\n\n---\n\n")

    print(f"Transcript saved to: {transcript_file}")

    return results_file


def generate_improved_document(
    original_doc: str,
    debate_results: list[dict],
    output_path: str
) -> str:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    agent = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8192)

    all_recommendations = []
    for debate in debate_results:
        all_recommendations.append(f"Topic: {debate['topic']}\n{debate['result']['moderator_decision']}")

    prompt = f"""You are tasked with improving a research document based on expert debate recommendations.

=== ORIGINAL DOCUMENT ===
{original_doc}

=== MODERATOR RECOMMENDATIONS FROM DEBATES ===
{chr(10).join(all_recommendations)}

=== YOUR TASK ===
1. Integrate the accepted recommendations into the document
2. Add new sections as recommended
3. Enhance existing sections with the proposed content
4. Maintain the document's structure and style
5. Update the "Last updated" date to today

Output the complete improved document in markdown format.
Preserve all existing valuable content while adding the improvements.
"""

    messages = [
        SystemMessage(content="You are an expert technical writer specializing in AI/ML research documentation."),
        HumanMessage(content=prompt)
    ]

    response = agent.invoke(messages)
    improved_doc = response.content

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(improved_doc)

    print(f"Improved document saved to: {output_path}")
    return improved_doc


@traceable(name="main_pipeline")
def main(args):
    print("=" * 70)
    print("MULTI-AGENT DEBATE SYSTEM")
    print("Academia vs Industry AI Perspectives")
    print("=" * 70)

    doc_path = args.document or "llm_agentic_evaluation_comprehensive.md"
    output_dir = args.output or "outputs"

    print(f"\nLoading document: {doc_path}")
    try:
        original_doc = load_document(doc_path)
        print(f"Document loaded: {len(original_doc)} characters")
    except FileNotFoundError:
        print(f"Error: Document not found at {doc_path}")
        sys.exit(1)

    results = {"debate_results": [], "evaluation": None}

    if not args.eval_only:
        print("\n" + "=" * 70)
        if args.self_debate:
            print(f"PHASE 1: RUNNING SELF-DEBATES ({args.rounds} rounds each)")
        else:
            print("PHASE 1: RUNNING MULTI-AGENT DEBATES")
        print("=" * 70)
        print(f"\nDebate topics ({len(DEBATE_TOPICS)}):")
        for i, topic in enumerate(DEBATE_TOPICS, 1):
            print(f"  {i}. {topic}")

        if args.self_debate:
            print(f"\nMode: Self-Debate (single agent argues both sides)")
            print(f"Rounds per topic: {args.rounds}")
        else:
            print(f"\nMode: Multi-Agent (Academia vs Industry + Moderator)")

        debate_results = run_all_debates(
            original_doc,
            use_self_debate=args.self_debate,
            num_rounds=args.rounds
        )
        results["debate_results"] = debate_results

        print("\n" + "-" * 50)
        print("DEBATE SUMMARY")
        print("-" * 50)
        for i, r in enumerate(debate_results, 1):
            mode = r.get('mode', 'multi-agent')
            rounds = r.get('num_rounds', 4)
            print(f"\nDebate {i}: {r['topic'][:50]}...")
            print(f"  Mode: {mode} | Rounds: {rounds}")
            print(f"  Novelty Score:    {r['result']['novelty_score']:.2f}")
            print(f"  Factuality Score: {r['result']['factuality_score']:.2f}")

    if not args.debate_only:
        print("\n" + "=" * 70)
        print("PHASE 2: EVALUATING DEBATE QUALITY")
        print("=" * 70)

        if not results["debate_results"]:
            # Load from previous run if eval-only mode
            print("No debate results to evaluate. Run with debates first.")
            sys.exit(1)

        evaluation = evaluate_all_debates(results["debate_results"], original_doc)
        results["evaluation"] = evaluation

        report = generate_evaluation_report(evaluation)
        print(report)

        print("\nGenerating markdown evaluation report...")
        md_report_path = os.path.join(output_dir, "evaluation_results.md")
        generate_markdown_report(evaluation, md_report_path)

        print("\nCreating LangSmith dataset...")
        create_langsmith_dataset(results["debate_results"])

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    save_results(results, output_dir)

    if not args.no_generate and results["debate_results"]:
        print("\n" + "=" * 70)
        print("PHASE 3: GENERATING IMPROVED DOCUMENT")
        print("=" * 70)

        improved_path = os.path.join(output_dir, "llm_agentic_evaluation_improved.md")
        generate_improved_document(original_doc, results["debate_results"], improved_path)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nLangSmith traces available at: https://smith.langchain.com")
    print(f"Project: {os.getenv('LANGSMITH_PROJECT', 'agent-eval-debate')}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate System for Document Improvement"
    )
    parser.add_argument(
        "--document", "-d",
        help="Path to the document to improve",
        default="llm_agentic_evaluation_comprehensive.md"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for results",
        default="outputs"
    )
    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Run debates without evaluation"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation on existing results"
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generating improved document"
    )
    parser.add_argument(
        "--self-debate",
        action="store_true",
        help="Use self-debate mode (single agent argues both sides)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=2,
        help="Number of debate rounds (for self-debate mode, default: 2)"
    )

    args = parser.parse_args()
    main(args)
