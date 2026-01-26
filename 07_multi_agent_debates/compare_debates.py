"""Compare Self-Debate vs Multi-Agent Debate Results with plots and tables."""

import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def load_latest_results(output_dir: str = "outputs") -> dict:
    """Load latest debate result files for each mode."""
    result_files = glob.glob(os.path.join(output_dir, "debate_results_*.json"))
    result_files.sort(reverse=True)

    results = {"self-debate": None, "multi-agent": None}

    for f in result_files:
        with open(f, 'r') as fp:
            data = json.load(fp)

        # Detect mode from evaluation data or transcript content
        mode = None

        # Check evaluation data first
        if data.get("evaluation"):
            evals = data["evaluation"].get("individual_evaluations", [])
            if evals and evals[0].get("mode"):
                mode = evals[0]["mode"]

        # Fallback: check transcript for self-debate markers
        if mode is None and data.get("debate_results"):
            first_result = data["debate_results"][0]
            transcript = first_result.get("transcript", [])
            if transcript:
                first_turn = str(transcript[0]) if transcript else ""
                if "PROPONENT" in first_turn or "OPPONENT" in first_turn:
                    mode = "self-debate"
                elif "ACADEMIA" in first_turn or "INDUSTRY" in first_turn:
                    mode = "multi-agent"

        if mode and results.get(mode) is None:
            results[mode] = data
            results[mode]["_source_file"] = f
            print(f"Loaded {mode}: {f}")

        if results["self-debate"] and results["multi-agent"]:
            break

    return results


def extract_scores(data: dict) -> dict:
    """Extract aggregate scores from debate results."""
    if data.get("evaluation") and data["evaluation"].get("aggregate_scores"):
        return data["evaluation"]["aggregate_scores"]
    return {}


def create_comparison_table(results: dict) -> str:
    """Create markdown comparison table."""
    md = []
    md.append("# Debate Mode Comparison: Self-Debate vs Multi-Agent\n")
    md.append(f"**Generated**: {datetime.now().isoformat()}\n")

    # Novelty explanation
    md.append("## What is Novelty Measured Against?\n")
    md.append("""
**Novelty Score** measures how different the debate recommendations are from the **original document**:
- Uses OpenAI `text-embedding-3-small` to embed both original document and new recommendations
- Calculates **cosine similarity** between embeddings
- **Novelty = 1 - similarity** (low similarity = high novelty = genuinely new content)
- Score interpretation:
  - 0.0-0.2: Low novelty (mostly restating existing content)
  - 0.2-0.4: Moderate novelty (some new perspectives)
  - 0.4+: High novelty (significant new insights)
""")

    # Aggregate scores comparison
    md.append("## Aggregate Scores Comparison\n")
    md.append("| Metric | Self-Debate | Multi-Agent | Δ (Self - Multi) | Winner |")
    md.append("|--------|-------------|-------------|------------------|--------|")

    metrics = ["factuality", "novelty", "coherence", "actionability", "balance"]

    self_scores = extract_scores(results.get("self-debate", {}))
    multi_scores = extract_scores(results.get("multi-agent", {}))

    for metric in metrics:
        self_val = self_scores.get(metric, 0)
        multi_val = multi_scores.get(metric, 0)

        if isinstance(self_val, (int, float)) and isinstance(multi_val, (int, float)):
            delta = self_val - multi_val
            winner = "Self" if delta > 0.02 else ("Multi" if delta < -0.02 else "Tie")
            md.append(f"| **{metric.capitalize()}** | {self_val:.2f} | {multi_val:.2f} | {delta:+.2f} | {winner} |")
        else:
            md.append(f"| **{metric.capitalize()}** | N/A | N/A | N/A | N/A |")

    # Per-topic comparison
    md.append("\n## Per-Topic Detailed Comparison\n")

    for mode_name, mode_data in results.items():
        if mode_data is None:
            continue

        md.append(f"\n### {mode_name.title()} Mode Results\n")

        eval_data = mode_data.get("evaluation", {})
        individual_evals = eval_data.get("individual_evaluations", [])

        if individual_evals:
            md.append("| # | Topic | Rounds | Factuality | Novelty | Coherence | Action. | Balance |")
            md.append("|---|-------|--------|------------|---------|-----------|---------|---------|")

            for i, eval_item in enumerate(individual_evals, 1):
                topic = eval_item.get("topic", "Unknown")[:30] + "..."
                rounds = eval_item.get("num_rounds", 4)
                scores = eval_item.get("evaluation", {})

                md.append(
                    f"| {i} | {topic} | {rounds} | "
                    f"{scores.get('factuality', 0):.2f} | {scores.get('novelty', 0):.2f} | "
                    f"{scores.get('coherence', 0):.2f} | {scores.get('actionability', 0):.2f} | "
                    f"{scores.get('balance', 0):.2f} |"
                )

    # Key insights
    md.append("\n## Key Insights\n")

    if self_scores and multi_scores:
        insights = []

        # Novelty comparison
        self_nov = self_scores.get("novelty", 0)
        multi_nov = multi_scores.get("novelty", 0)
        if self_nov > multi_nov + 0.02:
            insights.append(f"- **Self-debate produces higher novelty** ({self_nov:.2f} vs {multi_nov:.2f}): Single agent exploring both sides generates more diverse ideas")
        elif multi_nov > self_nov + 0.02:
            insights.append(f"- **Multi-agent produces higher novelty** ({multi_nov:.2f} vs {self_nov:.2f}): Different agent personas bring distinct perspectives")
        else:
            insights.append(f"- **Novelty is comparable** ({self_nov:.2f} vs {multi_nov:.2f}): Both modes generate similar amounts of new content")

        # Coherence comparison
        self_coh = self_scores.get("coherence", 0)
        multi_coh = multi_scores.get("coherence", 0)
        if self_coh > multi_coh + 0.05:
            insights.append(f"- **Self-debate is more coherent** ({self_coh:.2f} vs {multi_coh:.2f}): Single agent maintains consistent reasoning thread")
        elif multi_coh > self_coh + 0.05:
            insights.append(f"- **Multi-agent is more coherent** ({multi_coh:.2f} vs {self_coh:.2f}): Structured debate format enforces clear argumentation")
        else:
            insights.append(f"- **Coherence is comparable** ({self_coh:.2f} vs {multi_coh:.2f})")

        # Balance comparison
        self_bal = self_scores.get("balance", 0)
        multi_bal = multi_scores.get("balance", 0)
        if self_bal > multi_bal + 0.05:
            insights.append(f"- **Self-debate is more balanced** ({self_bal:.2f} vs {multi_bal:.2f}): Agent actively tries to be fair to both sides")
        elif multi_bal > self_bal + 0.05:
            insights.append(f"- **Multi-agent is more balanced** ({multi_bal:.2f} vs {self_bal:.2f}): Natural tension between agents ensures both views heard")
        else:
            insights.append(f"- **Balance is comparable** ({self_bal:.2f} vs {multi_bal:.2f})")

        # Rounds comparison
        insights.append("")
        insights.append("### Mode Characteristics")
        insights.append("- **Self-Debate**: 2 rounds per topic, single agent argues proponent/opponent")
        insights.append("- **Multi-Agent**: 4 rounds per topic (Academia → Industry → Rebuttals → Moderator)")

        for insight in insights:
            md.append(insight)
    else:
        md.append("- Waiting for both debate modes to complete for comparison insights")

    return "\n".join(md)


def create_comparison_plots(results: dict, output_dir: str = "outputs"):
    """Create comparison visualizations."""

    metrics = ["factuality", "novelty", "coherence", "actionability", "balance"]

    self_scores = extract_scores(results.get("self-debate", {}))
    multi_scores = extract_scores(results.get("multi-agent", {}))

    if not self_scores or not multi_scores:
        print("Need both debate modes for comparison plots")
        return None

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Self-Debate vs Multi-Agent Debate Comparison", fontsize=14, fontweight='bold')

    # 1. Bar chart comparison
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.35

    self_vals = [self_scores.get(m, 0) for m in metrics]
    multi_vals = [multi_scores.get(m, 0) for m in metrics]

    bars1 = ax1.bar(x - width/2, self_vals, width, label='Self-Debate (2 rounds)', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, multi_vals, width, label='Multi-Agent (4 rounds)', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Score')
    ax1.set_title('Aggregate Scores by Metric')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in metrics], rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # 2. Radar chart
    ax2 = axes[1]
    ax2.remove()
    ax2 = fig.add_subplot(1, 3, 2, projection='polar')

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    self_vals_radar = self_vals + self_vals[:1]
    multi_vals_radar = multi_vals + multi_vals[:1]

    ax2.plot(angles, self_vals_radar, 'o-', linewidth=2, label='Self-Debate', color='#3498db')
    ax2.fill(angles, self_vals_radar, alpha=0.25, color='#3498db')
    ax2.plot(angles, multi_vals_radar, 'o-', linewidth=2, label='Multi-Agent', color='#e74c3c')
    ax2.fill(angles, multi_vals_radar, alpha=0.25, color='#e74c3c')

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([m.capitalize() for m in metrics])
    ax2.set_ylim(0, 1)
    ax2.set_title('Radar Comparison')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 3. Delta chart (difference)
    ax3 = axes[2]
    deltas = [s - m for s, m in zip(self_vals, multi_vals)]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]

    bars3 = ax3.barh(metrics, deltas, color=colors, alpha=0.8)
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.set_xlabel('Δ Score (Self-Debate - Multi-Agent)')
    ax3.set_title('Score Differences\n(Green = Self-Debate better)')
    ax3.set_xlim(-0.3, 0.3)

    # Add value labels
    for bar, delta in zip(bars3, deltas):
        width = bar.get_width()
        ax3.annotate(f'{delta:+.2f}', xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if width >= 0 else -5, 0), textcoords="offset points",
                    ha='left' if width >= 0 else 'right', va='center', fontsize=9)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "debate_comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {plot_path}")

    plt.close()
    return plot_path


def main():
    """Generate comparison report and plots."""
    print("=" * 60)
    print("DEBATE MODE COMPARISON: Self-Debate vs Multi-Agent")
    print("=" * 60)

    print("\nLoading debate results...")
    results = load_latest_results()

    print(f"\nSelf-debate data: {'Yes' if results['self-debate'] else 'No'}")
    print(f"Multi-agent data: {'Yes' if results['multi-agent'] else 'No'}")

    # Generate comparison table
    print("\nGenerating comparison table...")
    table_md = create_comparison_table(results)

    table_path = "outputs/debate_mode_comparison.md"
    os.makedirs("outputs", exist_ok=True)
    with open(table_path, 'w') as f:
        f.write(table_md)
    print(f"Saved comparison table to: {table_path}")

    # Generate plots if we have both
    if results["self-debate"] and results["multi-agent"]:
        print("\nGenerating comparison plots...")
        create_comparison_plots(results)
    else:
        print("\nSkipping plots - need both debate modes")

    # Print table to console
    print("\n" + "=" * 60)
    print(table_md)


if __name__ == "__main__":
    main()
