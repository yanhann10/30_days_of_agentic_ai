"""Compare Self-Debate vs Multi-Agent Debate Results."""

import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def load_latest_results(output_dir: str = "outputs") -> dict:
    result_files = glob.glob(os.path.join(output_dir, "debate_results_*.json"))
    result_files.sort(reverse=True)

    results = {"self-debate": None, "multi-agent": None}

    for f in result_files:
        with open(f, 'r') as fp:
            data = json.load(fp)

        if data.get("debate_results"):
            mode = data["debate_results"][0].get("mode", "multi-agent")
            if results[mode] is None:
                results[mode] = data
                print(f"Loaded {mode}: {f}")

        if results["self-debate"] and results["multi-agent"]:
            break

    return results


def create_comparison_table(results: dict) -> str:
    md = []
    md.append("# Debate Mode Comparison: Self-Debate vs Multi-Agent\n")
    md.append(f"**Generated**: {datetime.now().isoformat()}\n")

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

    md.append("## Aggregate Scores Comparison\n")
    md.append("| Metric | Self-Debate | Multi-Agent | Δ (Self - Multi) |")
    md.append("|--------|-------------|-------------|------------------|")

    metrics = ["factuality", "novelty", "coherence", "actionability", "balance"]

    self_scores = results.get("self-debate", {}).get("evaluation", {}).get("aggregate_scores", {})
    multi_scores = results.get("multi-agent", {}).get("evaluation", {}).get("aggregate_scores", {})

    for metric in metrics:
        self_val = self_scores.get(metric, "N/A")
        multi_val = multi_scores.get(metric, "N/A")

        if isinstance(self_val, (int, float)) and isinstance(multi_val, (int, float)):
            delta = self_val - multi_val
            delta_str = f"{delta:+.2f}"
            self_str = f"{self_val:.2f}"
            multi_str = f"{multi_val:.2f}"
        else:
            delta_str = "N/A"
            self_str = str(self_val)
            multi_str = str(multi_val)

        md.append(f"| **{metric.capitalize()}** | {self_str} | {multi_str} | {delta_str} |")

    md.append("\n## Per-Topic Comparison\n")
    md.append("| Topic | Mode | Rounds | Factuality | Novelty | Coherence | Actionability | Balance |")
    md.append("|-------|------|--------|------------|---------|-----------|---------------|---------|")

    for mode_name, mode_data in results.items():
        if mode_data is None:
            continue

        eval_data = mode_data.get("evaluation", {})
        for i, eval_item in enumerate(eval_data.get("individual_evaluations", [])):
            topic = eval_item.get("topic", "Unknown")[:30] + "..."
            mode = eval_item.get("mode", mode_name)
            rounds = eval_item.get("num_rounds", 4)
            scores = eval_item.get("evaluation", {})

            md.append(
                f"| {topic} | {mode} | {rounds} | "
                f"{scores.get('factuality', 0):.2f} | {scores.get('novelty', 0):.2f} | "
                f"{scores.get('coherence', 0):.2f} | {scores.get('actionability', 0):.2f} | "
                f"{scores.get('balance', 0):.2f} |"
            )

    md.append("\n## Key Insights\n")

    if self_scores and multi_scores:
        insights = []

        if self_scores.get("novelty", 0) > multi_scores.get("novelty", 0):
            insights.append("- **Self-debate produces higher novelty**: Single agent exploring both sides generates more diverse ideas")
        else:
            insights.append("- **Multi-agent produces higher novelty**: Different agent personas bring distinct perspectives")

        if self_scores.get("coherence", 0) > multi_scores.get("coherence", 0):
            insights.append("- **Self-debate is more coherent**: Single agent maintains consistent reasoning thread")
        else:
            insights.append("- **Multi-agent is more coherent**: Structured debate format enforces clear argumentation")

        if self_scores.get("balance", 0) > multi_scores.get("balance", 0):
            insights.append("- **Self-debate is more balanced**: Agent actively tries to be fair to both sides")
        else:
            insights.append("- **Multi-agent is more balanced**: Natural tension between agents ensures both views heard")

        for insight in insights:
            md.append(insight)
    else:
        md.append("- Waiting for both debate modes to complete for comparison insights")

    return "\n".join(md)


def create_comparison_plots(results: dict, output_dir: str = "outputs"):
    metrics = ["factuality", "novelty", "coherence", "actionability", "balance"]

    self_scores = results.get("self-debate", {}).get("evaluation", {}).get("aggregate_scores", {})
    multi_scores = results.get("multi-agent", {}).get("evaluation", {}).get("aggregate_scores", {})

    if not self_scores or not multi_scores:
        print("Need both debate modes for comparison plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.35

    self_vals = [self_scores.get(m, 0) for m in metrics]
    multi_vals = [multi_scores.get(m, 0) for m in metrics]

    bars1 = ax1.bar(x - width/2, self_vals, width, label='Self-Debate', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, multi_vals, width, label='Multi-Agent', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Score')
    ax1.set_title('Aggregate Scores: Self-Debate vs Multi-Agent')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in metrics], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

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

    ax3 = axes[2]
    deltas = [s - m for s, m in zip(self_vals, multi_vals)]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]

    bars3 = ax3.barh(metrics, deltas, color=colors, alpha=0.8)
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.set_xlabel('Δ Score (Self-Debate - Multi-Agent)')
    ax3.set_title('Score Differences\n(+ve = Self-Debate better)')
    ax3.set_xlim(-0.5, 0.5)

    for bar, delta in zip(bars3, deltas):
        width = bar.get_width()
        ax3.annotate(f'{delta:+.2f}', xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if width >= 0 else -5, 0), textcoords="offset points",
                    ha='left' if width >= 0 else 'right', va='center', fontsize=9)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "debate_comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {plot_path}")

    plt.close()
    return plot_path


def main():
    print("Loading debate results...")
    results = load_latest_results()

    print(f"\nSelf-debate data: {'Yes' if results['self-debate'] else 'No'}")
    print(f"Multi-agent data: {'Yes' if results['multi-agent'] else 'No'}")

    # Generate comparison table
    print("\nGenerating comparison table...")
    table_md = create_comparison_table(results)

    table_path = "outputs/debate_mode_comparison.md"
    with open(table_path, 'w') as f:
        f.write(table_md)
    print(f"Saved comparison table to: {table_path}")

    if results["self-debate"] and results["multi-agent"]:
        print("\nGenerating comparison plots...")
        create_comparison_plots(results)
    else:
        print("\nSkipping plots - need both debate modes")

    print("\n" + "="*60)
    print(table_md)


if __name__ == "__main__":
    main()
