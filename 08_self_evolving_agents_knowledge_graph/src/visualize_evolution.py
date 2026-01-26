#!/usr/bin/env python3
"""
Visualize EVOLUTION methods graph.
Focused on HOW agents evolve, with papers and temporal view.
"""

import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "viz"

# Category colors - evolution focused
CATEGORY_COLORS = {
    'evolutionary': '#FF6B6B',      # Red - genetic/evolutionary
    'rl-evolution': '#4ECDC4',      # Teal - RL-based
    'self-improvement': '#45B7D1',  # Blue - self-refine loops
    'self-training': '#96CEB4',     # Green - bootstrapping
    'prompt-evolution': '#FFEAA7',  # Yellow - prompt optimization
    'experience-based': '#DDA0DD',  # Plum - memory/experience
    'meta-learning': '#FF8C42',     # Orange - meta-learning
    'multi-agent': '#98D8C8',       # Mint - multi-agent
    'architecture': '#C9B1FF',      # Lavender - architecture search
    'other': '#95A5A6'              # Gray
}

# Method descriptions focused on evolution
METHOD_DESCRIPTIONS = {
    'Genetic Algorithm': {
        'desc': 'Evolves solutions via selection, crossover, and mutation',
        'how_evolves': 'Population of prompts/agents compete and combine successful traits',
        'eval': 'Fitness score, generation improvement'
    },
    'PPO': {
        'desc': 'Proximal Policy Optimization - stable RL training',
        'how_evolves': 'Updates policy weights based on reward signal with stability constraints',
        'eval': 'Reward, KL divergence from reference'
    },
    'RLHF': {
        'desc': 'Reinforcement Learning from Human Feedback',
        'how_evolves': 'Aligns model behavior using human preference data',
        'eval': 'Human preference win rate, reward model score'
    },
    'DPO': {
        'desc': 'Direct Preference Optimization',
        'how_evolves': 'Directly optimizes policy on preference pairs without reward model',
        'eval': 'Preference accuracy, downstream task performance'
    },
    'Self-Refine': {
        'desc': 'LLM critiques and improves its own output',
        'how_evolves': 'Iterative feedback loop: generate -> critique -> refine',
        'eval': 'Quality improvement per iteration'
    },
    'Self-Reflection': {
        'desc': 'Agent reflects on actions to improve future behavior',
        'how_evolves': 'Explicit reflection step extracts lessons from experience',
        'eval': 'Task success rate over episodes'
    },
    'Self-Debug': {
        'desc': 'LLM debugs its own code using error feedback',
        'how_evolves': 'Uses execution errors to iteratively fix code',
        'eval': 'Pass@k, fix success rate'
    },
    'Bootstrapping': {
        'desc': 'Uses own outputs as training data',
        'how_evolves': 'Filters high-quality self-generated examples for fine-tuning',
        'eval': 'Output quality, filtering precision'
    },
    'Prompt Optimization': {
        'desc': 'Automatically optimizes prompts for tasks',
        'how_evolves': 'Search/gradient methods find better prompts',
        'eval': 'Task accuracy, prompt transferability'
    },
    'Automatic Prompt Engineering': {
        'desc': 'LLM generates and tests prompts automatically',
        'how_evolves': 'Meta-prompt generates candidates, evaluates, selects best',
        'eval': 'Downstream task performance'
    },
    'Meta-Prompt': {
        'desc': 'Prompt that generates/improves other prompts',
        'how_evolves': 'Higher-order prompt evolves task-specific prompts',
        'eval': 'Generated prompt quality'
    },
    'EvoPrompt': {
        'desc': 'Evolutionary algorithm for prompt optimization',
        'how_evolves': 'Treats prompts as individuals, evolves via LLM-based mutation',
        'eval': 'Fitness on held-out examples'
    },
    'Curriculum Learning': {
        'desc': 'Training on progressively harder examples',
        'how_evolves': 'Staged difficulty enables learning complex skills',
        'eval': 'Final task performance, learning stability'
    },
    'Meta-Learning': {
        'desc': 'Learning to learn across tasks',
        'how_evolves': 'Inner loop adapts to task, outer loop improves adaptation',
        'eval': 'Few-shot generalization'
    },
    'Self-Play': {
        'desc': 'Agent improves by playing against itself',
        'how_evolves': 'Competitive self-play drives continuous improvement',
        'eval': 'Elo rating, game-theoretic metrics'
    },
    'Multi-Agent Debate': {
        'desc': 'Multiple agents argue to reach better answers',
        'how_evolves': 'Diverse perspectives improve reasoning through argumentation',
        'eval': 'Factual accuracy, consensus quality'
    },
    'Agent Collaboration': {
        'desc': 'Specialized agents work together',
        'how_evolves': 'Division of labor enables complex task completion',
        'eval': 'Task success, coordination efficiency'
    },
    'Tool Creation': {
        'desc': 'Agent creates new tools for itself',
        'how_evolves': 'Expands capabilities by generating reusable tools',
        'eval': 'Tool reuse rate, task coverage'
    },
    'Experience Replay': {
        'desc': 'Reuses past experiences for learning',
        'how_evolves': 'Replays successful trajectories to reinforce good behavior',
        'eval': 'Sample efficiency, learning speed'
    },
    'Workflow Memory': {
        'desc': 'Stores successful action sequences',
        'how_evolves': 'Reuses proven workflows for similar tasks',
        'eval': 'Workflow reuse success rate'
    },
    'Neural Architecture Search': {
        'desc': 'Automatically searches for optimal architectures',
        'how_evolves': 'Explores architecture space to find better structures',
        'eval': 'Accuracy-efficiency tradeoff'
    },
    'Iterative Refinement': {
        'desc': 'Repeated cycles of improvement',
        'how_evolves': 'Each iteration builds on previous output',
        'eval': 'Quality convergence, iteration count'
    },
    'Self-Correction': {
        'desc': 'Agent detects and fixes its own errors',
        'how_evolves': 'Internal verification triggers correction when errors detected',
        'eval': 'Error detection rate, correction success'
    },
    'Self-Instruct': {
        'desc': 'LLM generates its own instruction-following data',
        'how_evolves': 'Creates training examples from seed tasks',
        'eval': 'Generated data quality, downstream performance'
    },
    'Mutation': {
        'desc': 'Random changes for exploration',
        'how_evolves': 'LLM-based semantic mutations create variants',
        'eval': 'Diversity, fitness improvement'
    },
    'Crossover': {
        'desc': 'Combines elements from multiple solutions',
        'how_evolves': 'Merges successful traits from different solutions',
        'eval': 'Offspring fitness vs parents'
    },
}


def create_visualization_html(methods: list, connections: list) -> str:
    """Create interactive D3.js visualization focused on evolution."""

    # Build nodes with paper details
    nodes = []
    for m in methods:
        info = METHOD_DESCRIPTIONS.get(m['display_name'], {})
        years = m.get('years', [])
        year_range = f"{min(years)}-{max(years)}" if years else "?"

        nodes.append({
            'id': m['display_name'],
            'category': m['category'],
            'frequency': m['frequency'],
            'years': years,
            'yearRange': year_range,
            'papers': m['papers'],  # Full paper details
            'desc': info.get('desc', ''),
            'howEvolves': info.get('how_evolves', ''),
            'eval': info.get('eval', '')
        })

    # Build links
    links = []
    for c in connections:
        links.append({
            'source': c['method1'],
            'target': c['method2'],
            'strength': c['strength'],
            'papers': c['co_occurrence_count']
        })

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Evolving Agents - Evolution Methods</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }}

        .header {{
            padding: 16px 32px;
            background: rgba(255,255,255,0.05);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header h1 {{
            font-size: 1.3rem;
            font-weight: 600;
        }}

        .header-stats {{
            display: flex;
            gap: 24px;
            font-size: 0.85rem;
            color: rgba(255,255,255,0.7);
        }}

        .header-stats span {{ color: #4ECDC4; font-weight: 600; }}

        .container {{
            display: flex;
            height: calc(100vh - 60px);
        }}

        .graph-container {{
            flex: 1;
            position: relative;
        }}

        #graph {{ width: 100%; height: 100%; }}

        .sidebar {{
            width: 340px;
            background: rgba(255,255,255,0.05);
            border-left: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .sidebar-section {{
            padding: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .sidebar-section h3 {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.5);
            margin-bottom: 12px;
        }}

        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 0.75rem;
            background: rgba(255,255,255,0.05);
            padding: 4px 8px;
            border-radius: 4px;
        }}

        .legend-color {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }}

        .method-detail {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }}

        .method-detail.empty {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: rgba(255,255,255,0.4);
            font-size: 0.9rem;
        }}

        .method-header {{
            margin-bottom: 16px;
        }}

        .method-header h2 {{
            font-size: 1.2rem;
            color: #4ECDC4;
            margin-bottom: 4px;
        }}

        .method-header .meta {{
            font-size: 0.8rem;
            color: rgba(255,255,255,0.6);
        }}

        .method-desc {{
            font-size: 0.85rem;
            color: rgba(255,255,255,0.8);
            margin-bottom: 16px;
            line-height: 1.5;
        }}

        .info-box {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
        }}

        .info-box h4 {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: rgba(255,255,255,0.5);
            margin-bottom: 6px;
        }}

        .info-box p {{
            font-size: 0.85rem;
            color: rgba(255,255,255,0.9);
        }}

        .info-box.evolves p {{ color: #96CEB4; }}
        .info-box.eval p {{ color: #45B7D1; }}

        .papers-section h4 {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: rgba(255,255,255,0.5);
            margin-bottom: 12px;
        }}

        .paper-card {{
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-left: 3px solid #4ECDC4;
        }}

        .paper-card .title {{
            font-size: 0.8rem;
            color: rgba(255,255,255,0.9);
            margin-bottom: 4px;
            line-height: 1.4;
        }}

        .paper-card .year {{
            font-size: 0.7rem;
            color: #4ECDC4;
            margin-bottom: 4px;
        }}

        .paper-card .how-enables {{
            font-size: 0.75rem;
            color: rgba(255,255,255,0.6);
            font-style: italic;
            line-height: 1.4;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.95);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.8rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 280px;
            z-index: 1000;
        }}

        .tooltip.visible {{ opacity: 1; }}
        .tooltip h5 {{ color: #4ECDC4; margin-bottom: 6px; }}
        .tooltip p {{ color: rgba(255,255,255,0.8); margin-bottom: 4px; }}

        .controls {{
            position: absolute;
            bottom: 16px;
            left: 16px;
        }}

        .controls button {{
            padding: 8px 14px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }}

        .controls button:hover {{
            background: rgba(255,255,255,0.2);
        }}

        .node {{ cursor: pointer; }}
        .node circle {{
            stroke: rgba(255,255,255,0.3);
            stroke-width: 2px;
            transition: all 0.2s;
        }}
        .node:hover circle, .node.selected circle {{
            stroke: #fff;
            stroke-width: 3px;
        }}
        .node.selected circle {{
            filter: drop-shadow(0 0 8px rgba(78, 205, 196, 0.6));
        }}
        .node text {{
            font-size: 9px;
            fill: rgba(255,255,255,0.85);
            pointer-events: none;
        }}
        .link {{
            stroke: rgba(255,255,255,0.12);
            stroke-width: 1px;
        }}
        .link.highlighted {{
            stroke: rgba(78, 205, 196, 0.5);
            stroke-width: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Evolution Methods for Self-Evolving Agents</h1>
        <div class="header-stats">
            <div><span>{len(nodes)}</span> methods</div>
            <div><span>{len(links)}</span> connections</div>
            <div><span>{sum(n['frequency'] for n in nodes)}</span> paper mentions</div>
        </div>
    </div>

    <div class="container">
        <div class="graph-container">
            <svg id="graph"></svg>
            <div class="tooltip" id="tooltip"></div>
            <div class="controls">
                <button onclick="resetZoom()">Reset View</button>
            </div>
        </div>

        <div class="sidebar">
            <div class="sidebar-section">
                <h3>Categories</h3>
                <div class="legend">
                    <div class="legend-item"><div class="legend-color" style="background: #FF6B6B"></div>Evolutionary</div>
                    <div class="legend-item"><div class="legend-color" style="background: #4ECDC4"></div>RL-Based</div>
                    <div class="legend-item"><div class="legend-color" style="background: #45B7D1"></div>Self-Improve</div>
                    <div class="legend-item"><div class="legend-color" style="background: #96CEB4"></div>Self-Train</div>
                    <div class="legend-item"><div class="legend-color" style="background: #FFEAA7"></div>Prompt Evo</div>
                    <div class="legend-item"><div class="legend-color" style="background: #DDA0DD"></div>Experience</div>
                    <div class="legend-item"><div class="legend-color" style="background: #FF8C42"></div>Meta-Learn</div>
                    <div class="legend-item"><div class="legend-color" style="background: #98D8C8"></div>Multi-Agent</div>
                </div>
            </div>

            <div class="method-detail empty" id="methodDetail">
                Click a node to see papers using this evolution method
            </div>
        </div>
    </div>

    <script>
        const data = {{
            nodes: {json.dumps(nodes)},
            links: {json.dumps(links)}
        }};

        const categoryColors = {json.dumps(CATEGORY_COLORS)};

        const svg = d3.select("#graph");
        const width = svg.node().parentNode.clientWidth;
        const height = svg.node().parentNode.clientHeight;
        svg.attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");
        const zoom = d3.zoom()
            .scaleExtent([0.3, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));
        svg.call(zoom);

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-250))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.frequency) * 6 + 15));

        const link = g.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-opacity", d => 0.15 + d.strength * 0.4);

        const node = g.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", d => Math.sqrt(d.frequency) * 5 + 6)
            .attr("fill", d => categoryColors[d.category] || categoryColors.other);

        node.append("text")
            .text(d => d.id)
            .attr("x", d => Math.sqrt(d.frequency) * 5 + 10)
            .attr("y", 3);

        const tooltip = d3.select("#tooltip");
        let selectedNode = null;

        node.on("mouseover", function(event, d) {{
            tooltip.classed("visible", true)
                .html(`<h5>${{d.id}}</h5>
                       <p>${{d.desc || 'Evolution method'}}</p>
                       <p><strong>Papers:</strong> ${{d.frequency}} | <strong>Years:</strong> ${{d.yearRange}}</p>`)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 10) + "px");
            link.classed("highlighted", l => l.source.id === d.id || l.target.id === d.id);
        }})
        .on("mouseout", function() {{
            tooltip.classed("visible", false);
            link.classed("highlighted", false);
        }})
        .on("click", function(event, d) {{
            // Deselect previous
            node.classed("selected", false);
            // Select this node
            d3.select(this).classed("selected", true);
            selectedNode = d;
            showMethodDetail(d);
        }});

        function showMethodDetail(d) {{
            const detail = document.getElementById("methodDetail");
            detail.classList.remove("empty");

            let papersHtml = d.papers.map(p => `
                <div class="paper-card">
                    <div class="title">${{p.title || 'Untitled'}}</div>
                    <div class="year">arXiv ${{p.arxiv_id}} (${{p.year || '?'}})</div>
                    ${{p.how_enables ? `<div class="how-enables">"${{p.how_enables}}..."</div>` : ''}}
                </div>
            `).join('');

            detail.innerHTML = `
                <div class="method-header">
                    <h2>${{d.id}}</h2>
                    <div class="meta">${{d.category}} | ${{d.frequency}} papers | ${{d.yearRange}}</div>
                </div>
                <div class="method-desc">${{d.desc || 'Evolution method for self-evolving agents.'}}</div>
                ${{d.howEvolves ? `<div class="info-box evolves"><h4>How it enables evolution</h4><p>${{d.howEvolves}}</p></div>` : ''}}
                ${{d.eval ? `<div class="info-box eval"><h4>How quality is evaluated</h4><p>${{d.eval}}</p></div>` : ''}}
                <div class="papers-section">
                    <h4>Papers using this method (${{d.papers.length}})</h4>
                    ${{papersHtml}}
                </div>
            `;
        }}

        simulation.on("tick", () => {{
            link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
        }}
        function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
        }}
        function resetZoom() {{
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }}
    </script>
</body>
</html>'''

    return html


def main():
    """Generate evolution methods visualization."""
    print("=" * 60)
    print("Generating evolution methods visualization")
    print("=" * 60)

    # Load evolution methods data
    methods_path = PROCESSED_DIR / "evolution_methods.json"
    connections_path = PROCESSED_DIR / "evolution_connections.json"

    if not methods_path.exists():
        print("Run extract_evolution_methods.py first.")
        return

    with open(methods_path) as f:
        methods = json.load(f)

    with open(connections_path) as f:
        connections = json.load(f)

    print(f"Loaded {len(methods)} methods, {len(connections)} connections")

    # Create output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html = create_visualization_html(methods, connections)

    output_path = OUTPUT_DIR / "index.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nSaved to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
