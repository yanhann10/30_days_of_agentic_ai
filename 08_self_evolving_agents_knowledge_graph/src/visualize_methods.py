#!/usr/bin/env python3
"""
Create interactive visualization of method-connection graph.
Deployable to Vercel as static HTML.
"""

import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "viz"

# Stopwords to filter out noisy methods
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
    'those', 'it', 'its', 'we', 'our', 'they', 'their', 'them',
    # Common generic words that aren't methods
    'large', 'small', 'new', 'different', 'various', 'multiple', 'several',
    'single', 'many', 'few', 'each', 'every', 'both', 'all', 'any', 'some',
    'other', 'only', 'same', 'such', 'more', 'most', 'less', 'least',
    'first', 'second', 'third', 'last', 'next', 'previous', 'following',
    'human', 'natural', 'language', 'model', 'models', 'learning',
    'task', 'tasks', 'data', 'dataset', 'datasets', 'input', 'output',
    'result', 'results', 'performance', 'accuracy', 'evaluation',
    'example', 'examples', 'sample', 'samples', 'test', 'tests',
    'training', 'train', 'trained', 'testing', 'validation',
    'based', 'using', 'used', 'use', 'given', 'given', 'simple',
    'complex', 'specific', 'general', 'standard', 'baseline',
    'proposed', 'existing', 'previous', 'recent', 'current', 'final',
    'initial', 'original', 'additional', 'further', 'experimental',
    'average', 'overall', 'total', 'individual', 'correct', 'incorrect',
    'similar', 'corresponding', 'related', 'common', 'main', 'primary',
    'high', 'low', 'good', 'better', 'best', 'bad', 'worse', 'worst',
    'effective', 'efficient', 'ablation', 'certain', 'respective',
    'scoring', 'step', 'steps', 'stage', 'stages', 'level', 'levels',
    'analysis', 'extensive', 'extensive', 'downstream', 'extensive',
    'arithmetic', 'zero-shot', 'few-shot', 'important', 'significant',
    'directly', 'independently', 'entire', 'separate', 'comprehensive',
    # More noisy words from extraction
    'selection', 'respect', 'genetic', 'reasoning', 'iterative', 'diverse',
    'base', 'code', 'greedy', 'improve', 'novel', 'generative', 'figure',
    'highest', 'generation', 'tuned', 'search', 'optimization', 'prompt',
    'task-specific', 'diverse', 'verifier', 'execution', 'section',
    'state-of-the-art', 'paper', 'method', 'approach', 'algorithm',
    'technique', 'framework', 'system', 'agent', 'process', 'mechanism',
    'component', 'module', 'architecture', 'structure', 'design',
    'implementation', 'solution', 'strategy', 'scheme', 'procedure',
    'operation', 'function', 'transformation', 'representation',
    'feature', 'parameter', 'configuration', 'setting', 'option',
    'choice', 'alternative', 'version', 'variant', 'type', 'kind',
    'form', 'format', 'style', 'manner', 'way', 'mode', 'pattern',
    # Model names (not methods)
    'alphacode', 'code-davinci-002', 'codex', 'gpt-3', 'gpt-4', 'gpt',
    'chatgpt', 'llama', 'palm', 'claude', 'gemini', 'mistral', 'falcon',
    'apis', 'api', 'prompting', 'prompted', 'finetuned', 'pretrained'
}

# Known good method terms to always include
KNOWN_METHODS = {
    'chain-of-thought', 'tree-of-thought', 'graph-of-thought',
    'retrieval-augmented generation', 'reinforcement learning',
    'supervised fine-tuning', 'proximal policy optimization',
    'direct preference optimization', 'reward model', 'self-training',
    'bootstrapping', 'curriculum learning', 'monte carlo tree search',
    'beam search', 'best-of-n', 'self-consistency', 'prompt tuning',
    'prompt optimization', 'evolutionary prompt', 'meta-prompt',
    'automatic prompt', 'memory bank', 'episodic memory', 'working memory',
    'long-term memory', 'context compression', 'tool learning', 'tool use',
    'api call', 'function calling', 'code execution', 'code interpreter',
    'multi-agent', 'agent collaboration', 'debate', 'consensus',
    'role-playing', 'delegation', 'coordinator', 'self-evolving',
    'self-improvement', 'meta-learning', 'neural architecture search',
    'automl', 'self-play', 'genetic algorithm', 'evolutionary',
    'mutation', 'react', 'self-refine', 'reflection', 'cot', 'tot',
    'rag', 'rlhf', 'ppo', 'dpo', 'sft', 'mcts', 'self-debug', 'self-edit',
    'tool creation', 'workflow memory', 'gist memory', 'experience replay'
}

# Categories and their colors
CATEGORY_COLORS = {
    'training': '#FF6B6B',      # Red
    'inference': '#4ECDC4',     # Teal
    'prompt': '#45B7D1',        # Blue
    'memory': '#96CEB4',        # Green
    'tool': '#FFEAA7',          # Yellow
    'multi-agent': '#DDA0DD',   # Plum
    'evolution': '#FF8C42',     # Orange
    'other': '#95A5A6'          # Gray
}

# Method descriptions and limitations for tooltips
METHOD_INFO = {
    'chain-of-thought': {
        'desc': 'Prompts LLM to reason step-by-step before answering',
        'enables': 'Complex reasoning, math problems, multi-step tasks',
        'limits': 'Increases latency, may hallucinate intermediate steps'
    },
    'tree-of-thought': {
        'desc': 'Explores multiple reasoning paths as a tree, backtracking when needed',
        'enables': 'Planning, puzzle solving, creative tasks',
        'limits': 'High compute cost, requires good evaluation heuristics'
    },
    'reinforcement learning': {
        'desc': 'Trains agent via reward signals from environment feedback',
        'enables': 'Behavior optimization, policy learning from experience',
        'limits': 'Reward hacking, sample inefficiency, unstable training'
    },
    'retrieval-augmented generation': {
        'desc': 'Retrieves relevant documents to augment LLM context',
        'enables': 'Up-to-date knowledge, reduced hallucination, citation',
        'limits': 'Retrieval quality bottleneck, context length limits'
    },
    'proximal policy optimization': {
        'desc': 'RL algorithm that constrains policy updates for stability',
        'enables': 'Stable RLHF training, fine-grained behavior control',
        'limits': 'Hyperparameter sensitive, compute intensive'
    },
    'direct preference optimization': {
        'desc': 'Trains on preference pairs without explicit reward model',
        'enables': 'Simpler RLHF pipeline, direct preference learning',
        'limits': 'Requires high-quality preference data'
    },
    'genetic algorithm': {
        'desc': 'Evolves population of solutions via selection/mutation',
        'enables': 'Prompt optimization, architecture search, exploration',
        'limits': 'Slow convergence, fitness function design critical'
    },
    'self-refine': {
        'desc': 'LLM iteratively critiques and improves its own output',
        'enables': 'Output quality improvement without training',
        'limits': 'May not fix fundamental errors, refinement loops'
    },
    'self-consistency': {
        'desc': 'Samples multiple reasoning paths, aggregates via voting',
        'enables': 'More robust answers, uncertainty estimation',
        'limits': 'Linear cost increase with samples'
    },
    'prompt tuning': {
        'desc': 'Optimizes soft prompts via gradient descent',
        'enables': 'Task adaptation without full fine-tuning',
        'limits': 'Requires gradient access, may not transfer'
    },
    'automatic prompt': {
        'desc': 'Auto-generates/optimizes prompts for specific tasks',
        'enables': 'Removes manual prompt engineering',
        'limits': 'Search space large, evaluation expensive'
    },
    'evolutionary': {
        'desc': 'Uses evolution-inspired operators for optimization',
        'enables': 'Gradient-free optimization, diverse solutions',
        'limits': 'Requires many evaluations, slow'
    },
    'mutation': {
        'desc': 'Random changes to solutions for exploration',
        'enables': 'Escaping local optima, diversity',
        'limits': 'Most mutations harmful, needs selection pressure'
    },
    'multi-agent': {
        'desc': 'Multiple specialized agents collaborate on tasks',
        'enables': 'Division of labor, diverse perspectives',
        'limits': 'Coordination overhead, error propagation'
    },
    'beam search': {
        'desc': 'Maintains top-k candidates during generation',
        'enables': 'Higher quality generation than greedy',
        'limits': 'Still local search, repetitive outputs'
    },
    'monte carlo tree search': {
        'desc': 'Tree search with random rollouts for evaluation',
        'enables': 'Strategic planning, game playing',
        'limits': 'High compute, needs good simulator/evaluator'
    },
    'react': {
        'desc': 'Interleaves reasoning and acting with observations',
        'enables': 'Grounded reasoning, tool use',
        'limits': 'Requires well-designed action space'
    },
    'debate': {
        'desc': 'Multiple agents argue different positions',
        'enables': 'Fact-checking, exploring counterarguments',
        'limits': 'May amplify confident wrong answers'
    },
    'self-debug': {
        'desc': 'LLM debugs its own code using error messages',
        'enables': 'Iterative code fixing without human',
        'limits': 'May not understand root cause'
    },
    'tool learning': {
        'desc': 'Agent learns to use external tools/APIs',
        'enables': 'Extended capabilities beyond LLM',
        'limits': 'Tool selection errors, API changes'
    },
    'memory bank': {
        'desc': 'Stores and retrieves past experiences',
        'enables': 'Long-term learning, personalization',
        'limits': 'Memory management, relevance decay'
    },
    'self-play': {
        'desc': 'Agent improves by playing against itself',
        'enables': 'Continuous improvement without new data',
        'limits': 'Can converge to local optima'
    },
    'meta-learning': {
        'desc': 'Learns how to learn across tasks',
        'enables': 'Fast adaptation to new tasks',
        'limits': 'Requires diverse task distribution'
    },
    'reflection': {
        'desc': 'Agent reflects on past actions to improve',
        'enables': 'Learning from mistakes, self-correction',
        'limits': 'Reflection quality depends on self-awareness'
    },
    'self-evolving': {
        'desc': 'Agent autonomously improves its own capabilities',
        'enables': 'Continuous improvement, adaptation',
        'limits': 'Safety concerns, goal drift'
    },
    'bootstrapping': {
        'desc': 'Uses own outputs as training data',
        'enables': 'Self-improvement without external data',
        'limits': 'Can amplify errors, needs filtering'
    },
    'curriculum learning': {
        'desc': 'Trains on progressively harder examples',
        'enables': 'Better generalization, stable training',
        'limits': 'Curriculum design non-trivial'
    },
    'tool creation': {
        'desc': 'Agent creates new tools for itself',
        'enables': 'Dynamic capability expansion',
        'limits': 'Tool quality varies, testing needed'
    },
    'workflow memory': {
        'desc': 'Stores successful action sequences',
        'enables': 'Reusable procedures, efficiency',
        'limits': 'May over-generalize workflows'
    },
    'experience replay': {
        'desc': 'Reuses past experiences for training',
        'enables': 'Sample efficiency, stable learning',
        'limits': 'Stale experiences, storage cost'
    }
}


def filter_methods(methods: list[dict], min_papers: int = 2) -> list[dict]:
    """Filter out noisy/generic methods, prioritize known methods."""
    known = []
    discovered = []

    for m in methods:
        name = m['name'].lower()

        # Skip stopwords and single-character methods
        if name in STOPWORDS or len(name) <= 2:
            continue

        # Skip methods appearing in too few papers
        if m['frequency'] < min_papers:
            continue

        # Prioritize known methods
        if name in KNOWN_METHODS:
            known.append(m)
        # Only include discovered methods with meaningful categories
        elif m['category'] != 'other':
            discovered.append(m)

    # Return known methods first, then discovered with good categories
    return known + discovered


def create_visualization_html(methods: list[dict], connections: list[dict], papers: list = None) -> str:
    """Create interactive D3.js visualization."""
    if papers is None:
        papers = []

    # Filter methods
    filtered_methods = filter_methods(methods, min_papers=2)
    method_names = {m['name'] for m in filtered_methods}

    # Filter connections to only include filtered methods
    filtered_connections = [
        c for c in connections
        if c['method1'] in method_names and c['method2'] in method_names
    ]

    # Take top 50 methods and their connections
    top_methods = filtered_methods[:50]
    top_method_names = {m['name'] for m in top_methods}

    top_connections = [
        c for c in filtered_connections
        if c['method1'] in top_method_names and c['method2'] in top_method_names
        and c['co_occurrence_count'] >= 1
    ][:100]  # Limit connections for readability

    # Build nodes and links for D3
    nodes = []
    for m in top_methods:
        info = METHOD_INFO.get(m['name'], {})
        nodes.append({
            'id': m['name'],
            'frequency': m['frequency'],
            'category': m['category'],
            'papers': m['papers'],
            'desc': info.get('desc', ''),
            'enables': info.get('enables', ''),
            'limits': info.get('limits', '')
        })

    links = []
    for c in top_connections:
        links.append({
            'source': c['method1'],
            'target': c['method2'],
            'strength': c['strength'],
            'papers': c['co_occurrence_count']
        })

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Evolving Agents - Method Connection Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }}

        .header {{
            padding: 20px 40px;
            background: rgba(255,255,255,0.05);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .header p {{
            font-size: 0.9rem;
            color: rgba(255,255,255,0.7);
        }}

        .container {{
            display: flex;
            height: calc(100vh - 100px);
        }}

        .graph-container {{
            flex: 1;
            position: relative;
        }}

        #graph {{
            width: 100%;
            height: 100%;
        }}

        .sidebar {{
            width: 280px;
            background: rgba(255,255,255,0.05);
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid rgba(255,255,255,0.1);
        }}

        .sidebar h3 {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.5);
            margin-bottom: 12px;
        }}

        .legend {{
            margin-bottom: 24px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }}

        .stats {{
            margin-bottom: 24px;
        }}

        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-size: 0.85rem;
        }}

        .stat-value {{
            font-weight: 600;
            color: #4ECDC4;
        }}

        .node-info {{
            display: none;
            padding: 16px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            margin-top: 16px;
        }}

        .node-info.active {{
            display: block;
        }}

        .node-info h4 {{
            font-size: 1rem;
            margin-bottom: 8px;
            color: #4ECDC4;
        }}

        .node-info p {{
            font-size: 0.8rem;
            color: rgba(255,255,255,0.7);
            margin-bottom: 4px;
        }}

        .node-info .desc {{
            font-style: italic;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .node-info .info-section {{
            margin-top: 8px;
        }}

        .node-info .info-label {{
            font-size: 0.75rem;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: block;
            margin-bottom: 2px;
        }}

        .node-info .info-value {{
            font-size: 0.85rem;
            color: rgba(255,255,255,0.9);
            display: block;
        }}

        .node-info .enables {{
            color: #96CEB4;
        }}

        .node-info .limits {{
            color: #FF6B6B;
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
            max-width: 320px;
            z-index: 1000;
        }}

        .tooltip.visible {{
            opacity: 1;
        }}

        .tooltip h5 {{
            margin-bottom: 6px;
            color: #4ECDC4;
            font-size: 0.95rem;
        }}

        .tooltip p {{
            margin-bottom: 4px;
            color: rgba(255,255,255,0.8);
        }}

        .tooltip .desc {{
            font-style: italic;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .tooltip .enables {{
            color: #96CEB4;
        }}

        .tooltip .limits {{
            color: #FF6B6B;
        }}

        .controls {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            display: flex;
            gap: 8px;
        }}

        .controls button {{
            padding: 8px 16px;
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

        .node {{
            cursor: pointer;
        }}

        .node circle {{
            stroke: rgba(255,255,255,0.3);
            stroke-width: 2px;
            transition: all 0.2s;
        }}

        .node:hover circle {{
            stroke: #fff;
            stroke-width: 3px;
        }}

        .node text {{
            font-size: 10px;
            fill: rgba(255,255,255,0.9);
            pointer-events: none;
        }}

        .link {{
            stroke: rgba(255,255,255,0.15);
            stroke-width: 1px;
        }}

        .link.highlighted {{
            stroke: rgba(78, 205, 196, 0.6);
            stroke-width: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Self-Evolving Agents - Method Connection Graph</h1>
        <p>Visualizing relationships between AI agent methods based on paper co-occurrence</p>
    </div>

    <div class="container">
        <div class="graph-container">
            <svg id="graph"></svg>
            <div class="tooltip" id="tooltip"></div>
            <div class="controls">
                <button onclick="resetZoom()">Reset View</button>
                <button onclick="toggleLabels()">Toggle Labels</button>
            </div>
        </div>

        <div class="sidebar">
            <div class="legend">
                <h3>Categories</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF6B6B"></div>
                    <span>Training</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4ECDC4"></div>
                    <span>Inference</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #45B7D1"></div>
                    <span>Prompt</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #96CEB4"></div>
                    <span>Memory</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FFEAA7"></div>
                    <span>Tool</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #DDA0DD"></div>
                    <span>Multi-Agent</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF8C42"></div>
                    <span>Evolution</span>
                </div>
            </div>

            <div class="stats">
                <h3>Statistics</h3>
                <div class="stat">
                    <span>Methods</span>
                    <span class="stat-value">{len(nodes)}</span>
                </div>
                <div class="stat">
                    <span>Connections</span>
                    <span class="stat-value">{len(links)}</span>
                </div>
                <div class="stat">
                    <span>Papers Analyzed</span>
                    <span class="stat-value">{len(papers)}</span>
                </div>
            </div>

            <div class="node-info" id="nodeInfo">
                <h4 id="nodeName"></h4>
                <p id="nodeCategory"></p>
                <p id="nodePapers"></p>
                <p id="nodeDesc" class="desc"></p>
                <div class="info-section">
                    <span class="info-label">Enables Evolution:</span>
                    <span id="nodeEnables" class="info-value enables"></span>
                </div>
                <div class="info-section">
                    <span class="info-label">Limitations:</span>
                    <span id="nodeLimits" class="info-value limits"></span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const data = {{
            nodes: {json.dumps(nodes)},
            links: {json.dumps(links)}
        }};

        const categoryColors = {json.dumps(CATEGORY_COLORS)};

        let showLabels = true;

        // Set up SVG
        const svg = d3.select("#graph");
        const width = svg.node().parentNode.clientWidth;
        const height = svg.node().parentNode.clientHeight;

        svg.attr("viewBox", [0, 0, width, height]);

        // Create container for zoom
        const g = svg.append("g");

        // Set up zoom
        const zoom = d3.zoom()
            .scaleExtent([0.3, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));

        svg.call(zoom);

        // Create force simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.frequency) * 8 + 20));

        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-opacity", d => 0.1 + d.strength * 0.5);

        // Create nodes
        const node = g.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Add circles to nodes
        node.append("circle")
            .attr("r", d => Math.sqrt(d.frequency) * 6 + 5)
            .attr("fill", d => categoryColors[d.category] || categoryColors.other);

        // Add labels to nodes
        node.append("text")
            .text(d => d.id)
            .attr("x", d => Math.sqrt(d.frequency) * 6 + 8)
            .attr("y", 4)
            .attr("class", "label");

        // Tooltip
        const tooltip = d3.select("#tooltip");

        node.on("mouseover", function(event, d) {{
            let tooltipHtml = `<h5>${{d.id}}</h5>`;
            if (d.desc) {{
                tooltipHtml += `<p class="desc">${{d.desc}}</p>`;
            }}
            tooltipHtml += `<p><strong>Category:</strong> ${{d.category}}</p>`;
            tooltipHtml += `<p><strong>Papers:</strong> ${{d.frequency}}</p>`;
            if (d.enables) {{
                tooltipHtml += `<p class="enables"><strong>Enables:</strong> ${{d.enables}}</p>`;
            }}
            if (d.limits) {{
                tooltipHtml += `<p class="limits"><strong>Limits:</strong> ${{d.limits}}</p>`;
            }}

            tooltip.classed("visible", true)
                .html(tooltipHtml)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 10) + "px");

            // Highlight connected links
            link.classed("highlighted", l => l.source.id === d.id || l.target.id === d.id);

            // Show node info in sidebar
            document.getElementById("nodeInfo").classList.add("active");
            document.getElementById("nodeName").textContent = d.id;
            document.getElementById("nodeCategory").textContent = "Category: " + d.category;
            document.getElementById("nodePapers").textContent = "Appears in " + d.frequency + " papers";
            document.getElementById("nodeDesc").textContent = d.desc || "No description available";
            document.getElementById("nodeEnables").textContent = d.enables || "-";
            document.getElementById("nodeLimits").textContent = d.limits || "-";
        }})
        .on("mouseout", function() {{
            tooltip.classed("visible", false);
            link.classed("highlighted", false);
        }});

        // Update positions on simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        // Control functions
        function resetZoom() {{
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity
            );
        }}

        function toggleLabels() {{
            showLabels = !showLabels;
            d3.selectAll(".label").style("opacity", showLabels ? 1 : 0);
        }}
    </script>
</body>
</html>'''

    return html


def main():
    """Generate visualization."""
    print("=" * 60)
    print("Generating method-connection visualization")
    print("=" * 60)

    # Load data
    methods_path = PROCESSED_DIR / "methods.json"
    connections_path = PROCESSED_DIR / "method_connections.json"
    papers_path = PROCESSED_DIR / "parsed_papers.json"

    if not methods_path.exists() or not connections_path.exists():
        print("Run extract_methods.py first.")
        return

    with open(methods_path) as f:
        methods = json.load(f)

    with open(connections_path) as f:
        connections = json.load(f)

    papers = []
    if papers_path.exists():
        with open(papers_path) as f:
            papers = json.load(f)

    print(f"Loaded {len(methods)} methods, {len(connections)} connections, {len(papers)} papers")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate HTML
    html = create_visualization_html(methods, connections, papers)

    # Save
    output_path = OUTPUT_DIR / "index.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nSaved visualization to {output_path}")
    print("Open in browser to view the interactive graph.")

    return output_path


if __name__ == "__main__":
    main()
