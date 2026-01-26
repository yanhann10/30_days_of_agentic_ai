#!/usr/bin/env python3
"""
Visualize FINE-GRAINED evolution methods.
- Categories from GitHub repo section subheaders
- Domain-specific methods with actual technique subnodes
- Dispersed layout, year-based animation
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "viz"

# Colors based on GitHub repo section structure
CATEGORY_COLORS = {
    # 1. Single-Agent Optimisation
    "LLM Behaviour Optimisation": "#4ECDC4",
    "Prompt Optimisation": "#FFEAA7",
    "Memory Optimization": "#96CEB4",
    "Tool Optimization": "#DDA0DD",
    # 2. Multi-Agent Optimisation
    "Multi-Agent Optimisation": "#98D8C8",
    # 3. Domain-Specific
    "Domain: Medical Diagnosis": "#FF6B6B",
    "Domain: Molecular Discovery": "#FF8C94",
    "Domain: Code Refinement": "#74B9FF",
    "Domain: Code Debugging": "#5F9EA0",
    "Domain: Scientific Research": "#A29BFE",
    "Domain: Finance": "#FD79A8",
    "Domain: Legal": "#E17055",
    # 4. Evaluation
    "Evaluation": "#45B7D1",
}


def create_html(methods: list, connections: list) -> str:
    all_years = set()
    for m in methods:
        all_years.update(m.get("years", []))
    all_years = sorted([y for y in all_years if y > 0])

    nodes = []
    for m in methods:
        year_range = f"{min(m['years'])}-{max(m['years'])}" if m.get('years') else "?"
        nodes.append({
            'id': m['name'],
            'category': m['category'],
            'how': m.get('how', ''),
            'what': m.get('what', ''),
            'frequency': m['frequency'],
            'years': m.get('years', []),
            'yearRange': year_range,
            'papers': m.get('papers', [])[:8],
            'color': CATEGORY_COLORS.get(m['category'], '#95A5A6')
        })

    links = [
        {'source': c['method1'], 'target': c['method2'], 'strength': c['strength']}
        for c in connections[:150]
    ]

    # Group categories for legend
    legend_groups = {
        "Single-Agent": ["LLM Behaviour Optimisation", "Prompt Optimisation", "Memory Optimization", "Tool Optimization"],
        "Multi-Agent": ["Multi-Agent Optimisation"],
        "Domain-Specific": ["Domain: Medical Diagnosis", "Domain: Molecular Discovery", "Domain: Code Refinement",
                           "Domain: Code Debugging", "Domain: Scientific Research", "Domain: Finance", "Domain: Legal"],
        "Evaluation": ["Evaluation"]
    }

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Evolving Agents - Evolution Methods Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a12 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: #fff;
        }}
        .header {{
            padding: 16px 28px;
            background: rgba(255,255,255,0.02);
            border-bottom: 1px solid rgba(255,255,255,0.06);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 1.2rem;
            font-weight: 500;
            color: rgba(255,255,255,0.9);
        }}
        .header h1 span {{
            background: linear-gradient(135deg, #4ECDC4, #44A08D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header-controls {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .year-display {{
            font-size: 2rem;
            font-weight: 700;
            color: #4ECDC4;
            min-width: 80px;
            text-align: center;
            font-family: 'SF Mono', Monaco, monospace;
        }}
        .play-btn {{
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: rgba(78,205,196,0.15);
            border: 2px solid rgba(78,205,196,0.4);
            color: #4ECDC4;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }}
        .play-btn:hover {{
            background: rgba(78,205,196,0.25);
            border-color: #4ECDC4;
        }}
        .play-btn svg {{
            width: 20px;
            height: 20px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            font-size: 0.75rem;
        }}
        .stat-value {{ color: #4ECDC4; font-weight: 600; }}
        .container {{
            display: flex;
            height: calc(100vh - 70px);
        }}
        .graph-container {{
            flex: 1;
            position: relative;
        }}
        #graph {{ width: 100%; height: 100%; }}
        .sidebar {{
            width: 380px;
            background: rgba(0,0,0,0.25);
            border-left: 1px solid rgba(255,255,255,0.06);
            display: flex;
            flex-direction: column;
        }}
        .legend-section {{
            padding: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            max-height: 200px;
            overflow-y: auto;
        }}
        .legend-group {{
            margin-bottom: 12px;
        }}
        .legend-group-title {{
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: rgba(255,255,255,0.35);
            margin-bottom: 6px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 0.65rem;
            background: rgba(255,255,255,0.04);
            padding: 4px 8px;
            border-radius: 12px;
        }}
        .legend-color {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
        }}
        .detail-panel {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}
        .detail-empty {{
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: rgba(255,255,255,0.25);
            text-align: center;
            padding: 40px;
        }}
        .method-card {{ animation: fadeIn 0.25s ease; }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .method-name {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #4ECDC4;
            margin-bottom: 4px;
        }}
        .method-category {{
            font-size: 0.7rem;
            color: rgba(255,255,255,0.4);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
        }}
        .info-box {{
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
        }}
        .info-box h4 {{
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.35);
            margin-bottom: 6px;
        }}
        .info-box p {{
            font-size: 0.85rem;
            line-height: 1.5;
            color: rgba(255,255,255,0.85);
        }}
        .info-box.how p {{ color: #96CEB4; }}
        .info-box.what p {{ color: #45B7D1; }}
        .stats-row {{
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
        }}
        .stat-box {{
            flex: 1;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }}
        .stat-box .num {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #4ECDC4;
        }}
        .stat-box .lbl {{
            font-size: 0.6rem;
            color: rgba(255,255,255,0.4);
            text-transform: uppercase;
        }}
        .papers-title {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.35);
            margin: 16px 0 10px;
        }}
        .paper-item {{
            background: rgba(255,255,255,0.02);
            border-left: 2px solid #4ECDC4;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-radius: 0 6px 6px 0;
        }}
        .paper-title {{
            font-size: 0.75rem;
            color: rgba(255,255,255,0.85);
            line-height: 1.4;
        }}
        .paper-year {{
            font-size: 0.65rem;
            color: #4ECDC4;
            margin-top: 4px;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.95);
            border: 1px solid rgba(78,205,196,0.3);
            border-radius: 10px;
            padding: 14px;
            font-size: 0.8rem;
            pointer-events: none;
            opacity: 0;
            max-width: 300px;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }}
        .tooltip.visible {{ opacity: 1; }}
        .tooltip h5 {{ color: #4ECDC4; margin-bottom: 6px; font-size: 0.9rem; }}
        .tooltip .cat {{ color: rgba(255,255,255,0.5); font-size: 0.7rem; margin-bottom: 6px; }}
        .tooltip p {{ color: rgba(255,255,255,0.7); margin-bottom: 4px; font-size: 0.75rem; }}
        .tooltip .how {{ color: #96CEB4; font-style: italic; }}
        .controls {{
            position: absolute;
            bottom: 16px;
            left: 16px;
        }}
        .controls button {{
            padding: 8px 14px;
            background: rgba(78,205,196,0.1);
            border: 1px solid rgba(78,205,196,0.3);
            border-radius: 6px;
            color: #4ECDC4;
            cursor: pointer;
            font-size: 0.75rem;
        }}
        .node {{ cursor: pointer; }}
        .node circle {{
            stroke-width: 1.5px;
            transition: all 0.3s;
        }}
        .node.dimmed circle {{ opacity: 0.15; }}
        .node.dimmed text {{ opacity: 0.15; }}
        .node:hover circle, .node.selected circle {{
            stroke: #fff;
            stroke-width: 2.5px;
            filter: drop-shadow(0 0 10px rgba(78,205,196,0.5));
        }}
        .node.highlighted circle {{
            stroke: #fff;
            stroke-width: 3px;
            filter: drop-shadow(0 0 15px rgba(78,205,196,0.7));
        }}
        .node text {{
            font-size: 8px;
            fill: rgba(255,255,255,0.75);
            pointer-events: none;
        }}
        .link {{
            stroke: rgba(255,255,255,0.06);
            stroke-width: 1px;
        }}
        .link.dimmed {{ opacity: 0.05; }}
        .link.highlighted {{
            stroke: rgba(78,205,196,0.35);
            stroke-width: 1.5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1><span>Evolution Methods</span> for Self-Evolving Agents</h1>
        <div class="header-controls">
            <div class="stats">
                <span><span class="stat-value">{len(nodes)}</span> methods</span>
                <span><span class="stat-value">{len(links)}</span> connections</span>
            </div>
            <button class="play-btn" id="playBtn" title="Play timeline">
                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
            </button>
            <div class="year-display" id="yearDisplay">ALL</div>
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
            <div class="legend-section">
                <div class="legend-group">
                    <div class="legend-group-title">Single-Agent Optimisation</div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-color" style="background:#4ECDC4"></div>LLM Behaviour</div>
                        <div class="legend-item"><div class="legend-color" style="background:#FFEAA7"></div>Prompt</div>
                        <div class="legend-item"><div class="legend-color" style="background:#96CEB4"></div>Memory</div>
                        <div class="legend-item"><div class="legend-color" style="background:#DDA0DD"></div>Tool</div>
                    </div>
                </div>
                <div class="legend-group">
                    <div class="legend-group-title">Multi-Agent</div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-color" style="background:#98D8C8"></div>MAS Optimisation</div>
                    </div>
                </div>
                <div class="legend-group">
                    <div class="legend-group-title">Domain-Specific</div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-color" style="background:#FF6B6B"></div>Medical</div>
                        <div class="legend-item"><div class="legend-color" style="background:#FF8C94"></div>Molecular</div>
                        <div class="legend-item"><div class="legend-color" style="background:#74B9FF"></div>Code Refine</div>
                        <div class="legend-item"><div class="legend-color" style="background:#5F9EA0"></div>Debugging</div>
                        <div class="legend-item"><div class="legend-color" style="background:#A29BFE"></div>Science</div>
                        <div class="legend-item"><div class="legend-color" style="background:#FD79A8"></div>Finance</div>
                        <div class="legend-item"><div class="legend-color" style="background:#E17055"></div>Legal</div>
                    </div>
                </div>
                <div class="legend-group">
                    <div class="legend-group-title">Evaluation</div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-color" style="background:#45B7D1"></div>Benchmarks</div>
                    </div>
                </div>
            </div>
            <div class="detail-panel" id="detailPanel">
                <div class="detail-empty">
                    <p>Click a method to see<br>HOW it enables evolution</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        const data = {{
            nodes: {json.dumps(nodes)},
            links: {json.dumps(links)}
        }};
        const years = {json.dumps(all_years)};
        let isPlaying = false;
        let playInterval = null;
        let currentYearIdx = -1;

        const svg = d3.select("#graph");
        const width = svg.node().parentNode.clientWidth;
        const height = svg.node().parentNode.clientHeight;
        svg.attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");
        const zoom = d3.zoom().scaleExtent([0.2, 5]).on("zoom", e => g.attr("transform", e.transform));
        svg.call(zoom);

        // Dispersed force simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(130).strength(0.25))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width/2, height/2))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.frequency)*3 + 30))
            .force("x", d3.forceX(width/2).strength(0.025))
            .force("y", d3.forceY(height/2).strength(0.025));

        const link = g.append("g").selectAll("line").data(data.links).join("line")
            .attr("class", "link").attr("stroke-opacity", d => 0.1 + d.strength*0.2);

        const node = g.append("g").selectAll("g").data(data.nodes).join("g").attr("class", "node")
            .call(d3.drag()
                .on("start", (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
                .on("drag", (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
                .on("end", (e,d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }}));

        node.append("circle")
            .attr("r", d => Math.sqrt(d.frequency)*3 + 6)
            .attr("fill", d => d.color)
            .attr("stroke", d => d.color);

        node.append("text")
            .text(d => d.id)
            .attr("x", d => Math.sqrt(d.frequency)*3 + 10)
            .attr("y", 3);

        const tooltip = d3.select("#tooltip");

        node.on("mouseover", (e, d) => {{
            tooltip.classed("visible", true)
                .html(`<h5>${{d.id}}</h5><div class="cat">${{d.category}}</div><p class="how">${{d.how}}</p><p>Papers: ${{d.frequency}} | Years: ${{d.yearRange}}</p>`)
                .style("left", (e.pageX+15)+"px").style("top", (e.pageY-10)+"px");
            link.classed("highlighted", l => l.source.id===d.id || l.target.id===d.id);
        }}).on("mouseout", () => {{
            tooltip.classed("visible", false);
            link.classed("highlighted", false);
        }}).on("click", (e, d) => {{
            node.classed("selected", false);
            d3.select(e.currentTarget).classed("selected", true);
            showDetail(d);
        }});

        function showDetail(d) {{
            const panel = document.getElementById("detailPanel");
            const papers = d.papers.map(p => `
                <div class="paper-item">
                    <div class="paper-title">${{p.title || 'Untitled'}}</div>
                    <div class="paper-year">${{p.year||'?'}} - arXiv:${{p.arxiv_id}}</div>
                </div>`).join('');

            panel.innerHTML = `
                <div class="method-card">
                    <div class="method-name">${{d.id}}</div>
                    <div class="method-category">${{d.category}}</div>
                    <div class="info-box how">
                        <h4>How It Works</h4>
                        <p>${{d.how}}</p>
                    </div>
                    <div class="info-box what">
                        <h4>What It Does</h4>
                        <p>${{d.what}}</p>
                    </div>
                    <div class="stats-row">
                        <div class="stat-box"><div class="num">${{d.frequency}}</div><div class="lbl">Papers</div></div>
                        <div class="stat-box"><div class="num">${{d.yearRange}}</div><div class="lbl">Years</div></div>
                    </div>
                    <div class="papers-title">Papers Using This Method</div>
                    ${{papers}}
                </div>`;
        }}

        simulation.on("tick", () => {{
            link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
            node.attr("transform", d=>`translate(${{d.x}},${{d.y}})`);
        }});

        function resetZoom() {{ svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity); }}

        // Year animation
        const playBtn = document.getElementById("playBtn");
        const yearDisplay = document.getElementById("yearDisplay");

        function highlightYear(year) {{
            if (year === null) {{
                yearDisplay.textContent = "ALL";
                node.classed("dimmed", false).classed("highlighted", false);
                link.classed("dimmed", false);
            }} else {{
                yearDisplay.textContent = year;
                node.classed("dimmed", d => !d.years.includes(year));
                node.classed("highlighted", d => d.years.includes(year));
                link.classed("dimmed", l => {{
                    const s = data.nodes.find(n => n.id === (l.source.id || l.source));
                    const t = data.nodes.find(n => n.id === (l.target.id || l.target));
                    return !s.years.includes(year) || !t.years.includes(year);
                }});
            }}
        }}

        playBtn.addEventListener("click", () => {{
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                playBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
                highlightYear(null);
                currentYearIdx = -1;
            }} else {{
                isPlaying = true;
                playBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>';
                currentYearIdx = 0;
                highlightYear(years[currentYearIdx]);
                playInterval = setInterval(() => {{
                    currentYearIdx++;
                    if (currentYearIdx >= years.length) {{
                        clearInterval(playInterval);
                        isPlaying = false;
                        playBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
                        setTimeout(() => highlightYear(null), 1500);
                        currentYearIdx = -1;
                    }} else {{
                        highlightYear(years[currentYearIdx]);
                    }}
                }}, 2000);
            }}
        }});
    </script>
</body>
</html>'''
    return html


def main():
    print("=" * 60)
    print("Generating fine-grained visualization")
    print("(Categories from GitHub repo section subheaders)")
    print("=" * 60)

    methods_path = PROCESSED_DIR / "finegrained_methods.json"
    connections_path = PROCESSED_DIR / "finegrained_connections.json"

    if not methods_path.exists():
        print("Run extract_finegrained.py first.")
        return

    with open(methods_path) as f:
        methods = json.load(f)
    with open(connections_path) as f:
        connections = json.load(f)

    print(f"Loaded {len(methods)} methods, {len(connections)} connections")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html = create_html(methods, connections)

    output_path = OUTPUT_DIR / "index.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
