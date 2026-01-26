#!/usr/bin/env python3
"""
Visualize the AUTHORITATIVE taxonomy of self-evolving agent methods.
Based on the landscape.png and README structure.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "viz"

# Colors matching the landscape.png visual
CATEGORY_COLORS = {
    "Single-Agent Optimisation": "#4ECDC4",  # Teal
    "Multi-Agent Optimisation": "#96CEB4",   # Green
    "Domain-Specific Optimisation": "#FF8C42",  # Orange
    "Evaluation": "#DDA0DD"  # Purple
}

# Subcategory colors
SUBCAT_COLORS = {
    "Training-Based": "#FF6B6B",
    "Test-Time (Inference)": "#45B7D1",
    "Prompt Optimisation": "#FFEAA7",
    "Memory Optimisation": "#98D8C8",
    "Tool Optimisation": "#C9B1FF",
    "MAS Optimisation": "#96CEB4",
    "Agent Collaboration": "#88D8B0",
    "Programming": "#FF6B6B",
    "Biomedicine": "#4ECDC4",
    "Scientific Research": "#45B7D1",
    "Financial & Legal": "#FFEAA7"
}


def create_html(methods: list, connections: list) -> str:
    """Create interactive visualization."""

    # Prepare nodes
    nodes = []
    for m in methods:
        year_range = f"{min(m['years'])}-{max(m['years'])}" if m['years'] else "?"
        color = CATEGORY_COLORS.get(m['category'], "#95A5A6")

        # Get papers with context
        papers_data = m.get('papers', [])[:10]  # Limit to 10

        nodes.append({
            'id': m['name'],
            'category': m['category'],
            'subcategory': m.get('subcategory', ''),
            'desc': m.get('desc', ''),
            'howEvolves': m.get('how_evolves', ''),
            'frequency': m['frequency'],
            'yearRange': year_range,
            'years': m.get('years', []),
            'papers': papers_data,
            'color': color
        })

    # Prepare links (limit for performance)
    links = [
        {'source': c['method1'], 'target': c['method2'], 'strength': c['strength']}
        for c in connections[:150]
    ]

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Evolving Agents - Evolution Methods Taxonomy</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }}
        .header {{
            padding: 20px 32px;
            background: rgba(255,255,255,0.03);
            border-bottom: 1px solid rgba(255,255,255,0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 1.4rem;
            font-weight: 600;
            background: linear-gradient(135deg, #4ECDC4, #44A08D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header-stats {{
            display: flex;
            gap: 20px;
            font-size: 0.8rem;
        }}
        .header-stats .stat {{
            text-align: center;
        }}
        .header-stats .stat-value {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #4ECDC4;
        }}
        .header-stats .stat-label {{
            color: rgba(255,255,255,0.5);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .container {{
            display: flex;
            height: calc(100vh - 80px);
        }}
        .graph-container {{
            flex: 1;
            position: relative;
            background: radial-gradient(ellipse at center, rgba(78,205,196,0.03) 0%, transparent 70%);
        }}
        #graph {{ width: 100%; height: 100%; }}
        .sidebar {{
            width: 380px;
            background: rgba(0,0,0,0.3);
            border-left: 1px solid rgba(255,255,255,0.08);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .sidebar-header {{
            padding: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        .sidebar-header h3 {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: rgba(255,255,255,0.4);
            margin-bottom: 16px;
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
            padding: 6px 10px;
            border-radius: 20px;
            transition: all 0.2s;
        }}
        .legend-item:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .legend-color {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
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
            color: rgba(255,255,255,0.3);
            text-align: center;
            padding: 40px;
        }}
        .detail-empty svg {{
            width: 48px;
            height: 48px;
            margin-bottom: 16px;
            opacity: 0.3;
        }}
        .method-card {{
            animation: fadeIn 0.3s ease;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .method-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #4ECDC4;
            margin-bottom: 4px;
        }}
        .method-path {{
            font-size: 0.75rem;
            color: rgba(255,255,255,0.4);
            margin-bottom: 16px;
        }}
        .method-desc {{
            font-size: 0.9rem;
            color: rgba(255,255,255,0.8);
            line-height: 1.6;
            margin-bottom: 20px;
        }}
        .info-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .info-card h4 {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.4);
            margin-bottom: 8px;
        }}
        .info-card p {{
            font-size: 0.85rem;
            line-height: 1.5;
        }}
        .info-card.evolves p {{ color: #96CEB4; }}
        .info-card.stats {{
            display: flex;
            gap: 20px;
        }}
        .info-card.stats .stat-item {{
            text-align: center;
        }}
        .info-card.stats .stat-num {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #4ECDC4;
        }}
        .info-card.stats .stat-lbl {{
            font-size: 0.65rem;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
        }}
        .papers-section {{
            margin-top: 20px;
        }}
        .papers-section h4 {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.4);
            margin-bottom: 12px;
        }}
        .paper-item {{
            background: rgba(255,255,255,0.02);
            border-left: 3px solid #4ECDC4;
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 0 8px 8px 0;
            transition: all 0.2s;
        }}
        .paper-item:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .paper-title {{
            font-size: 0.8rem;
            color: rgba(255,255,255,0.9);
            line-height: 1.4;
            margin-bottom: 4px;
        }}
        .paper-meta {{
            font-size: 0.7rem;
            color: #4ECDC4;
        }}
        .paper-context {{
            font-size: 0.75rem;
            color: rgba(255,255,255,0.5);
            font-style: italic;
            margin-top: 6px;
            line-height: 1.4;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.95);
            border: 1px solid rgba(78,205,196,0.3);
            border-radius: 12px;
            padding: 16px;
            font-size: 0.8rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 300px;
            z-index: 1000;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        .tooltip.visible {{ opacity: 1; }}
        .tooltip h5 {{
            color: #4ECDC4;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }}
        .tooltip p {{
            color: rgba(255,255,255,0.7);
            margin-bottom: 4px;
        }}
        .controls {{
            position: absolute;
            bottom: 20px;
            left: 20px;
        }}
        .controls button {{
            padding: 10px 18px;
            background: rgba(78,205,196,0.1);
            border: 1px solid rgba(78,205,196,0.3);
            border-radius: 8px;
            color: #4ECDC4;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }}
        .controls button:hover {{
            background: rgba(78,205,196,0.2);
        }}
        .node {{ cursor: pointer; }}
        .node circle {{
            stroke-width: 2px;
            transition: all 0.3s;
        }}
        .node:hover circle, .node.selected circle {{
            stroke: #fff;
            stroke-width: 3px;
            filter: drop-shadow(0 0 12px rgba(78,205,196,0.5));
        }}
        .node text {{
            font-size: 9px;
            fill: rgba(255,255,255,0.8);
            pointer-events: none;
        }}
        .link {{
            stroke: rgba(255,255,255,0.08);
            stroke-width: 1px;
        }}
        .link.highlighted {{
            stroke: rgba(78,205,196,0.4);
            stroke-width: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Self-Evolving Agents: Evolution Methods Taxonomy</h1>
        <div class="header-stats">
            <div class="stat">
                <div class="stat-value">{len(nodes)}</div>
                <div class="stat-label">Methods</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(links)}</div>
                <div class="stat-label">Connections</div>
            </div>
            <div class="stat">
                <div class="stat-value">50</div>
                <div class="stat-label">Papers</div>
            </div>
            <div class="stat">
                <div class="stat-value">2022-25</div>
                <div class="stat-label">Timeline</div>
            </div>
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
            <div class="sidebar-header">
                <h3>Taxonomy Categories</h3>
                <div class="legend">
                    <div class="legend-item"><div class="legend-color" style="background:#4ECDC4"></div>Single-Agent</div>
                    <div class="legend-item"><div class="legend-color" style="background:#96CEB4"></div>Multi-Agent</div>
                    <div class="legend-item"><div class="legend-color" style="background:#FF8C42"></div>Domain-Specific</div>
                    <div class="legend-item"><div class="legend-color" style="background:#DDA0DD"></div>Evaluation</div>
                </div>
            </div>
            <div class="detail-panel" id="detailPanel">
                <div class="detail-empty">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 8v4m0 4h.01"/>
                    </svg>
                    <p>Click on a method node to explore how it enables agent evolution</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        const data = {{
            nodes: {json.dumps(nodes)},
            links: {json.dumps(links)}
        }};
        const svg = d3.select("#graph");
        const width = svg.node().parentNode.clientWidth;
        const height = svg.node().parentNode.clientHeight;
        svg.attr("viewBox", [0, 0, width, height]);
        const g = svg.append("g");
        const zoom = d3.zoom().scaleExtent([0.2, 5]).on("zoom", e => g.attr("transform", e.transform));
        svg.call(zoom);
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(70))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width/2, height/2))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.frequency)*4 + 20));
        const link = g.append("g").selectAll("line").data(data.links).join("line").attr("class", "link").attr("stroke-opacity", d => 0.1 + d.strength*0.3);
        const node = g.append("g").selectAll("g").data(data.nodes).join("g").attr("class", "node")
            .call(d3.drag().on("start", (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
                         .on("drag", (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
                         .on("end", (e,d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }}));
        node.append("circle").attr("r", d => Math.sqrt(d.frequency)*4 + 8).attr("fill", d => d.color).attr("stroke", d => d.color);
        node.append("text").text(d => d.id).attr("x", d => Math.sqrt(d.frequency)*4 + 12).attr("y", 4);
        const tooltip = d3.select("#tooltip");
        node.on("mouseover", (e, d) => {{
            tooltip.classed("visible", true)
                .html(`<h5>${{d.id}}</h5><p>${{d.desc}}</p><p><strong>Papers:</strong> ${{d.frequency}} | <strong>Years:</strong> ${{d.yearRange}}</p>`)
                .style("left", (e.pageX+15)+"px").style("top", (e.pageY-10)+"px");
            link.classed("highlighted", l => l.source.id===d.id || l.target.id===d.id);
        }}).on("mouseout", () => {{ tooltip.classed("visible", false); link.classed("highlighted", false); }})
        .on("click", (e, d) => {{ node.classed("selected", false); d3.select(e.currentTarget).classed("selected", true); showDetail(d); }});
        function showDetail(d) {{
            const panel = document.getElementById("detailPanel");
            const papers = d.papers.map(p => `
                <div class="paper-item">
                    <div class="paper-title">${{p.title || 'Untitled'}}</div>
                    <div class="paper-meta">arXiv:${{p.arxiv_id}} (${{p.year||'?'}})</div>
                    ${{p.context ? `<div class="paper-context">"${{p.context.substring(0,150)}}..."</div>` : ''}}
                </div>`).join('');
            panel.innerHTML = `
                <div class="method-card">
                    <div class="method-title">${{d.id}}</div>
                    <div class="method-path">${{d.category}}${{d.subcategory ? ' > '+d.subcategory : ''}}</div>
                    <div class="method-desc">${{d.desc || 'Evolution method for self-evolving agents.'}}</div>
                    ${{d.howEvolves ? `<div class="info-card evolves"><h4>How It Enables Evolution</h4><p>${{d.howEvolves}}</p></div>` : ''}}
                    <div class="info-card stats">
                        <div class="stat-item"><div class="stat-num">${{d.frequency}}</div><div class="stat-lbl">Papers</div></div>
                        <div class="stat-item"><div class="stat-num">${{d.yearRange}}</div><div class="stat-lbl">Years</div></div>
                    </div>
                    <div class="papers-section">
                        <h4>Papers Using This Method</h4>
                        ${{papers}}
                    </div>
                </div>`;
        }}
        simulation.on("tick", () => {{
            link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
            node.attr("transform", d=>`translate(${{d.x}},${{d.y}})`);
        }});
        function resetZoom() {{ svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity); }}
    </script>
</body>
</html>'''
    return html


def main():
    print("=" * 60)
    print("Generating taxonomy visualization")
    print("=" * 60)

    methods_path = PROCESSED_DIR / "taxonomy_methods.json"
    connections_path = PROCESSED_DIR / "taxonomy_connections.json"

    if not methods_path.exists():
        print("Run extract_taxonomy.py first.")
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
