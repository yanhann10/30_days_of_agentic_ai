#!/usr/bin/env python3
"""
Visualize evolution methods by METHOD TYPE.
Categories: RL-Based, Feedback-Based, Search-Based, Evolutionary Prompt, etc.
V3-style aesthetic: coherent clusters, sized nodes, proper legend.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "viz"

# Colors by evolution method type
CATEGORY_COLORS = {
    "RL-Based": "#E74C3C",           # Red
    "Feedback-Based": "#3498DB",      # Blue
    "Search-Based": "#9B59B6",        # Purple
    "Evolutionary Prompt": "#2ECC71", # Green
    "Text Gradient": "#1ABC9C",       # Teal
    "Bootstrapping": "#F39C12",       # Orange
    "Tool Evolution": "#E91E63",      # Pink
    "Memory Evolution": "#00BCD4",    # Cyan
    "Multi-Agent": "#FF9800",         # Amber
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
            'papers': m.get('papers', [])[:10],
            'color': CATEGORY_COLORS.get(m['category'], '#95A5A6')
        })

    links = [
        {'source': c['method1'], 'target': c['method2'], 'strength': c['strength']}
        for c in connections[:200]
    ]

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
            padding: 16px 24px;
            background: rgba(255,255,255,0.03);
            border-bottom: 1px solid rgba(255,255,255,0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 1.3rem;
            font-weight: 600;
            background: linear-gradient(135deg, #3498DB, #2ECC71);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header-controls {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .year-display {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #3498DB;
            min-width: 70px;
            text-align: center;
            font-family: 'SF Mono', Monaco, monospace;
        }}
        .play-btn {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: rgba(52,152,219,0.15);
            border: 2px solid rgba(52,152,219,0.4);
            color: #3498DB;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }}
        .play-btn:hover {{
            background: rgba(52,152,219,0.25);
            border-color: #3498DB;
        }}
        .play-btn svg {{ width: 18px; height: 18px; }}
        .stats {{
            display: flex;
            gap: 16px;
            font-size: 0.8rem;
            color: rgba(255,255,255,0.6);
        }}
        .stat-value {{ color: #3498DB; font-weight: 600; }}
        .container {{
            display: flex;
            height: calc(100vh - 65px);
        }}
        .graph-container {{
            flex: 1;
            position: relative;
        }}
        #graph {{ width: 100%; height: 100%; }}
        .sidebar {{
            width: 340px;
            background: rgba(0,0,0,0.3);
            border-left: 1px solid rgba(255,255,255,0.08);
            display: flex;
            flex-direction: column;
        }}
        .legend-section {{
            padding: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        .legend-title {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: rgba(255,255,255,0.4);
            margin-bottom: 10px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 0.7rem;
            background: rgba(255,255,255,0.05);
            padding: 5px 10px;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .legend-item:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .legend-item.active {{
            background: rgba(255,255,255,0.15);
        }}
        .legend-color {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .detail-panel {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }}
        .detail-empty {{
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: rgba(255,255,255,0.3);
            text-align: center;
            padding: 30px;
        }}
        .detail-empty p {{ font-size: 0.85rem; }}
        .method-card {{ animation: fadeIn 0.2s ease; }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .method-name {{
            font-size: 1.15rem;
            font-weight: 600;
            color: #3498DB;
            margin-bottom: 4px;
        }}
        .method-category {{
            display: inline-block;
            font-size: 0.65rem;
            color: rgba(255,255,255,0.9);
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 3px 8px;
            border-radius: 10px;
            margin-bottom: 14px;
        }}
        .info-box {{
            background: rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }}
        .info-box h4 {{
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.4);
            margin-bottom: 5px;
        }}
        .info-box p {{
            font-size: 0.85rem;
            line-height: 1.45;
            color: rgba(255,255,255,0.85);
        }}
        .info-box.how {{ border-left: 3px solid #2ECC71; }}
        .info-box.what {{ border-left: 3px solid #3498DB; }}
        .stats-row {{
            display: flex;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .stat-box {{
            flex: 1;
            background: rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }}
        .stat-box .num {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #3498DB;
        }}
        .stat-box .lbl {{
            font-size: 0.55rem;
            color: rgba(255,255,255,0.4);
            text-transform: uppercase;
        }}
        .papers-title {{
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.4);
            margin: 14px 0 8px;
        }}
        .paper-item {{
            background: rgba(255,255,255,0.03);
            border-left: 2px solid #3498DB;
            padding: 8px 10px;
            margin-bottom: 6px;
            border-radius: 0 6px 6px 0;
        }}
        .paper-title {{
            font-size: 0.72rem;
            color: rgba(255,255,255,0.85);
            line-height: 1.35;
        }}
        .paper-year {{
            font-size: 0.62rem;
            color: #3498DB;
            margin-top: 3px;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(10,10,20,0.95);
            border: 1px solid rgba(52,152,219,0.4);
            border-radius: 8px;
            padding: 12px;
            font-size: 0.78rem;
            pointer-events: none;
            opacity: 0;
            max-width: 280px;
            z-index: 1000;
            box-shadow: 0 6px 24px rgba(0,0,0,0.5);
        }}
        .tooltip.visible {{ opacity: 1; }}
        .tooltip h5 {{ color: #3498DB; margin-bottom: 4px; font-size: 0.88rem; }}
        .tooltip .cat {{ color: rgba(255,255,255,0.5); font-size: 0.65rem; margin-bottom: 5px; }}
        .tooltip .how {{ color: #2ECC71; font-style: italic; font-size: 0.72rem; }}
        .tooltip .freq {{ color: rgba(255,255,255,0.6); font-size: 0.68rem; margin-top: 4px; }}
        .controls {{
            position: absolute;
            bottom: 14px;
            left: 14px;
        }}
        .controls button {{
            padding: 7px 12px;
            background: rgba(52,152,219,0.12);
            border: 1px solid rgba(52,152,219,0.3);
            border-radius: 5px;
            color: #3498DB;
            cursor: pointer;
            font-size: 0.72rem;
        }}
        .node {{ cursor: pointer; }}
        .node circle {{
            stroke-width: 2px;
            transition: all 0.25s;
        }}
        .node.dimmed circle {{ opacity: 0.12; }}
        .node.dimmed text {{ opacity: 0.12; }}
        .node:hover circle, .node.selected circle {{
            stroke: #fff;
            stroke-width: 3px;
            filter: drop-shadow(0 0 12px rgba(52,152,219,0.6));
        }}
        .node.highlighted circle {{
            stroke: #fff;
            stroke-width: 3px;
            filter: drop-shadow(0 0 16px rgba(52,152,219,0.8));
        }}
        .node text {{
            font-size: 9px;
            fill: rgba(255,255,255,0.8);
            pointer-events: none;
            font-weight: 500;
        }}
        .category-node circle {{
            fill-opacity: 0.18;
            stroke-width: 2.5px;
        }}
        .category-node text {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            fill: rgba(255,255,255,0.7);
        }}
        .link {{
            stroke: rgba(255,255,255,0.08);
            stroke-width: 1px;
        }}
        .link.category-link {{
            stroke: rgba(255,255,255,0.18);
            stroke-dasharray: 2 3;
        }}
        .link.dimmed {{ opacity: 0.03; }}
        .link.highlighted {{
            stroke: rgba(52,152,219,0.4);
            stroke-width: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Self-Evolving Agent Methods</h1>
        <div class="header-controls">
            <div class="stats">
                <span><span class="stat-value">{len(nodes)}</span> methods</span>
                <span><span class="stat-value">{len(links)}</span> connections</span>
            </div>
            <button class="play-btn" id="playBtn" title="Animate by year">
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
                <div class="legend-title">Evolution Method Types</div>
                <div class="legend" id="legend"></div>
            </div>
            <div class="detail-panel" id="detailPanel">
                <div class="detail-empty">
                    <p>Click a method node to see<br>how it enables agent evolution</p>
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
        const categoryColors = {json.dumps(CATEGORY_COLORS)};
        const categories = Object.keys(categoryColors);
        const categoryNodes = categories.map(cat => ({{
            id: cat,
            category: cat,
            isCategory: true,
            frequency: 10,
            years: years,
            color: categoryColors[cat]
        }}));
        const categoryLinks = data.nodes.map(n => ({{
            source: n.id,
            target: n.category,
            strength: 0.9,
            type: "category"
        }}));
        const allNodes = [...categoryNodes, ...data.nodes];
        const allLinks = [...data.links, ...categoryLinks];
        const nodeById = new Map(allNodes.map(n => [n.id, n]));

        let isPlaying = false;
        let playInterval = null;
        let currentYearIdx = -1;
        let activeCategory = null;

        // Build legend
        const legendEl = document.getElementById("legend");
        Object.entries(categoryColors).forEach(([cat, color]) => {{
            const count = data.nodes.filter(n => n.category === cat).length;
            if (count > 0) {{
                const item = document.createElement("div");
                item.className = "legend-item";
                item.innerHTML = `<div class="legend-color" style="background:${{color}}"></div>${{cat}} (${{count}})`;
                item.onclick = () => filterByCategory(cat, item);
                legendEl.appendChild(item);
            }}
        }});

        function filterByCategory(cat, el) {{
            if (activeCategory === cat) {{
                activeCategory = null;
                document.querySelectorAll('.legend-item').forEach(e => e.classList.remove('active'));
                node.classed("dimmed", false);
                link.classed("dimmed", false);
            }} else {{
                activeCategory = cat;
                document.querySelectorAll('.legend-item').forEach(e => e.classList.remove('active'));
                el.classList.add('active');
                node.classed("dimmed", d => !d.isCategory && d.category !== cat);
                link.classed("dimmed", l => {{
                    const s = nodeById.get(l.source.id || l.source);
                    const t = nodeById.get(l.target.id || l.target);
                    if (!s || !t) return true;
                    return s.category !== cat && t.category !== cat;
                }});
            }}
        }}

        const svg = d3.select("#graph");
        const width = svg.node().parentNode.clientWidth;
        const height = svg.node().parentNode.clientHeight;
        svg.attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");
        const zoom = d3.zoom().scaleExtent([0.3, 4]).on("zoom", e => g.attr("transform", e.transform));
        svg.call(zoom);

        // Category centers to keep related methods clustered (reduce scattering)
        const centerRadius = Math.min(width, height) * 0.28;
        const categoryCenters = {{}};
        categories.forEach((cat, i) => {{
            const angle = (i / categories.length) * Math.PI * 2;
            categoryCenters[cat] = {{
                x: width / 2 + Math.cos(angle) * centerRadius,
                y: height / 2 + Math.sin(angle) * centerRadius
            }};
        }});

        // V3-style force simulation: coherent clusters with stronger local ties
        const simulation = d3.forceSimulation(allNodes)
            .force("link", d3.forceLink(allLinks).id(d => d.id)
                .distance(d => 120 - Math.min(1, d.strength ?? 0.5) * 70)
                .strength(d => 0.3 + (d.strength ?? 0.5) * 0.7))
            .force("charge", d3.forceManyBody().strength(-160))
            .force("center", d3.forceCenter(width/2, height/2))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.frequency)*4 + 14))
            .force("categoryX", d3.forceX(d => (categoryCenters[d.category]?.x ?? width/2)).strength(0.18))
            .force("categoryY", d3.forceY(d => (categoryCenters[d.category]?.y ?? height/2)).strength(0.18));

        const link = g.append("g").selectAll("line").data(allLinks).join("line")
            .attr("class", d => d.type === "category" ? "link category-link" : "link")
            .attr("stroke-opacity", d => 0.12 + (d.strength ?? 0.4)*0.3);

        const node = g.append("g").selectAll("g").data(allNodes).join("g")
            .attr("class", d => d.isCategory ? "node category-node" : "node")
            .call(d3.drag()
                .on("start", (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
                .on("drag", (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
                .on("end", (e,d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }}));

        // Sized nodes based on frequency
        node.append("circle")
            .attr("r", d => d.isCategory ? 22 : Math.sqrt(d.frequency)*4 + 5)
            .attr("fill", d => d.color)
            .attr("stroke", d => d.color);

        node.append("text")
            .text(d => d.id)
            .attr("x", d => d.isCategory ? 0 : Math.sqrt(d.frequency)*4 + 8)
            .attr("y", d => d.isCategory ? 4 : 3)
            .attr("text-anchor", d => d.isCategory ? "middle" : "start");

        const tooltip = d3.select("#tooltip");

        node.on("mouseover", (e, d) => {{
            if (d.isCategory) return;
            tooltip.classed("visible", true)
                .html(`<h5>${{d.id}}</h5><div class="cat">${{d.category}}</div><div class="how">${{d.how}}</div><div class="freq">${{d.frequency}} papers | ${{d.yearRange}}</div>`)
                .style("left", (e.pageX+12)+"px").style("top", (e.pageY-8)+"px");
            link.classed("highlighted", l => l.source.id===d.id || l.target.id===d.id);
        }}).on("mouseout", () => {{
            tooltip.classed("visible", false);
            link.classed("highlighted", false);
        }}).on("click", (e, d) => {{
            if (d.isCategory) return;
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
                    <div class="method-category" style="background:${{d.color}}">${{d.category}}</div>
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

        function resetZoom() {{ svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity); }}

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
                node.classed("dimmed", d => !d.isCategory && !d.years.includes(year));
                node.classed("highlighted", d => !d.isCategory && d.years.includes(year));
                link.classed("dimmed", l => {{
                    const s = nodeById.get(l.source.id || l.source);
                    const t = nodeById.get(l.target.id || l.target);
                    if (!s || !t) return true;
                    if (s.isCategory || t.isCategory) return false;
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
                        setTimeout(() => highlightYear(null), 1200);
                        currentYearIdx = -1;
                    }} else {{
                        highlightYear(years[currentYearIdx]);
                    }}
                }}, 1800);
            }}
        }});
    </script>
</body>
</html>'''
    return html


def main():
    print("=" * 60)
    print("Generating visualization (V3-style)")
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
