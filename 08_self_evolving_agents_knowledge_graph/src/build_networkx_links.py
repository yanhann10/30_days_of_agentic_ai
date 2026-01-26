#!/usr/bin/env python3
"""
Build inter-category method links and export to JSON.
These links are used by the D3 graph to show cross-cluster connections.
"""

import json
from pathlib import Path
import networkx as nx
from networkx.algorithms import bipartite


BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def paper_key(paper: dict) -> str:
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        return str(arxiv_id).strip()
    title = paper.get("title", "")
    return title.strip().lower()


def build_links(methods: list, min_sim: float = 0.02, top_k: int = 8) -> list[dict]:
    method_papers = {}
    method_category = {}
    for m in methods:
        papers = {paper_key(p) for p in m.get("papers", []) if paper_key(p)}
        method_papers[m["name"]] = papers
        method_category[m["name"]] = m.get("category", "Other")

    # Build bipartite graph: methods <-> papers
    b = nx.Graph()
    for m in methods:
        b.add_node(m["name"], bipartite=0, category=m["category"])
    for m in methods:
        for p in method_papers.get(m["name"], set()):
            b.add_node(p, bipartite=1)
            b.add_edge(m["name"], p)

    method_nodes = [m["name"] for m in methods]
    projected = bipartite.weighted_projected_graph(b, method_nodes)

    all_edges = []
    for a, b_name, data in projected.edges(data=True):
        ma = next(m for m in methods if m["name"] == a)
        mb = next(m for m in methods if m["name"] == b_name)
        if ma["category"] == mb["category"]:
            continue
        pa = method_papers.get(a, set())
        pb = method_papers.get(b_name, set())
        if not pa or not pb:
            continue
        inter = pa.intersection(pb)
        if not inter:
            continue
        union = pa.union(pb)
        sim = len(inter) / max(1, len(union))
        if sim <= 0:
            continue
        all_edges.append((a, b_name, sim))

    # Keep strongest inter-category edges per node
    per_node = {}
    for a, b, sim in all_edges:
        per_node.setdefault(a, []).append((b, sim))
        per_node.setdefault(b, []).append((a, sim))

    keep = set()
    for node, edges in per_node.items():
        edges = sorted(edges, key=lambda x: x[1], reverse=True)
        for other, sim in edges[:top_k]:
            if sim >= min_sim:
                keep.add(tuple(sorted([node, other])))

    links = []
    for a, b, sim in all_edges:
        if tuple(sorted([a, b])) in keep:
            links.append({
                "source": a,
                "target": b,
                "strength": round(sim, 4),
                "type": "inter"
            })

    # Category-to-category bridges based on shared papers
    category_papers = {}
    for method, papers in method_papers.items():
        cat = method_category.get(method, "Other")
        category_papers.setdefault(cat, set()).update(papers)

    cats = sorted(category_papers.keys())
    cat_edges = []
    for i, c1 in enumerate(cats):
        for c2 in cats[i + 1:]:
            p1 = category_papers.get(c1, set())
            p2 = category_papers.get(c2, set())
            if not p1 or not p2:
                continue
            inter = p1.intersection(p2)
            if not inter:
                continue
            union = p1.union(p2)
            strength = len(inter) / max(1, len(union))
            cat_edges.append((c1, c2, strength, len(inter)))

    for c1, c2, strength, co in cat_edges:
        links.append({
            "source": c1,
            "target": c2,
            "strength": round(strength, 4),
            "type": "cat-bridge",
            "co_occurrence": co
        })
    return links


def main():
    methods_path = PROCESSED_DIR / "finegrained_methods.json"
    if not methods_path.exists():
        print("Run extract_finegrained.py first.")
        return

    with open(methods_path, "r", encoding="utf-8") as f:
        methods = json.load(f)

    links = build_links(methods)
    output_path = PROCESSED_DIR / "networkx_links.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(links, f, indent=2)

    print(f"Saved {len(links)} inter-category links to {output_path}")


if __name__ == "__main__":
    main()
