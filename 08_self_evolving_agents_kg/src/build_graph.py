#!/usr/bin/env python3
"""
Build a NetworkX knowledge graph from enriched paper data.

Input:
- data/processed/enriched_papers.json
- data/processed/concepts.json (optional)
- data/processed/paper_concepts.json (optional)
- data/processed/categories.json

Output:
- data/graph/graph.graphml
- data/graph/graph.json
"""

import json
from pathlib import Path
from typing import Optional

try:
    import networkx as nx
except ImportError:
    print("networkx not installed. Install with: pip install networkx")
    raise

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
GRAPH_DIR = BASE_DIR / "data" / "graph"


def load_json(path: Path) -> list[dict]:
    """Load JSON file if exists."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def safe_str(val, default='') -> str:
    """Convert value to string, handling None."""
    if val is None:
        return default
    return str(val)


def safe_int(val, default=0) -> int:
    """Convert value to int, handling None."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def build_paper_nodes(G: nx.DiGraph, papers: list[dict]):
    """Add paper nodes to graph."""
    for paper in papers:
        arxiv_id = paper.get('arxiv_id')
        if not arxiv_id:
            continue

        abstract = paper.get('abstract', '') or ''
        G.add_node(
            f"paper:{arxiv_id}",
            node_type="paper",
            title=safe_str(paper.get('title', '')),
            year=safe_int(paper.get('year'), 0),
            category=safe_str(paper.get('category', '')),
            subcategory=safe_str(paper.get('subcategory', '')),
            citation_count=safe_int(paper.get('citation_count'), 0),
            abstract=abstract[:500],
            authors=','.join(paper.get('authors', [])[:3]),  # First 3 authors
            url=safe_str(paper.get('url', '')),
        )


def build_category_nodes(G: nx.DiGraph, categories: list[dict]):
    """Add category nodes and hierarchy edges."""
    for cat in categories:
        name = cat.get('name', '')
        if not name:
            continue

        G.add_node(
            f"category:{name}",
            node_type="category",
            name=safe_str(name),
            level=safe_int(cat.get('level'), 1),
        )

        # Add hierarchy edge to parent
        parent = cat.get('parent')
        if parent:
            G.add_edge(
                f"category:{name}",
                f"category:{parent}",
                edge_type="CHILD_OF",
            )


def build_concept_nodes(G: nx.DiGraph, concepts: list[dict]):
    """Add concept nodes to graph."""
    for concept in concepts:
        name = concept.get('normalized_name', concept.get('name', ''))
        if not name:
            continue

        G.add_node(
            f"concept:{name}",
            node_type="concept",
            name=safe_str(name),
            concept_category=safe_str(concept.get('category', '')),
            mention_count=safe_int(concept.get('mention_count'), 0),
        )


def build_paper_category_edges(G: nx.DiGraph, papers: list[dict]):
    """Connect papers to their categories."""
    for paper in papers:
        arxiv_id = paper.get('arxiv_id')
        category = paper.get('category')

        if arxiv_id and category:
            paper_node = f"paper:{arxiv_id}"
            cat_node = f"category:{category}"

            if paper_node in G and cat_node in G:
                G.add_edge(
                    paper_node,
                    cat_node,
                    edge_type="BELONGS_TO",
                )


def build_citation_edges(G: nx.DiGraph, papers: list[dict]):
    """Add citation edges between papers."""
    # Build set of known paper IDs
    known_papers = {p['arxiv_id'] for p in papers if p.get('arxiv_id')}

    for paper in papers:
        arxiv_id = paper.get('arxiv_id')
        if not arxiv_id:
            continue

        paper_node = f"paper:{arxiv_id}"

        # References (papers this paper cites)
        for ref_id in paper.get('references', []):
            if ref_id in known_papers:
                ref_node = f"paper:{ref_id}"
                G.add_edge(
                    paper_node,
                    ref_node,
                    edge_type="CITES",
                )

        # Cited by (papers that cite this paper)
        for citer_id in paper.get('cited_by', []):
            if citer_id in known_papers:
                citer_node = f"paper:{citer_id}"
                G.add_edge(
                    citer_node,
                    paper_node,
                    edge_type="CITES",
                )


def build_paper_concept_edges(G: nx.DiGraph, paper_concepts: list[dict]):
    """Connect papers to concepts they mention."""
    for pc in paper_concepts:
        arxiv_id = pc.get('arxiv_id')
        if not arxiv_id:
            continue

        paper_node = f"paper:{arxiv_id}"
        if paper_node not in G:
            continue

        # Concepts
        for concept_name in pc.get('concepts', []):
            concept_node = f"concept:{concept_name}"
            if concept_node in G:
                G.add_edge(paper_node, concept_node, edge_type="MENTIONS")

        # Mechanisms
        for mech_name in pc.get('mechanisms', []):
            mech_node = f"concept:{mech_name}"
            if mech_node in G:
                G.add_edge(paper_node, mech_node, edge_type="USES_MECHANISM")

        # Benchmarks
        for bench_name in pc.get('benchmarks', []):
            bench_node = f"concept:{bench_name}"
            if bench_node in G:
                G.add_edge(paper_node, bench_node, edge_type="EVALUATED_ON")

        # Introduces
        for intro_name in pc.get('introduces', []):
            intro_node = f"concept:{intro_name}"
            if intro_node in G:
                G.add_edge(paper_node, intro_node, edge_type="INTRODUCES")


def compute_graph_stats(G: nx.DiGraph) -> dict:
    """Compute basic graph statistics."""
    # Count node types
    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get('node_type', 'unknown')
        node_types[ntype] = node_types.get(ntype, 0) + 1

    # Count edge types
    edge_types = {}
    for _, _, data in G.edges(data=True):
        etype = data.get('edge_type', 'unknown')
        edge_types[etype] = edge_types.get(etype, 0) + 1

    return {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'node_types': node_types,
        'edge_types': edge_types,
    }


def save_graph(G: nx.DiGraph):
    """Save graph in multiple formats."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # GraphML (for Gephi, Neo4j import, etc.)
    graphml_path = GRAPH_DIR / "graph.graphml"
    nx.write_graphml(G, graphml_path)
    print(f"Saved GraphML to {graphml_path}")

    # JSON (for web visualization, custom tools)
    json_path = GRAPH_DIR / "graph.json"
    graph_data = {
        'nodes': [
            {'id': node, **data}
            for node, data in G.nodes(data=True)
        ],
        'edges': [
            {'source': u, 'target': v, **data}
            for u, v, data in G.edges(data=True)
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"Saved JSON to {json_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Building Knowledge Graph")
    print("=" * 60)

    # Load data
    papers = load_json(PROCESSED_DIR / "enriched_papers.json")
    if not papers:
        # Fall back to basic papers.json
        papers = load_json(PROCESSED_DIR / "papers.json")

    categories = load_json(PROCESSED_DIR / "categories.json")
    concepts = load_json(PROCESSED_DIR / "concepts.json")
    paper_concepts = load_json(PROCESSED_DIR / "paper_concepts.json")

    print(f"Loaded: {len(papers)} papers, {len(categories)} categories, {len(concepts)} concepts")

    # Build graph
    G = nx.DiGraph()

    print("\nBuilding nodes...")
    build_paper_nodes(G, papers)
    build_category_nodes(G, categories)
    build_concept_nodes(G, concepts)

    print("Building edges...")
    build_paper_category_edges(G, papers)
    build_citation_edges(G, papers)
    build_paper_concept_edges(G, paper_concepts)

    # Compute statistics
    stats = compute_graph_stats(G)

    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print("\nNode types:")
    for ntype, count in stats['node_types'].items():
        print(f"  {ntype}: {count}")
    print("\nEdge types:")
    for etype, count in stats['edge_types'].items():
        print(f"  {etype}: {count}")

    # Save graph
    save_graph(G)

    # Find most connected papers
    print("\nMost connected papers (by degree):")
    paper_nodes = [(n, d) for n, d in G.degree() if n.startswith('paper:')]
    top_papers = sorted(paper_nodes, key=lambda x: x[1], reverse=True)[:10]
    for node, degree in top_papers:
        title = G.nodes[node].get('title', '')[:50]
        print(f"  {degree}: {title}...")

    return G


if __name__ == "__main__":
    main()
