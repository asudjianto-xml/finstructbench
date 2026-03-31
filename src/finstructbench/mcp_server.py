"""
FinStructBench MCP Server — Expose DocumentGraph operations as tools.

Lets Claude (or any MCP client) perform deterministic graph traversal
instead of attempting structured retrieval from raw text.

Run:
    python -m finstructbench.mcp_server                 # stdio transport
    python -m finstructbench.mcp_server --transport sse  # SSE transport

Claude Code config (~/.claude/settings.json):
    {
      "mcpServers": {
        "finstructbench": {
          "command": "python",
          "args": ["-m", "finstructbench.mcp_server"],
          "env": {}
        }
      }
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from finstructbench.graph import DocumentGraph
from finstructbench.ingest import ingest_markdown
from finstructbench.instances import get_instance_path, list_instances

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "finstructbench",
    instructions=(
        "FinStructBench graph retrieval server. Provides deterministic, "
        "provably correct lookups over financial document knowledge graphs. "
        "Use these tools instead of trying to extract structured data from "
        "raw document text — the graph is the source of truth."
    ),
)

# Cache: doc_id -> (DocumentGraph, source_path)
_graphs: dict[str, tuple[DocumentGraph, str]] = {}


def _get_graph(doc_id: str) -> DocumentGraph:
    """Return a cached graph, loading from bundled instances if needed."""
    if doc_id not in _graphs:
        # Try bundled instances first
        available = list_instances()
        if doc_id in available:
            path = get_instance_path(doc_id)
            graph = ingest_markdown(path)
            _graphs[doc_id] = (graph, path)
        else:
            raise ValueError(
                f"Unknown document '{doc_id}'. "
                f"Bundled instances: {available}. "
                f"Use load_document to load a custom markdown file."
            )
    return _graphs[doc_id][0]


def _format_float(v: float) -> str:
    """Format a float preserving all significant digits."""
    # Use repr to avoid rounding, then strip trailing zeros
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s


# ---------------------------------------------------------------------------
# Management tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_documents() -> str:
    """List all available benchmark document instances.

    Returns the names of bundled financial report instances and any
    custom documents that have been loaded. Use these names as the
    doc_id parameter for other tools.
    """
    bundled = list_instances()
    loaded = list(_graphs.keys())
    custom = [d for d in loaded if d not in bundled]
    return json.dumps({
        "bundled_instances": bundled,
        "loaded": loaded,
        "custom_loaded": custom,
    }, indent=2)


@mcp.tool()
def load_document(path: str, doc_id: str | None = None) -> str:
    """Load a custom markdown financial document into the graph store.

    Args:
        path: Absolute path to a markdown file.
        doc_id: Optional identifier. Defaults to the filename stem.

    Returns:
        Graph statistics for the loaded document.
    """
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if doc_id is None:
        doc_id = p.stem
    graph = ingest_markdown(str(p))
    _graphs[doc_id] = (graph, str(p))
    stats = graph.stats()
    stats["doc_id"] = doc_id
    stats["source"] = str(p)
    return json.dumps(stats, indent=2)


@mcp.tool()
def graph_stats(doc_id: str) -> str:
    """Get statistics about a document's knowledge graph.

    Returns ENM entry counts by type, triple counts by relation,
    and available phase encoders. Useful for understanding the
    document's structure before querying.

    Args:
        doc_id: Document identifier (e.g. "model_validation").
    """
    graph = _get_graph(doc_id)
    stats = graph.stats()
    stats["doc_id"] = doc_id
    return json.dumps(stats, indent=2)


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

@mcp.tool()
def query_enm(
    doc_id: str,
    enm_type: str | None = None,
    entity_id: str | None = None,
) -> str:
    """Query the Exact Numeric Memory (ENM) store.

    Low-level tool for direct key-value lookups. Each ENM entry stores
    an exact numeric value with SHA-256 integrity verification.

    Args:
        doc_id: Document identifier.
        enm_type: Filter by ENM category (e.g. "capital_adequacy_ratios").
                  If None, returns all types.
        entity_id: Filter by entity ID (e.g. "CET1/Tier_1"). If None,
                   returns all entries for the type.

    Returns:
        Matching ENM entries as {type, id, value} objects.
    """
    graph = _get_graph(doc_id)

    if enm_type and entity_id:
        val = graph.lookup(enm_type, entity_id)
        if val is None:
            return json.dumps({"error": f"No entry for ({enm_type}, {entity_id})"})
        return json.dumps({"type": enm_type, "id": entity_id, "value": val})

    results = []
    for key, entry in graph.enm.items():
        if enm_type and key.type != enm_type:
            continue
        if entity_id and key.id != entity_id:
            continue
        results.append({
            "type": key.type,
            "id": key.id,
            "value": entry.value,
        })

    if not results:
        # List available types to help the caller
        types = sorted(set(k.type for k in graph.enm))
        return json.dumps({
            "error": "No matching entries",
            "available_enm_types": types,
        })
    return json.dumps(results, indent=2)


@mcp.tool()
def query_triples(
    doc_id: str,
    head: str | None = None,
    relation: str | None = None,
    tail: str | None = None,
    limit: int = 100,
) -> str:
    """Query the knowledge graph triples with pattern matching.

    Low-level tool for direct triple lookups. Supports wildcard
    matching — pass None for any position to match all values.

    Args:
        doc_id: Document identifier.
        head: Filter by head entity. None matches all.
        relation: Filter by relation type. None matches all.
        tail: Filter by tail value. None matches all.
        limit: Maximum results to return (default 100).

    Returns:
        Matching triples as [head, relation, tail] arrays.
    """
    graph = _get_graph(doc_id)
    matches = graph.query_triples(head=head, relation=relation, tail=tail)
    total = len(matches)
    truncated = total > limit
    matches = matches[:limit]
    return json.dumps({
        "total": total,
        "truncated": truncated,
        "triples": [[h, r, t] for h, r, t in matches],
    }, indent=2)


# ---------------------------------------------------------------------------
# Category 1: Exact Recall
# ---------------------------------------------------------------------------

@mcp.tool()
def exact_recall(doc_id: str, enm_type: str, entity_id: str) -> str:
    """Look up a single exact numeric value by key.

    Performs a deterministic ENM lookup with SHA-256 integrity
    verification. This is the ground-truth operation for the
    "exact recall" question category.

    Args:
        doc_id: Document identifier (e.g. "stress_test").
        enm_type: ENM category (e.g. "capital_projections").
        entity_id: Entity ID, possibly composite with "/" separators
                   (e.g. "Baseline/Q1_2026/CET1").

    Returns:
        The exact numeric value with full precision.
    """
    graph = _get_graph(doc_id)
    val = graph.lookup(enm_type, entity_id)
    if val is None:
        # Help the caller find the right key
        available = [
            {"id": k.id, "value": e.value}
            for k, e in graph.enm.items() if k.type == enm_type
        ]
        if not available:
            types = sorted(set(k.type for k in graph.enm))
            return json.dumps({
                "error": f"Unknown ENM type '{enm_type}'",
                "available_types": types,
            })
        return json.dumps({
            "error": f"No entry '{entity_id}' in type '{enm_type}'",
            "available_entries": available[:20],
        })
    return json.dumps({
        "enm_type": enm_type,
        "entity_id": entity_id,
        "value": val,
    })


# ---------------------------------------------------------------------------
# Category 2: Threshold Check
# ---------------------------------------------------------------------------

@mcp.tool()
def threshold_check(
    doc_id: str,
    enm_type: str,
    entity_id: str,
    threshold: float,
    operator: str = "ge",
) -> str:
    """Check whether a numeric value meets a threshold.

    Looks up the value from ENM and applies an inequality check
    via the document's phase encoder system. This is the ground-truth
    operation for "threshold" questions.

    Args:
        doc_id: Document identifier.
        enm_type: ENM category containing the value.
        entity_id: Entity ID for the value.
        threshold: The numeric threshold to compare against.
        operator: Comparison operator — "ge" (>=), "gt" (>),
                  "le" (<=), or "lt" (<).

    Returns:
        Whether the value satisfies the threshold, the actual value,
        and the margin (value - threshold).
    """
    graph = _get_graph(doc_id)
    val = graph.lookup(enm_type, entity_id)
    if val is None:
        return json.dumps({"error": f"No entry ({enm_type}, {entity_id})"})

    # Find a matching encoder, or do raw comparison
    ops = {"ge": lambda v, l: v >= l, "gt": lambda v, l: v > l,
           "le": lambda v, l: v <= l, "lt": lambda v, l: v < l}
    if operator not in ops:
        return json.dumps({"error": f"Unknown operator '{operator}'. Use: ge, gt, le, lt"})

    satisfied = ops[operator](val, threshold)
    return json.dumps({
        "entity_id": entity_id,
        "value": val,
        "threshold": threshold,
        "operator": operator,
        "satisfied": satisfied,
        "margin": val - threshold,
    })


# ---------------------------------------------------------------------------
# Category 3: Cross-Reference
# ---------------------------------------------------------------------------

@mcp.tool()
def cross_reference(
    doc_id: str,
    relation1: str,
    relation2: str,
) -> str:
    """Find entities appearing in two different relation types.

    Computes the set intersection of head entities across two
    relations. This is the ground-truth operation for
    "cross-reference" questions.

    Args:
        doc_id: Document identifier.
        relation1: First relation type (e.g. "has_risk_rating").
        relation2: Second relation type (e.g. "has_npl").

    Returns:
        The set of entities that participate in both relations.
    """
    graph = _get_graph(doc_id)
    heads1 = {h for h, _, _ in graph.query_triples(relation=relation1)}
    heads2 = {h for h, _, _ in graph.query_triples(relation=relation2)}
    overlap = sorted(heads1 & heads2)
    return json.dumps({
        "relation1": relation1,
        "relation2": relation2,
        "count": len(overlap),
        "entities": overlap,
        "relation1_total": len(heads1),
        "relation2_total": len(heads2),
    }, indent=2)


# ---------------------------------------------------------------------------
# Category 4: Counting
# ---------------------------------------------------------------------------

@mcp.tool()
def count_entities(
    doc_id: str,
    relation: str,
    tail: str | None = None,
    count_unique_heads: bool = False,
) -> str:
    """Count triples or unique entities matching a pattern.

    Three modes:
    - count_entities(doc, rel) — count all triples with this relation
    - count_entities(doc, rel, tail="X") — count entities where rel=X
    - count_entities(doc, rel, count_unique_heads=True) — count distinct heads

    This is the ground-truth operation for "counting" questions.

    Args:
        doc_id: Document identifier.
        relation: The relation type to count.
        tail: If provided, only count triples with this tail value.
        count_unique_heads: If True, count unique head entities instead
                           of total triples.

    Returns:
        The count and the matching entities/triples.
    """
    graph = _get_graph(doc_id)
    matches = graph.query_triples(relation=relation, tail=tail)

    if count_unique_heads:
        unique = sorted(set(h for h, _, _ in matches))
        return json.dumps({
            "relation": relation,
            "unique_head_count": len(unique),
            "heads": unique,
        }, indent=2)

    if tail:
        heads = sorted(set(h for h, _, _ in matches))
        return json.dumps({
            "relation": relation,
            "tail": tail,
            "count": len(matches),
            "matching_heads": heads,
        }, indent=2)

    return json.dumps({
        "relation": relation,
        "triple_count": len(matches),
    })


# ---------------------------------------------------------------------------
# Category 5: Contradiction Detection
# ---------------------------------------------------------------------------

@mcp.tool()
def find_contradictions(doc_id: str) -> str:
    """Detect pass/fail inconsistencies across segments.

    Identifies features that both pass and fail tests within the
    same group — a contradiction. This is the ground-truth operation
    for "contradiction" questions.

    Only applicable to documents with boolean (pass/fail) columns,
    such as model validation reports.

    Args:
        doc_id: Document identifier.

    Returns:
        Contradicting features with their pass/fail test details.
    """
    graph = _get_graph(doc_id)
    contradictions = graph.find_contradictions()

    if not contradictions:
        return json.dumps({
            "contradiction_count": 0,
            "features": [],
            "detail": "No pass/fail contradictions found.",
        })

    # Extract unique features
    features = set()
    for entity, p, f in contradictions:
        for test in [p, f]:
            parts = test.split("_", 1)
            if len(parts) >= 2:
                feat = parts[1].split(":")[0].split("/")[0]
                features.add(feat)

    return json.dumps({
        "contradiction_count": len(features),
        "features": sorted(features),
        "raw_contradictions": [
            {"entity": e, "passes": p, "fails": f}
            for e, p, f in contradictions
        ],
    }, indent=2)


# ---------------------------------------------------------------------------
# Category 6: Multi-Hop
# ---------------------------------------------------------------------------

@mcp.tool()
def multi_hop_argminmax(
    doc_id: str,
    enm_type: str,
    which: str = "lowest",
) -> str:
    """Find the entity with the min or max value in an ENM type.

    First hop of a multi-hop query: scans all entries in an ENM
    category and returns the extremum.

    Args:
        doc_id: Document identifier.
        enm_type: ENM category to search.
        which: "lowest" for argmin, "highest" for argmax.

    Returns:
        The entity ID and exact value of the extremum.
    """
    graph = _get_graph(doc_id)
    best_id = None
    best_val = None

    for key, entry in graph.enm.items():
        if key.type != enm_type:
            continue
        v = entry.value
        if best_val is None:
            best_id, best_val = key.id, v
        elif which == "lowest" and v < best_val:
            best_id, best_val = key.id, v
        elif which == "highest" and v > best_val:
            best_id, best_val = key.id, v

    if best_id is None:
        types = sorted(set(k.type for k in graph.enm))
        return json.dumps({
            "error": f"No entries for type '{enm_type}'",
            "available_types": types,
        })

    return json.dumps({
        "enm_type": enm_type,
        "which": which,
        "entity_id": best_id,
        "base_entity": best_id.split("/")[0],
        "value": best_val,
    })


@mcp.tool()
def multi_hop_chain(
    doc_id: str,
    source_type: str,
    target_type: str,
    which: str = "lowest",
) -> str:
    """Chained multi-hop query: argmin/max in one type, then lookup in another.

    Step 1: Find the entity with the min/max value in source_type.
    Step 2: Look up that entity's entries in target_type.

    This is the ground-truth operation for "multi-hop" questions —
    the category where LLMs score 0%.

    Args:
        doc_id: Document identifier.
        source_type: ENM type to find extremum in.
        target_type: ENM type to look up the entity in.
        which: "lowest" for argmin, "highest" for argmax.

    Returns:
        The entity name, its source value, and all target values.
    """
    graph = _get_graph(doc_id)

    # Step 1: find extremum in source
    best_id = None
    best_val = None
    for key, entry in graph.enm.items():
        if key.type != source_type:
            continue
        v = entry.value
        if best_val is None or \
           (which == "lowest" and v < best_val) or \
           (which == "highest" and v > best_val):
            best_id, best_val = key.id, v

    if best_id is None:
        return json.dumps({"error": f"No entries for source type '{source_type}'"})

    # Step 2: lookup base entity in target type
    base = best_id.split("/")[0]
    target_results = {}
    for key, entry in graph.enm.items():
        if key.type == target_type and (
            key.id == base or key.id.startswith(base + "/")
        ):
            target_results[key.id] = entry.value

    if not target_results:
        return json.dumps({
            "source_type": source_type,
            "which": which,
            "entity_id": best_id,
            "base_entity": base,
            "source_value": best_val,
            "error": f"No matching entries for '{base}' in target type '{target_type}'",
        })

    return json.dumps({
        "source_type": source_type,
        "target_type": target_type,
        "which": which,
        "entity_id": best_id,
        "base_entity": base,
        "source_value": best_val,
        "target_values": target_results,
    }, indent=2)


# ---------------------------------------------------------------------------
# Bonus: List available relations and ENM types for discovery
# ---------------------------------------------------------------------------

@mcp.tool()
def list_relations(doc_id: str) -> str:
    """List all relation types in the knowledge graph with counts.

    Useful for discovering what relations are available before
    calling cross_reference or count_entities.

    Args:
        doc_id: Document identifier.
    """
    graph = _get_graph(doc_id)
    rel_counts = {}
    for _, r, _ in graph.triples:
        rel_counts[r] = rel_counts.get(r, 0) + 1
    sorted_rels = sorted(rel_counts.items(), key=lambda x: -x[1])
    return json.dumps({
        "doc_id": doc_id,
        "total_triples": len(graph.triples),
        "relations": [{"name": r, "count": c} for r, c in sorted_rels],
    }, indent=2)


@mcp.tool()
def list_enm_types(doc_id: str) -> str:
    """List all ENM category types with entry counts.

    Useful for discovering what categories are available before
    calling exact_recall or multi_hop tools.

    Args:
        doc_id: Document identifier.
    """
    graph = _get_graph(doc_id)
    type_counts = {}
    for k in graph.enm:
        type_counts[k.type] = type_counts.get(k.type, 0) + 1
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
    return json.dumps({
        "doc_id": doc_id,
        "total_entries": len(graph.enm),
        "types": [{"name": t, "count": c} for t, c in sorted_types],
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server."""
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
