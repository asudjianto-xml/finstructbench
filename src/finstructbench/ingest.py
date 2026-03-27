"""
Generic Markdown-to-Graph Ingester.

Parses any markdown document with tables into a DocumentGraph:
  - Every numeric cell becomes an ENM entry
  - Table structure becomes KG triples
  - Pass/fail, weak/strong columns become typed triples
  - Bullet-list key-value pairs become ENM entries
  - Cross-table entity references are detected automatically

Domain-agnostic: works with model validation reports, fair lending analyses,
stress test results, financial statements, etc.
"""

import re
from finstructbench.graph import DocumentGraph


# ============================================================================
# MARKDOWN PARSING
# ============================================================================

def parse_md_table(text: str) -> list[dict]:
    """Parse a markdown table into list of row dicts."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return []

    header_line = None
    sep_idx = 0
    for i, line in enumerate(lines):
        cleaned = line.replace("|", "").replace(" ", "").replace("-", "").replace(":", "")
        if "|" in line and cleaned == "":
            header_line = lines[i - 1] if i > 0 else None
            sep_idx = i
            break

    if header_line is None:
        return []

    def split_row(line):
        parts = line.split("|")
        cells = [c.strip() for c in parts]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        return cells

    headers = split_row(header_line)

    # Detect unnamed index column
    has_index = (headers[0] == "" or headers[0].strip() == "")

    rows = []
    for line in lines[sep_idx + 1:]:
        if not line.strip() or "|" not in line:
            break
        cells = split_row(line)

        if has_index and len(cells) == len(headers):
            row = dict(zip(headers[1:], cells[1:]))
            row["_index"] = cells[0]
        elif len(cells) >= len(headers):
            row = dict(zip(headers, cells[:len(headers)]))
        elif len(cells) > 0:
            padded = cells + [""] * (len(headers) - len(cells))
            row = dict(zip(headers, padded))
        else:
            continue
        rows.append(row)

    return rows


def parse_sections(text: str) -> dict[str, str]:
    """Parse markdown into hierarchical sections."""
    sections = {}
    current_h2 = None
    current_h3 = None
    current_content = []

    def save():
        if current_h2:
            key = current_h2
            if current_h3:
                key = f"{current_h2}/{current_h3}"
            content = "\n".join(current_content).strip()
            if content:
                sections[key] = content

    for line in text.split("\n"):
        if line.startswith("## "):
            save()
            current_h2 = line[3:].strip()
            current_h3 = None
            current_content = []
        elif line.startswith("### "):
            if current_h2 and current_h3 is None:
                sections[current_h2] = "\n".join(current_content).strip()
            elif current_h3:
                save()
            current_h3 = line[4:].strip()
            current_content = []
        else:
            current_content.append(line)

    save()
    return sections


def parse_bullet_kvs(text: str) -> dict[str, str]:
    """Extract key-value pairs from bullet lists (- Key: Value)."""
    kvs = {}
    for m in re.finditer(r"^[-*]\s+(.+?):\s+(.+)$", text, re.MULTILINE):
        kvs[m.group(1).strip()] = m.group(2).strip()
    return kvs


def try_float(s: str) -> float | None:
    """Try to parse a string as float."""
    if not s or s.strip().lower() in ("nan", "", "-"):
        return None
    try:
        return float(s.strip())
    except (ValueError, TypeError):
        return None


# ============================================================================
# COLUMN TYPE DETECTION
# ============================================================================

def classify_columns(rows: list[dict]) -> dict[str, str]:
    """Classify each column as: numeric, boolean, label, or entity.

    Returns dict of column_name -> type.
    """
    if not rows:
        return {}

    col_types = {}
    for col in rows[0].keys():
        if col.startswith("_"):
            continue

        values = [r.get(col, "").strip() for r in rows]
        non_empty = [v for v in values if v and v.lower() != "nan"]

        if not non_empty:
            col_types[col] = "empty"
            continue

        # Check boolean-like
        bool_values = {"true", "false", "yes", "no", "pass", "fail", "weak", "strong"}
        if all(v.lower() in bool_values for v in non_empty):
            col_types[col] = "boolean"
            continue

        # Check numeric
        numeric_count = sum(1 for v in non_empty if try_float(v) is not None)
        if numeric_count / len(non_empty) > 0.7:
            col_types[col] = "numeric"
            continue

        # Otherwise it's a label/entity
        col_types[col] = "label"

    return col_types


def detect_entity_column(rows: list[dict], col_types: dict) -> str | None:
    """Find the primary entity/label column in a table."""
    label_cols = [c for c, t in col_types.items() if t == "label"]

    # Prefer columns named "Feature", "Name", "Entity", "Model", etc.
    entity_names = {"feature", "name", "entity", "model", "segment", "cluster",
                    "scenario", "group", "subgroup", "category", "variable",
                    "factor", "portfolio", "product", "metric", "effect"}
    for col in label_cols:
        if col.lower() in entity_names:
            return col

    # First label column
    return label_cols[0] if label_cols else None


def detect_pass_fail_column(rows: list[dict], col_types: dict) -> str | None:
    """Find a pass/fail or weak/strong boolean column."""
    for col, ctype in col_types.items():
        if ctype == "boolean":
            return col
    return None


# ============================================================================
# PHASE ENCODER INFERENCE
# ============================================================================

# Common financial metric patterns and their ranges
METRIC_PATTERNS = {
    "AUC": (r"(?i)\bAUC\b", 0.0, 1.0),
    "accuracy": (r"(?i)\b(ACC|accuracy)\b", 0.0, 1.0),
    "AIR": (r"(?i)\bAIR\b", 0.0, 2.0),
    "ratio": (r"(?i)\bratio\b", 0.0, 5.0),
    "rate": (r"(?i)\b(rate|default.rate|approval.rate)\b", 0.0, 1.0),
    "importance": (r"(?i)\b(importance|score|weight)\b", 0.0, 1.0),
    "divergence": (r"(?i)\b(divergence|distance|JS|KL)\b", 0.0, 1.0),
    "loss": (r"(?i)\b(loss|logloss|brier)\b", 0.0, 2.0),
    "pvalue": (r"(?i)\bp.?value\b", 0.0, 1.0),
    "percentage": (r"(?i)\b(pct|percent|%)\b", 0.0, 100.0),
    "capital_ratio": (r"(?i)\b(CET1|Tier.?1|capital.ratio|leverage)\b", 0.0, 30.0),
}


def infer_phase_encoders(sections: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Infer phase encoders from column names across all tables."""
    encoders = {}
    all_text = " ".join(sections.values())

    for name, (pattern, vmin, vmax) in METRIC_PATTERNS.items():
        if re.search(pattern, all_text):
            encoders[name] = (vmin, vmax)

    return encoders


# ============================================================================
# GENERIC INGESTION
# ============================================================================

def ingest_table(graph: DocumentGraph, section_name: str, subsection: str | None,
                 rows: list[dict], col_types: dict):
    """Ingest a single table into the graph."""
    if not rows:
        return

    entity_col = detect_entity_column(rows, col_types)
    pass_fail_col = detect_pass_fail_column(rows, col_types)
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    label_cols = [c for c, t in col_types.items() if t == "label" and c != entity_col]

    # Build a clean section tag for ENM categories
    section_tag = re.sub(r"^\d+\.\s*", "", section_name).strip()
    section_tag = re.sub(r"[^a-zA-Z0-9_\s]", "", section_tag).strip()
    section_tag = section_tag.replace(" ", "_").lower()

    if subsection:
        sub_tag = re.sub(r"[^a-zA-Z0-9_\s]", "", subsection).strip()
        sub_tag = sub_tag.replace(" ", "_").lower()

    for i, row in enumerate(rows):
        # Determine entity name
        if entity_col and row.get(entity_col, "").strip():
            entity = row[entity_col].strip()
        elif subsection:
            entity = subsection
        else:
            entity = f"row_{i}"

        # Build composite entity ID for non-unique entities
        secondary_labels = []
        for lc in label_cols:
            val = row.get(lc, "").strip()
            if val:
                secondary_labels.append(val)

        if secondary_labels:
            entity_full = f"{entity}/{'/'.join(secondary_labels)}"
        else:
            entity_full = entity

        # Store numeric values as ENM entries
        for col in numeric_cols:
            val = try_float(row.get(col, ""))
            if val is not None:
                # ENM category = section_tag, entity_id = entity/column
                enm_id = f"{entity_full}/{col}" if len(numeric_cols) > 1 else entity_full
                graph.store_value(section_tag, enm_id, val)

                # Triple: entity has_<column> value
                rel = f"has_{col.replace(' ', '_').lower()}"
                graph.add_triple(entity, rel, str(val))

        # Store boolean/pass-fail as triples
        if pass_fail_col:
            pf_val = row.get(pass_fail_col, "").strip().lower()
            if pf_val in ("true", "yes", "weak", "fail"):
                graph.add_triple(entity_full, "fails", f"{section_tag}_{entity_full}")
                graph.add_triple(entity_full, "is_weak", section_tag)
            elif pf_val in ("false", "no", "strong", "pass"):
                graph.add_triple(entity_full, "passes", f"{section_tag}_{entity_full}")
                graph.add_triple(entity_full, "is_strong", section_tag)

        # Label columns become typed triples
        for lc in label_cols:
            val = row.get(lc, "").strip()
            if val:
                rel = f"has_{lc.replace(' ', '_').lower()}"
                graph.add_triple(entity, rel, val)

        # Entity participates in section
        graph.add_triple(entity, "in_section", section_tag)


def ingest_bullets(graph: DocumentGraph, section_name: str, kvs: dict[str, str]):
    """Ingest bullet-list key-value pairs."""
    section_tag = re.sub(r"^\d+\.\s*", "", section_name).strip()
    section_tag = section_tag.replace(" ", "_").lower()
    section_tag = re.sub(r"[^a-z0-9_]", "", section_tag)

    for key, value in kvs.items():
        # Try to store as numeric
        num = try_float(value)
        if num is not None:
            graph.store_value(section_tag, key, num)
            graph.add_triple(section_tag, f"has_{key.replace(' ', '_').lower()}", str(num))
        else:
            graph.add_triple(section_tag, f"has_{key.replace(' ', '_').lower()}", value)


def ingest_markdown(path: str, document_entity: str | None = None) -> DocumentGraph:
    """Ingest any markdown document into a DocumentGraph.

    Args:
        path: Path to markdown file.
        document_entity: Optional name for the document root entity.

    Returns:
        Populated DocumentGraph ready for benchmark generation.
    """
    with open(path) as f:
        text = f.read()

    graph = DocumentGraph()
    sections = parse_sections(text)

    # Extract document title
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if title_match:
        graph.metadata["title"] = title_match.group(1).strip()
    if document_entity:
        graph.metadata["document_entity"] = document_entity

    # Infer and register phase encoders
    encoder_ranges = infer_phase_encoders(sections)
    for name, (vmin, vmax) in encoder_ranges.items():
        graph.add_phase_encoder(name, vmin, vmax)

    # Process each section
    for section_key, content in sections.items():
        parts = section_key.split("/", 1)
        section_name = parts[0]
        subsection = parts[1] if len(parts) > 1 else None

        # Ingest tables
        # Find table blocks in content (lines starting with |)
        table_blocks = []
        current_block = []
        for line in content.split("\n"):
            if line.strip().startswith("|"):
                current_block.append(line)
            else:
                if current_block:
                    table_blocks.append("\n".join(current_block))
                    current_block = []
        if current_block:
            table_blocks.append("\n".join(current_block))

        for table_text in table_blocks:
            rows = parse_md_table(table_text)
            if rows:
                col_types = classify_columns(rows)
                ingest_table(graph, section_name, subsection, rows, col_types)

        # Ingest bullet-list KVs
        kvs = parse_bullet_kvs(content)
        if kvs:
            ingest_bullets(graph, section_name, kvs)

    return graph
