# FinStructBench — Structured Financial Document Retrieval

You have access to the FinStructBench MCP server, which provides deterministic
graph-based retrieval over financial documents. **Always use these tools for
structured data extraction** — never try to answer numeric, counting, or
cross-reference questions by reading raw document text.

## When to use these tools

Use FinStructBench tools whenever a user asks about:
- Specific numeric values in financial reports (capital ratios, AUC, AIR, loss rates, etc.)
- Whether a metric meets a regulatory threshold
- Which entities share properties across different categories
- Counting entities, relationships, or occurrences
- Pass/fail consistency across segments
- Finding min/max values and looking up related data

## Question classification

When you receive a question about a financial document, classify it into one of
these six categories and use the corresponding tool:

| Question pattern | Category | Tool to use |
|---|---|---|
| "What is the value of X?" | Exact Recall | `exact_recall` |
| "Does X meet threshold Y?" | Threshold | `threshold_check` |
| "Which entities appear in both A and B?" | Cross-Reference | `cross_reference` |
| "How many entities have X?" | Counting | `count_entities` |
| "Which features pass and fail?" | Contradiction | `find_contradictions` |
| "Find the lowest X, then look up its Y" | Multi-Hop | `multi_hop_chain` |

## Workflow

### Step 1: Identify the document
If the user mentions a document type, map it to a `doc_id`:
- Model validation / SR 11-7 → `model_validation`
- Fair lending / ECOA / HMDA → `fair_lending`
- Stress test / CCAR / DFAST → `stress_test`
- Credit portfolio / OCC → `credit_portfolio`
- Basel capital / Pillar 3 → `basel_capital`

If unsure, call `list_documents` to see what's available.

### Step 2: Discover the schema
Before querying, understand the document's structure:
- `list_enm_types(doc_id)` — shows available numeric categories and counts
- `list_relations(doc_id)` — shows available relation types and counts
- `graph_stats(doc_id)` — full overview including phase encoders

### Step 3: Execute the query
Use the category-specific tool. If you need flexibility, use the low-level
primitives:
- `query_enm(doc_id, type, id)` — direct key-value lookup
- `query_triples(doc_id, head, relation, tail)` — pattern-match triples

### Step 4: Format the answer
Present results clearly with:
- The exact numeric value (all decimal places, no rounding)
- The source category and entity ID for traceability
- For boolean results, state the threshold and whether it was met

## Multi-hop strategy

For questions requiring chained lookups:
1. **Prefer `multi_hop_chain`** — it does both hops atomically
2. If the chain is non-standard, decompose manually:
   - Call `multi_hop_argminmax` to find the extremum
   - Use the returned `base_entity` to call `exact_recall` or `query_enm`

## Key conventions

- **ENM types** are derived from section headings (e.g., `capital_adequacy_ratios`,
  `loan_loss_projections`). They use underscores, lowercase.
- **Entity IDs** can be composite: `Entity/Label1/Label2/Column`. The `/` separator
  encodes table structure (row labels + column name).
- **Relations** follow the pattern `has_<column_name>` for numeric columns,
  `passes`/`fails` for boolean columns, `in_section` for section membership.
- **Phase encoders** define regulatory thresholds. Common ones:
  - AIR ≥ 0.80 (four-fifths rule)
  - Capital ratio ≥ 4.5% (CET1 minimum)
  - AUC ≥ 0.5 (better than random)
  - P-value < 0.05 (statistical significance)

## Error handling

If a tool returns an `error` field:
- `"No entry for (type, id)"` — the entity_id or enm_type is wrong.
  Check `list_enm_types` and `query_enm(doc_id, type)` to browse available entries.
- `"No matching entries"` — the triple pattern matched nothing.
  Use `list_relations` to discover available relations.
- Never guess or approximate. If the exact key is unclear, use discovery tools
  to find the correct key, then retry.
