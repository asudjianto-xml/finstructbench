# FinStructBench

**Benchmark for Structured Information Retrieval from Financial Documents**

FinStructBench evaluates how well LLMs extract, aggregate and reason over structured data in financial documents. Questions are auto-generated from a document's knowledge graph, making ground truth **provably correct by construction**.

## Key Results

**LLM Evaluation** (454 questions, deterministic mode, 5 instances):

| Category | Graph Baseline | Sonnet 4 | Opus 4.6 | Opus Gain |
|---|---|---|---|---|
| Absence | 100% | 94% | 98% | +4pp |
| Threshold | 100% | 74% | 88% | +14pp |
| Cross-Reference | 100% | 58% | 76% | +18pp |
| Ranking | 100% | 66% | 74% | +8pp |
| Multi-Hop | 100% | 68% | 66% | -2pp |
| Exact Recall | 100% | 52% | 54% | +2pp |
| Numeric Comp. | 100% | 34% | 54% | +20pp |
| Cross-Table Agg. | 100% | 48% | 48% | 0pp |
| Counting | 100% | 32% | 38% | +6pp |
| Contradiction | 100% | 30% | 20% | -10pp |
| **Overall** | **100%** | **58%** | **65%** | **+7pp** |

Scaling from Sonnet to Opus improves numeric computation (+20pp) and cross-reference (+18pp), but categories requiring exhaustive enumeration (counting, cross-table aggregation) or global consistency checking (contradiction) remain **scale-resistant**.

**Ingestion Mode Comparison** (Claude Opus 4, 5 instances):

| Mode | ENM Entries | Triples (unique) | Regex Recovery |
|---|---|---|---|
| Regex (default) | 3,887 | 7,997 | --- |
| LLM-Only | 487 | 1,251 | 1.8% |
| Hybrid | 4,745 | 9,083 | 97.2% |

LLM-only extraction recovers only 12.5% of structured data. Hybrid ingestion preserves 100% of regex values exactly while adding 22% more entries from prose extraction.

## Installation

```bash
pip install finstructbench
```

For LLM evaluation (requires an Anthropic API key):

```bash
pip install finstructbench[llm]
```

## Quick Start

### Command Line

```bash
# List bundled financial report instances
finstructbench list-instances

# Run graph-only evaluation (no API calls)
finstructbench run finstructbench/instances/model_validation.md --no-llm

# Run with LLM evaluation
export ANTHROPIC_API_KEY=sk-ant-...
finstructbench run finstructbench/instances/fair_lending.md

# Customize
finstructbench run report.md --max-per-category 20 --model claude-sonnet-4-20250514 --seed 42

# Hybrid ingestion mode
finstructbench run report.md --ingest-mode hybrid
```

### Python API

```python
from finstructbench import Benchmark, get_instance_path

# Load a bundled instance
bench = Benchmark(get_instance_path("model_validation"))

# Graph-only evaluation
result = bench.run()
bench.print_results(result)

# With LLM
from finstructbench.llm_caller import create_client
client = create_client()
result = bench.run(llm_client=client)
bench.print_results(result)
bench.save_results(result, "results.json")
```

### Bring Your Own Document

FinStructBench works with any markdown document containing tables:

```python
from finstructbench import Benchmark

bench = Benchmark("path/to/your/report.md")
result = bench.run()
bench.print_results(result)
```

## Bundled Instances

| Instance | Regulatory Context | ENM | Triples | Questions | Categories |
|---|---|---|---|---|---|
| `model_validation` | SR 11-7 | 275 | 5,586 | 100 | 10 |
| `fair_lending` | ECOA / HMDA | 1,139 | 2,014 | 90 | 9 |
| `stress_test` | CCAR / DFAST | 476 | 1,417 | 90 | 9 |
| `credit_portfolio` | OCC Guidelines | 1,503 | 2,356 | 90 | 9 |
| `basel_capital` | Basel III Pillar 3 | 494 | 784 | 84 | 9 |
| **Total** | | **3,887** | **12,157** | **454** | |

## Question Categories

1. **Exact Recall** — Look up a single numeric value by key
2. **Threshold** — Check if a value meets a regulatory bound
3. **Cross-Reference** — Find entities appearing in two relation types
4. **Counting** — Count triples matching a pattern
5. **Contradiction** — Detect pass/fail inconsistencies across segments
6. **Multi-Hop** — Chained queries (argmin/argmax + cross-type lookup)
7. **Absence** — Verify that an entity or relation does not exist
8. **Ranking** — Rank entities by a numeric attribute
9. **Numeric Computation** — Compute ratios, differences or aggregates
10. **Cross-Table Aggregation** — Aggregate values across multiple tables

## How It Works

1. **Ingest**: Markdown document → `DocumentGraph` (ENM entries + KG triples + phase encoders)
2. **Generate**: Mine graph topology → questions with provably correct answers
3. **Evaluate**: Graph baseline (100% by construction) vs. LLM under test

The ingestion pipeline supports three modes:
- **Regex** (default): Deterministic regex extraction with SHA-256 integrity hashing
- **Hybrid**: Regex base + LLM fallback for ambiguous columns and prose extraction
- **LLM-Only**: Full LLM extraction (for research comparison only)

## MCP Server (Claude Integration)

FinStructBench includes an MCP server that exposes graph operations as tools, enabling Claude to perform **deterministic graph traversal** instead of extracting structured data from raw text.

### Installation

```bash
pip install finstructbench[mcp]
```

### Running the Server

```bash
# stdio transport (for Claude Code / desktop)
python -m finstructbench.mcp_server

# SSE transport (for web clients)
python -m finstructbench.mcp_server --transport sse

# Or via entry point
finstructbench-mcp
```

### Claude Code Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "finstructbench": {
      "command": "python",
      "args": ["-m", "finstructbench.mcp_server"]
    }
  }
}
```

### Available Tools

**Category-specific** (one per question type):

| Tool | Category | What it does |
|---|---|---|
| `exact_recall` | Exact Recall | Look up a single ENM value by key |
| `threshold_check` | Threshold | Check if a value meets a bound |
| `cross_reference` | Cross-Reference | Find entities in two relation types |
| `count_entities` | Counting | Count triples matching a pattern |
| `find_contradictions` | Contradiction | Detect pass/fail inconsistencies |
| `multi_hop_chain` | Multi-Hop | Chained argmin/max + cross-type lookup |
| `multi_hop_argminmax` | Multi-Hop (hop 1) | Find extremum within one ENM type |

**Low-level primitives:**

| Tool | What it does |
|---|---|
| `query_enm` | Direct ENM key-value lookup with filtering |
| `query_triples` | Pattern-match KG triples (head/relation/tail) |

**Discovery and management:**

| Tool | What it does |
|---|---|
| `list_documents` | Show available document instances |
| `load_document` | Ingest a custom markdown file |
| `list_enm_types` | Browse ENM categories with counts |
| `list_relations` | Browse KG relation types with counts |
| `graph_stats` | Full graph overview |

### Skill File

`SKILL.md` provides Claude with conventions for classifying questions into the ten categories, discovering document schema and composing tool calls. Place it in your project root or reference it in your Claude Code configuration.

## Extending

### Add a question generator

```python
from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion

class MyGenerator(QuestionGenerator):
    category = "my_category"

    def generate(self, graph):
        questions = []
        # Mine graph topology for questions...
        return questions
```

### Add a benchmark instance

Write a markdown document with tables and run:

```bash
finstructbench info my_report.md        # Check ingestion statistics
finstructbench run my_report.md --no-llm  # Verify graph baseline
```

## Citation

```bibtex
@article{sudjianto2026finstructbench,
  title={FinStructBench: A Benchmark for Structured Information Retrieval
         from Financial Documents Using Graph-Verifiable Questions},
  author={Sudjianto, Agus and Lau, Wingyan},
  year={2026}
}
```

## License

Apache 2.0
