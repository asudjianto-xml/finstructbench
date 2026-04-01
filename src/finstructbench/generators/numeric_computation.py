"""Numeric Computation — arithmetic over pairs of extracted values.

Tests whether the LLM can extract two numeric values and compute a derived
quantity (sum, difference, ratio, percentage change).  The graph baseline
computes the answer deterministically from stored ENM entries.

Literature motivation:
  - Zhao et al. (2509.06332): LLMs struggle with multi-step numeric reasoning
  - Liu et al. (2505.23667): arithmetic over table cells is a key failure mode
"""

import re
from itertools import combinations
from statistics import mean, median
from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS

# Aggregate / summary row names — skip these as operands
_AGGREGATE_NAMES = re.compile(
    r"^(total|grand total|all|overall|combined|summary|cumulative|"
    r"aggregate|net|portfolio total|total top \d+)$",
    re.I,
)

# Internal synthetic entity names
_INTERNAL_ID = re.compile(r"^row_\d+$", re.I)

# Operations we generate
_OPS = [
    ("sum", "sum ({a} + {b})", lambda a, b: a + b),
    ("difference", "difference ({a} − {b})", lambda a, b: a - b),
    ("ratio", "ratio ({a} / {b})", lambda a, b: a / b if abs(b) > 1e-12 else None),
    ("pct_change", "percentage change from {a} to {b}", lambda a, b: ((b - a) / abs(a)) * 100 if abs(a) > 1e-12 else None),
]


def _parse_column(entity_id: str) -> tuple[str, str | None]:
    parts = entity_id.split("/")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return entity_id, None


def _is_skip(name: str) -> bool:
    return bool(_AGGREGATE_NAMES.match(name.strip())) or bool(_INTERNAL_ID.match(name.strip()))


class NumericComputationGenerator(QuestionGenerator):
    category = "numeric_computation"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Group ENM entries by (type, column)
        by_type_col: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
        for key, entry in list(graph.enm.items()):
            base, col = _parse_column(key.id)
            if col is None or _is_skip(base):
                continue
            by_type_col.setdefault((key.type, col), []).append(
                (base, key.id, entry.value)
            )

        for (enm_type, col), entries in by_type_col.items():
            if len(entries) < 2:
                continue

            type_label = enm_type.replace("_", " ")

            # Generate pairwise arithmetic questions
            for (base_a, eid_a, val_a), (base_b, eid_b, val_b) in combinations(entries, 2):
                for op_name, op_template, op_fn in _OPS:
                    result = op_fn(val_a, val_b)
                    if result is None:
                        continue

                    desc = op_template.format(a=base_a, b=base_b)

                    def make_fn(tp, ea, eb, fn):
                        def answer(g):
                            va = g.lookup(tp, ea)
                            vb = g.lookup(tp, eb)
                            if va is None or vb is None:
                                return None
                            return fn(va, vb)
                        return answer

                    q = GeneratedQuestion(
                        qid=f"numeric_computation_{idx:03d}",
                        category=self.category,
                        natural_language=(
                            f"What is the {desc} for '{col}' "
                            f"in '{type_label}'?"
                        ),
                        graph_answer_fn=make_fn(enm_type, eid_a, eid_b, op_fn),
                        ground_truth=result,
                        llm_prompt=(
                            f"In the '{type_label}' section, find the '{col}' values "
                            f"for '{base_a}' and '{base_b}'.\n"
                            f"Compute the {op_name}: "
                            + (f"'{base_a}' + '{base_b}'" if op_name == "sum"
                               else f"'{base_a}' − '{base_b}'" if op_name == "difference"
                               else f"'{base_a}' / '{base_b}'" if op_name == "ratio"
                               else f"('{base_b}' − '{base_a}') / |'{base_a}'| × 100")
                            + f".\nReport the exact result.\n"
                            f"{ANSWER_FORMATS['float']}"
                        ),
                        answer_type="float",
                        metadata={
                            "op": op_name, "enm_type": enm_type, "column": col,
                            "entity_a": base_a, "entity_b": base_b,
                        },
                    )

                    if self._validate(graph, q):
                        questions.append(q)
                        idx += 1

        # ================================================================
        # N-ary aggregates: mean and median over all entities in a column
        #
        # Tests extraction completeness — the LLM must find ALL values
        # in a column, not just a pair, then perform the aggregation.
        # ================================================================
        _AGG_OPS = [
            ("mean", "mean (average)", mean),
            ("median", "median", median),
        ]

        for (enm_type, col), entries in by_type_col.items():
            if len(entries) < 3:
                continue

            type_label = enm_type.replace("_", " ")
            values = [v for _, _, v in entries]

            for agg_name, agg_desc, agg_fn in _AGG_OPS:
                result = agg_fn(values)

                def make_agg_fn(tp, c, fn):
                    def answer(g):
                        vals = []
                        for k, entry in g.enm.items():
                            if k.type != tp:
                                continue
                            b, ec = _parse_column(k.id)
                            if ec != c or _is_skip(b):
                                continue
                            vals.append(entry.value)
                        if len(vals) < 2:
                            return None
                        return fn(vals)
                    return answer

                entity_names = sorted(b for b, _, _ in entries)

                q = GeneratedQuestion(
                    qid=f"numeric_computation_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"What is the {agg_desc} of '{col}' across all "
                        f"entities in '{type_label}'?"
                    ),
                    graph_answer_fn=make_agg_fn(enm_type, col, agg_fn),
                    ground_truth=result,
                    llm_prompt=(
                        f"In the '{type_label}' section, find the '{col}' "
                        f"value for EVERY entity (there are "
                        f"{len(entries)}).\n"
                        f"Compute the {agg_desc} of all these values.\n"
                        f"Report the exact result.\n"
                        f"{ANSWER_FORMATS['float']}"
                    ),
                    answer_type="float",
                    metadata={
                        "op": agg_name, "enm_type": enm_type,
                        "column": col, "n_entities": len(entries),
                        "entities": entity_names,
                    },
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

        return questions
