"""Cross-Table Aggregation — numeric aggregation across ENM types.

Tests whether the LLM can aggregate values for the same entity across
different table sections (e.g., sum an entity's values from two categories).
Requires both locating the entity in multiple sections and performing
arithmetic.

Literature motivation:
  - Zhang et al. (2603.22608): cross-section reasoning as a distinct failure
"""

import re
from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS

_AGGREGATE_NAMES = re.compile(
    r"^(total|grand total|all|overall|combined|summary|cumulative|"
    r"aggregate|net|portfolio total|total top \d+)$",
    re.I,
)
_INTERNAL_ID = re.compile(r"^row_\d+$", re.I)


def _parse_column(entity_id: str) -> tuple[str, str | None]:
    parts = entity_id.split("/")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return entity_id, None


def _is_skip(name: str) -> bool:
    return bool(_AGGREGATE_NAMES.match(name.strip())) or bool(_INTERNAL_ID.match(name.strip()))


class CrossTableAggregationGenerator(QuestionGenerator):
    category = "cross_table_aggregation"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Index: entity_base → [(type, col, eid, value), ...]
        entity_index: dict[str, list[tuple[str, str, str, float]]] = {}
        for key, entry in list(graph.enm.items()):
            base, col = _parse_column(key.id)
            if col is None or _is_skip(base):
                continue
            entity_index.setdefault(base, []).append(
                (key.type, col, key.id, entry.value)
            )

        for entity, occurrences in entity_index.items():
            # Need entity in 2+ distinct types
            types_seen = set(t for t, _, _, _ in occurrences)
            if len(types_seen) < 2:
                continue

            # Group by type
            by_type: dict[str, list[tuple[str, str, float]]] = {}
            for t, c, eid, v in occurrences:
                by_type.setdefault(t, []).append((c, eid, v))

            type_list = sorted(by_type.keys())

            # Pattern 1: Sum of same-column values across types
            # Find columns that appear in multiple types for this entity
            col_to_type_vals: dict[str, list[tuple[str, str, float]]] = {}
            for t, entries in by_type.items():
                for c, eid, v in entries:
                    col_to_type_vals.setdefault(c, []).append((t, eid, v))

            for col, type_vals in col_to_type_vals.items():
                if len(type_vals) < 2:
                    continue

                # Sum across types
                total = sum(v for _, _, v in type_vals)
                type_labels = [t.replace("_", " ") for t, _, _ in type_vals]
                type_eids = [(t, eid) for t, eid, _ in type_vals]

                def make_sum_fn(ent, pairs):
                    def fn(g):
                        s = 0.0
                        for tp, eid in pairs:
                            val = g.lookup(tp, eid)
                            if val is None:
                                return None
                            s += val
                        return s
                    return fn

                sections = " and ".join(f"'{tl}'" for tl in type_labels)
                q = GeneratedQuestion(
                    qid=f"cross_table_aggregation_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"What is the sum of '{col}' values for '{entity}' "
                        f"across the {sections} sections?"
                    ),
                    graph_answer_fn=make_sum_fn(entity, type_eids),
                    ground_truth=total,
                    llm_prompt=(
                        f"Find '{entity}' in each of these sections: {sections}.\n"
                        f"Extract the '{col}' value from each section.\n"
                        f"Compute the sum of all these values.\n"
                        f"Report the exact total.\n"
                        f"{ANSWER_FORMATS['float']}"
                    ),
                    answer_type="float",
                    metadata={
                        "entity": entity, "column": col,
                        "types": [t for t, _, _ in type_vals],
                        "values": [v for _, _, v in type_vals],
                    },
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

            # Pattern 2: Difference between same entity in two types
            for i in range(len(type_list)):
                for j in range(i + 1, len(type_list)):
                    t1, t2 = type_list[i], type_list[j]
                    # Find matching columns
                    cols1 = {c: (eid, v) for c, eid, v in by_type[t1]}
                    cols2 = {c: (eid, v) for c, eid, v in by_type[t2]}
                    common_cols = set(cols1) & set(cols2)

                    for col in sorted(common_cols):
                        eid1, v1 = cols1[col]
                        eid2, v2 = cols2[col]
                        diff = v1 - v2

                        t1_label = t1.replace("_", " ")
                        t2_label = t2.replace("_", " ")

                        def make_diff_fn(tp1, e1, tp2, e2):
                            def fn(g):
                                va = g.lookup(tp1, e1)
                                vb = g.lookup(tp2, e2)
                                if va is None or vb is None:
                                    return None
                                return va - vb
                            return fn

                        q = GeneratedQuestion(
                            qid=f"cross_table_aggregation_{idx:03d}",
                            category=self.category,
                            natural_language=(
                                f"What is the difference in '{col}' for '{entity}' "
                                f"between '{t1_label}' and '{t2_label}'?"
                            ),
                            graph_answer_fn=make_diff_fn(t1, eid1, t2, eid2),
                            ground_truth=diff,
                            llm_prompt=(
                                f"Find '{entity}' in two sections:\n"
                                f"  1. '{t1_label}' — extract the '{col}' value\n"
                                f"  2. '{t2_label}' — extract the '{col}' value\n"
                                f"Compute: (value from '{t1_label}') − (value from '{t2_label}').\n"
                                f"Report the exact difference.\n"
                                f"{ANSWER_FORMATS['float']}"
                            ),
                            answer_type="float",
                            metadata={
                                "entity": entity, "column": col,
                                "type1": t1, "type2": t2,
                                "value1": v1, "value2": v2,
                            },
                        )

                        if self._validate(graph, q):
                            questions.append(q)
                            idx += 1

        return questions
