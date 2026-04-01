"""Ranking — queries that require sorting entities by value.

Tests whether the LLM can identify the entity at a specific rank position
within a column.  Requires extracting all values, sorting, and returning
the k-th entity — a task that stresses both exhaustive scanning and
ordering accuracy.

Literature motivation:
  - Li et al. (2506.13405): LLMs fail at rank-based reasoning over tables
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


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][n % 10] if n % 10 < 4 else 'th'}"


class RankingGenerator(QuestionGenerator):
    category = "ranking"

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
            if len(entries) < 3:
                continue

            type_label = enm_type.replace("_", " ")

            # Sort descending (rank 1 = highest)
            sorted_desc = sorted(entries, key=lambda x: -x[2])

            # Check for duplicate values — skip entire column if any ties
            values = [v for _, _, v in sorted_desc]
            if len(set(values)) != len(values):
                continue

            # Generate rank-k queries for a few interesting positions
            n = len(sorted_desc)
            positions = set()
            positions.add(1)             # top
            positions.add(n)             # bottom
            if n >= 3:
                positions.add(2)         # second highest
            if n >= 5:
                positions.add(n // 2 + 1)  # median position

            for rank in sorted(positions):
                target_base = sorted_desc[rank - 1][0]
                target_val = sorted_desc[rank - 1][1]

                def make_fn(tp, c, r):
                    def fn(g):
                        vals = []
                        for k, entry in g.enm.items():
                            if k.type != tp:
                                continue
                            b, ec = _parse_column(k.id)
                            if ec != c or _is_skip(b):
                                continue
                            vals.append((b, entry.value))
                        ranked = sorted(vals, key=lambda x: -x[1])
                        if r <= len(ranked):
                            return ranked[r - 1][0]
                        return None
                    return fn

                q = GeneratedQuestion(
                    qid=f"ranking_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"Which entity has the {_ordinal(rank)} highest '{col}' "
                        f"in '{type_label}'?"
                    ),
                    graph_answer_fn=make_fn(enm_type, col, rank),
                    ground_truth=target_base,
                    llm_prompt=(
                        f"In the '{type_label}' section, rank all entities by their "
                        f"'{col}' values from highest to lowest.\n"
                        f"Which entity is ranked {_ordinal(rank)} (i.e., has the "
                        f"{_ordinal(rank)} highest value)?\n"
                        f"Report ONLY the entity name.\n"
                        f"{ANSWER_FORMATS['str']}"
                    ),
                    answer_type="str",
                    metadata={
                        "enm_type": enm_type, "column": col,
                        "rank": rank, "total_entities": n,
                        "expected_entity": target_base,
                        "expected_value": target_val,
                    },
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

        return questions
