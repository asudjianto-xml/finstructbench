"""Counting — aggregation over triple sets and ENM entries."""

import re
from statistics import median
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


class CountingGenerator(QuestionGenerator):
    category = "counting"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Count triples by relation
        relation_counts = {}
        for h, r, t in graph.triples:
            relation_counts.setdefault(r, 0)
            relation_counts[r] += 1

        # Skip noisy high-cardinality relations
        skip = {"in_section", "has_value", "has_effect"}

        for rel, count in relation_counts.items():
            if count < 2 or rel in skip:
                continue

            def make_fn(relation):
                def fn(g):
                    return len(g.query_triples(relation=relation))
                return fn

            q = GeneratedQuestion(
                qid=f"counting_{idx:03d}",
                category=self.category,
                natural_language=f"How many '{rel}' relationships exist in the report?",
                graph_answer_fn=make_fn(rel),
                ground_truth=count,
                llm_prompt=(
                    f"Count ALL instances where an entity has the '{rel}' "
                    f"relationship. Include every occurrence across all tables. "
                    f"{ANSWER_FORMATS['int']}"
                ),
                answer_type="int",
                metadata={"relation": rel},
            )

            if self._validate(graph, q):
                questions.append(q)
                idx += 1

        # Count by (relation, tail) — filtered counts
        relation_tail = {}
        for h, r, t in graph.triples:
            if r in skip:
                continue
            key = (r, t)
            relation_tail.setdefault(key, set()).add(h)

        for (rel, tail), heads in relation_tail.items():
            if len(heads) < 2 or len(heads) > 50:
                continue

            def make_filtered_fn(relation, tail_val):
                def fn(g):
                    return len(g.query_triples(relation=relation, tail=tail_val))
                return fn

            q = GeneratedQuestion(
                qid=f"counting_{idx:03d}",
                category=self.category,
                natural_language=f"How many entities have '{rel}' = '{tail}'?",
                graph_answer_fn=make_filtered_fn(rel, tail),
                ground_truth=len(heads),
                llm_prompt=(
                    f"Count how many entities are associated with "
                    f"'{tail}' for the '{rel}' category. "
                    f"{ANSWER_FORMATS['int']}"
                ),
                answer_type="int",
                metadata={"relation": rel, "tail": tail},
            )

            if self._validate(graph, q):
                questions.append(q)
                idx += 1

        # Count unique heads per relation
        relation_unique = {}
        for h, r, t in graph.triples:
            if r in skip:
                continue
            relation_unique.setdefault(r, set()).add(h)

        for rel, heads in relation_unique.items():
            if len(heads) < 3:
                continue

            def make_unique_fn(relation):
                def fn(g):
                    return len(set(h for h, _, _ in g.query_triples(relation=relation)))
                return fn

            q = GeneratedQuestion(
                qid=f"counting_{idx:03d}",
                category=self.category,
                natural_language=f"How many unique entities participate in '{rel}' relationships?",
                graph_answer_fn=make_unique_fn(rel),
                ground_truth=len(heads),
                llm_prompt=(
                    f"Count the number of UNIQUE entities that have a "
                    f"'{rel}' relationship in the report. "
                    f"{ANSWER_FORMATS['int']}"
                ),
                answer_type="int",
                metadata={"relation": rel},
            )

            if self._validate(graph, q):
                questions.append(q)
                idx += 1

        # ================================================================
        # Conditional count: entities with a KG relation whose numeric
        # value in a column is above the median for that column.
        #
        # "How many entities that 'passes' 'test_X' have AUC above the
        #  median AUC in model_validation?"
        # ================================================================

        # Build column index: (enm_type, column) → {base_entity: (eid, value)}
        by_type_col: dict[tuple[str, str], dict[str, tuple[str, float]]] = {}
        for key, entry in list(graph.enm.items()):
            base, col = _parse_column(key.id)
            if col is None or _is_skip(base):
                continue
            by_type_col.setdefault((key.type, col), {})[base] = (
                key.id, entry.value
            )

        # Build (relation, tail) → set of head entities
        rel_tail_heads: dict[tuple[str, str], set[str]] = {}
        for h, r, t in graph.triples:
            if r in skip:
                continue
            rel_tail_heads.setdefault((r, t), set()).add(h)

        for (rel, tail), heads in rel_tail_heads.items():
            if len(heads) < 3:
                continue

            for (enm_type, col), col_map in by_type_col.items():
                # Entities in both the relation set and this column
                overlap = heads & set(col_map.keys())
                if len(overlap) < 3:
                    continue

                # Compute median of the full column (not just overlap)
                all_vals = [v for _, v in col_map.values()]
                if len(all_vals) < 3:
                    continue
                med = median(all_vals)

                # Count entities in overlap that are above median
                count_above = sum(
                    1 for ent in overlap if col_map[ent][1] > med
                )
                if count_above < 1 or count_above == len(overlap):
                    continue  # trivial answer

                type_label = enm_type.replace("_", " ")

                def make_cond_fn(rl, tl, tp, cl, cm, skip_set):
                    def fn(g):
                        # Get heads for relation
                        triple_heads = set(
                            h for h, _, _ in g.query_triples(
                                relation=rl, tail=tl
                            )
                        )
                        # Compute median of full column
                        vals = []
                        entity_vals = {}
                        for k, entry in g.enm.items():
                            if k.type != tp:
                                continue
                            b, ec = _parse_column(k.id)
                            if ec != cl or _is_skip(b):
                                continue
                            vals.append(entry.value)
                            entity_vals[b] = entry.value
                        if len(vals) < 2:
                            return None
                        m = median(vals)
                        # Count overlap above median
                        return sum(
                            1 for ent in triple_heads
                            if ent in entity_vals and entity_vals[ent] > m
                        )
                    return fn

                q = GeneratedQuestion(
                    qid=f"counting_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"How many entities that have '{rel}' = '{tail}' "
                        f"also have '{col}' above the median in "
                        f"'{type_label}'?"
                    ),
                    graph_answer_fn=make_cond_fn(
                        rel, tail, enm_type, col, col_map, skip
                    ),
                    ground_truth=count_above,
                    llm_prompt=(
                        f"Step 1: Find all entities where '{rel}' = "
                        f"'{tail}'.\n"
                        f"Step 2: In the '{type_label}' section, compute "
                        f"the median of ALL '{col}' values.\n"
                        f"Step 3: Of the entities from Step 1, count how "
                        f"many have '{col}' strictly above that median.\n"
                        f"{ANSWER_FORMATS['int']}"
                    ),
                    answer_type="int",
                    metadata={
                        "relation": rel, "tail": tail,
                        "enm_type": enm_type, "column": col,
                        "pattern": "conditional_above_median",
                    },
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

        # Count unique entities per column header (matches KG enrichment)
        col_entities: dict[str, set[str]] = {}
        for key, meta in graph.enm_meta.items():
            col = meta.get("column", "")
            entity = meta.get("entity", "")
            if col and entity:
                col_entities.setdefault(col, set()).add(entity.lower())

        for col, entities in col_entities.items():
            if len(entities) < 2:
                continue

            count = len(entities)

            def make_col_fn(column):
                def fn(g):
                    ents = set()
                    for k, m in g.enm_meta.items():
                        if m.get("column") == column and m.get("entity"):
                            ents.add(m["entity"].lower())
                    return len(ents)
                return fn

            q = GeneratedQuestion(
                qid=f"counting_{idx:03d}",
                category=self.category,
                natural_language=(
                    f"How many unique entities have a '{col}' value "
                    f"in the report tables?"
                ),
                graph_answer_fn=make_col_fn(col),
                ground_truth=count,
                llm_prompt=(
                    f"Count the number of UNIQUE entities that have a "
                    f"'{col}' column value in the report tables. "
                    f"{ANSWER_FORMATS['int']}"
                ),
                answer_type="int",
                metadata={"relation": col, "column": col},
            )

            if self._validate(graph, q):
                questions.append(q)
                idx += 1

        return questions
