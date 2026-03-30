"""Counting — aggregation over triple sets and ENM entries."""

import random

from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


class CountingGenerator(QuestionGenerator):
    category = "counting"

    def sample(self, candidates: list[GeneratedQuestion],
               max_questions: int, seed: int = 42) -> list[GeneratedQuestion]:
        """Prioritize column-based counting questions.

        Column-based questions (with 'column' in metadata) match
        KG enrichment counts. Fill remaining slots with other types.
        """
        col_based = [q for q in candidates if q.metadata.get("column")]
        others = [q for q in candidates if not q.metadata.get("column")]
        rng = random.Random(seed)

        selected: list[GeneratedQuestion] = []
        if len(col_based) <= max_questions:
            selected.extend(col_based)
        else:
            selected.extend(rng.sample(col_based, max_questions))

        remaining = max_questions - len(selected)
        if remaining > 0 and others:
            selected.extend(rng.sample(others, min(remaining, len(others))))
        return selected

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
