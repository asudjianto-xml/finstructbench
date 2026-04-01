"""Contradiction — detects pass/fail inconsistencies in the graph."""

from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS


def _extract_contradicting_entities(contradictions):
    """Extract unique entity names from contradiction tuples.

    Returns the actual entity names (e.g., "Inquiry", "Mortgage") rather
    than section-prefixed feature names. This aligns with KG enrichment
    which tracks contradictions per entity.
    """
    return {entity for entity, _p, _f in contradictions}


class ContradictionGenerator(QuestionGenerator):
    category = "contradiction"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        contradictions = graph.find_contradictions()
        if not contradictions:
            return questions

        entities_with_contradictions = _extract_contradicting_entities(contradictions)

        # Q: How many entities have contradictions?
        def make_count_fn():
            def fn(g):
                contras = g.find_contradictions()
                return len(_extract_contradicting_entities(contras))
            return fn

        questions.append(GeneratedQuestion(
            qid=f"contradiction_{idx:03d}",
            category=self.category,
            natural_language=(
                "How many entities have contradictory performance — "
                "passing in some segments but failing in others?"
            ),
            graph_answer_fn=make_count_fn(),
            ground_truth=len(entities_with_contradictions),
            llm_prompt=(
                "Examine all per-entity performance tables. For each entity, "
                "check if it has BOTH passing AND failing segments. "
                "An entity with ALL segments failing is NOT a contradiction. "
                "How many entities show this pass/fail contradiction? "
                f"{ANSWER_FORMATS['int']}"
            ),
            answer_type="int",
            metadata={"entities": sorted(entities_with_contradictions)},
        ))
        idx += 1

        # Q: Which entities have contradictions?
        def make_set_fn():
            def fn(g):
                contras = g.find_contradictions()
                return _extract_contradicting_entities(contras)
            return fn

        questions.append(GeneratedQuestion(
            qid=f"contradiction_{idx:03d}",
            category=self.category,
            natural_language="Which entities have contradictory performance across segments?",
            graph_answer_fn=make_set_fn(),
            ground_truth=entities_with_contradictions,
            llm_prompt=(
                "List every entity that has BOTH passing and failing segments. "
                "Exclude entities where ALL segments fail. "
                f"{ANSWER_FORMATS['set_str']}"
            ),
            answer_type="set_str",
            metadata={"entities": sorted(entities_with_contradictions)},
        ))
        idx += 1

        # Per-entity boolean questions
        all_entities = set()
        for h, r, t in graph.triples:
            if r in ("passes", "fails"):
                all_entities.add(h)

        for entity in sorted(all_entities):
            has_contra = entity in entities_with_contradictions

            def make_per_fn(ent):
                def fn(g):
                    contras = g.find_contradictions()
                    ents = _extract_contradicting_entities(contras)
                    return ent in ents
                return fn

            questions.append(GeneratedQuestion(
                qid=f"contradiction_{idx:03d}",
                category=self.category,
                natural_language=f"Does '{entity}' have contradictory performance across segments?",
                graph_answer_fn=make_per_fn(entity),
                ground_truth=has_contra,
                llm_prompt=(
                    f"Look at the performance tables for '{entity}'. Does it have BOTH "
                    f"passing AND failing segments? If all segments fail, answer false. "
                    f"{ANSWER_FORMATS['bool']}"
                ),
                answer_type="bool",
                metadata={"entity": entity},
            ))
            idx += 1

        return questions
