"""Absence — exhaustive negative-scan questions.

Tests whether the LLM can correctly verify that an entity does NOT appear
in a given section, resisting the temptation to hallucinate a value.
Questions present plausible-sounding entity names that exist in one
category but are asked about in a different category where they are absent.

Literature motivation:
  - Feng et al. (2602.10881): LLMs hallucinate facts under negation pressure
"""

import re
import random
from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS

_INTERNAL_ID = re.compile(r"^row_\d+$", re.I)
_AGGREGATE_NAMES = re.compile(
    r"^(total|grand total|all|overall|combined|summary|cumulative|"
    r"aggregate|net|portfolio total|total top \d+)$",
    re.I,
)


def _parse_column(entity_id: str) -> tuple[str, str | None]:
    parts = entity_id.split("/")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return entity_id, None


def _is_skip(name: str) -> bool:
    return bool(_AGGREGATE_NAMES.match(name.strip())) or bool(_INTERNAL_ID.match(name.strip()))


class AbsenceGenerator(QuestionGenerator):
    category = "absence"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Collect entities per category
        entities_by_type: dict[str, set[str]] = {}
        for key in graph.enm:
            base, _ = _parse_column(key.id)
            if _is_skip(base):
                continue
            entities_by_type.setdefault(key.type, set()).add(base)

        types = sorted(entities_by_type.keys())
        if len(types) < 2:
            return questions

        # For each type, find entities that exist ONLY in that type
        # (plausible distractors for other types)
        all_entities = set()
        for ents in entities_by_type.values():
            all_entities.update(ents)

        for target_type in types:
            target_entities = entities_by_type[target_type]
            type_label = target_type.replace("_", " ")

            # Collect plausible absent entities: exist in other types but NOT here
            absent_candidates = []
            for other_type in types:
                if other_type == target_type:
                    continue
                for entity in entities_by_type[other_type]:
                    if entity not in target_entities:
                        absent_candidates.append((entity, other_type))

            # Deduplicate by entity name
            seen = set()
            unique_absent = []
            for entity, source in absent_candidates:
                if entity not in seen:
                    seen.add(entity)
                    unique_absent.append((entity, source))

            for entity, source_type in unique_absent:
                def make_fn(tp, ent):
                    def fn(g):
                        # Check if entity exists in target type
                        for k in g.enm:
                            if k.type != tp:
                                continue
                            b, _ = _parse_column(k.id)
                            if b == ent:
                                return True
                        return False
                    return fn

                q = GeneratedQuestion(
                    qid=f"absence_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"Does '{entity}' appear in the '{type_label}' section?"
                    ),
                    graph_answer_fn=make_fn(target_type, entity),
                    ground_truth=False,
                    llm_prompt=(
                        f"Search the '{type_label}' section of the report. "
                        f"Does an entity named '{entity}' appear in any table "
                        f"in that section?\n"
                        f"Answer true if it appears, false if it does not.\n"
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={
                        "target_type": target_type,
                        "absent_entity": entity,
                        "source_type": source_type,
                    },
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

            # Near-miss distractors: entity names that are prefixes or
            # slight variants of present entities, but don't actually exist.
            # Tests resistance to partial-match hallucination.
            sorted_present = sorted(target_entities)
            near_miss_seen = set()

            for present_entity in sorted_present:
                # Generate prefix distractors: if "Segment_A1" is present,
                # try "Segment_A" (a prefix that doesn't exist)
                for cut in range(1, len(present_entity)):
                    prefix = present_entity[:cut]
                    # Must be a reasonable length and not itself a present entity
                    if len(prefix) < 3:
                        continue
                    if prefix in target_entities:
                        continue
                    if prefix in all_entities:
                        # Only useful if it exists elsewhere (plausible name)
                        # but not in target
                        pass
                    else:
                        continue  # skip if not a real entity anywhere
                    if prefix in near_miss_seen:
                        continue
                    near_miss_seen.add(prefix)

                    def make_near_fn(tp, ent):
                        def fn(g):
                            for k in g.enm:
                                if k.type != tp:
                                    continue
                                b, _ = _parse_column(k.id)
                                if b == ent:
                                    return True
                            return False
                        return fn

                    q = GeneratedQuestion(
                        qid=f"absence_{idx:03d}",
                        category=self.category,
                        natural_language=(
                            f"Does '{prefix}' appear in the "
                            f"'{type_label}' section?"
                        ),
                        graph_answer_fn=make_near_fn(target_type, prefix),
                        ground_truth=False,
                        llm_prompt=(
                            f"Search the '{type_label}' section of the "
                            f"report. Does an entity named exactly "
                            f"'{prefix}' appear in any table in that "
                            f"section?\n"
                            f"Note: partial matches do NOT count. The "
                            f"entity name must match exactly.\n"
                            f"Answer true if it appears, false if not.\n"
                            f"{ANSWER_FORMATS['bool']}"
                        ),
                        answer_type="bool",
                        metadata={
                            "target_type": target_type,
                            "absent_entity": prefix,
                            "near_miss_of": present_entity,
                            "distractor_type": "prefix",
                        },
                    )

                    if self._validate(graph, q):
                        questions.append(q)
                        idx += 1

                # Suffix distractors: append a digit to a present entity
                # e.g., "Model_1" → "Model_10" or "Model_1a"
                for suffix in ["0", "a", "_v2"]:
                    variant = present_entity + suffix
                    if variant in target_entities or variant in near_miss_seen:
                        continue
                    # Must not exist anywhere in the graph
                    if variant in all_entities:
                        continue
                    near_miss_seen.add(variant)

                    def make_suffix_fn(tp, ent):
                        def fn(g):
                            for k in g.enm:
                                if k.type != tp:
                                    continue
                                b, _ = _parse_column(k.id)
                                if b == ent:
                                    return True
                            return False
                        return fn

                    q = GeneratedQuestion(
                        qid=f"absence_{idx:03d}",
                        category=self.category,
                        natural_language=(
                            f"Does '{variant}' appear in the "
                            f"'{type_label}' section?"
                        ),
                        graph_answer_fn=make_suffix_fn(
                            target_type, variant
                        ),
                        ground_truth=False,
                        llm_prompt=(
                            f"Search the '{type_label}' section of the "
                            f"report. Does an entity named exactly "
                            f"'{variant}' appear in any table in that "
                            f"section?\n"
                            f"Note: partial matches do NOT count. The "
                            f"entity name must match exactly.\n"
                            f"Answer true if it appears, false if not.\n"
                            f"{ANSWER_FORMATS['bool']}"
                        ),
                        answer_type="bool",
                        metadata={
                            "target_type": target_type,
                            "absent_entity": variant,
                            "near_miss_of": present_entity,
                            "distractor_type": "suffix",
                        },
                    )

                    if self._validate(graph, q):
                        questions.append(q)
                        idx += 1

            # Also generate some positive controls (entity IS present)
            present_list = sorted(target_entities)
            for entity in present_list:
                def make_present_fn(tp, ent):
                    def fn(g):
                        for k in g.enm:
                            if k.type != tp:
                                continue
                            b, _ = _parse_column(k.id)
                            if b == ent:
                                return True
                        return False
                    return fn

                q = GeneratedQuestion(
                    qid=f"absence_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"Does '{entity}' appear in the '{type_label}' section?"
                    ),
                    graph_answer_fn=make_present_fn(target_type, entity),
                    ground_truth=True,
                    llm_prompt=(
                        f"Search the '{type_label}' section of the report. "
                        f"Does an entity named '{entity}' appear in any table "
                        f"in that section?\n"
                        f"Answer true if it appears, false if it does not.\n"
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={
                        "target_type": target_type,
                        "present_entity": entity,
                        "expected": True,
                    },
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

        return questions
