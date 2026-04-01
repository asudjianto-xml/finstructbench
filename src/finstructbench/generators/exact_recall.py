"""Exact Recall — mines ENM for scalar value questions."""

import re
from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS

# Internal / synthetic entity names that don't appear in the source document
_INTERNAL_ID = re.compile(r"^row_\d+$", re.I)


def _parse_enm_id(eid: str) -> tuple[str, str | None]:
    """Parse composite ENM ID into (entity_name, column_name).

    ENM IDs use '/' separators: 'entity/label1/.../column' or just 'entity'.
    The last component is typically the column name when there are multiple
    numeric columns.
    """
    parts = eid.split("/")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return eid, None


class ExactRecallGenerator(QuestionGenerator):
    category = "exact_recall"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        for key, entry in list(graph.enm.items()):
            cat, eid = key.type, key.id
            entity_name, col_name = _parse_enm_id(eid)
            meta = graph.enm_meta.get(key, {})

            # Skip internal/synthetic entity names not in the source document
            if _INTERNAL_ID.match(entity_name):
                continue

            def make_fn(c, e):
                def fn(g):
                    return g.lookup(c, e)
                return fn

            # Build clear question text that specifies entity and column
            if col_name:
                nl = (f"What is the '{col_name}' value for '{entity_name}' "
                      f"(category: {cat})?")
                prompt = (
                    f"In the '{cat}' section, find '{entity_name}' and report "
                    f"its '{col_name}' value. "
                    f"Copy the number exactly as shown with ALL decimal places. "
                    f"{ANSWER_FORMATS['float']}"
                )
            else:
                nl = f"What is the exact value of '{entity_name}' (category: {cat})?"
                prompt = (
                    f"Find the exact numeric value for '{entity_name}' in the "
                    f"'{cat}' section of the report. "
                    f"Copy the number exactly as shown with ALL decimal places. "
                    f"{ANSWER_FORMATS['float']}"
                )

            q = GeneratedQuestion(
                qid=f"exact_recall_{idx:03d}",
                category=self.category,
                natural_language=nl,
                graph_answer_fn=make_fn(cat, eid),
                ground_truth=entry.value,
                llm_prompt=prompt,
                answer_type="float",
                metadata={"enm_type": cat, "enm_id": eid,
                          "entity": entity_name, "column": col_name,
                          "meta_column": meta.get("column", ""),
                          "meta_entity": meta.get("entity", "")},
            )

            if self._validate(graph, q):
                questions.append(q)
                idx += 1

        return questions
