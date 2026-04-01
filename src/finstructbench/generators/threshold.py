"""Threshold — mines numeric values against phase encoder bounds."""

import re
from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS

# Internal / synthetic entity names that don't appear in the source document
_INTERNAL_ID = re.compile(r"^row_\d+$", re.I)

# Default thresholds per encoder type
DEFAULT_THRESHOLDS = {
    "AUC": [(0.5, "ge", "better than random (AUC >= 0.5)"),
            (0.5, "lt", "worse than random (AUC < 0.5)")],
    "accuracy": [(0.5, "ge", "better than random")],
    "AIR": [(0.8, "ge", "the regulatory 4/5ths rule (AIR >= 0.80)")],
    "importance": [(0.1, "ge", "significance (>= 0.1)")],
    "divergence": [(0.3, "ge", "high divergence (>= 0.3)")],
    "loss": [(0.5, "lt", "acceptable loss (< 0.5)")],
    "rate": [(0.05, "lt", "below 5% threshold")],
    "capital_ratio": [(4.5, "ge", "minimum CET1 (>= 4.5%)")],
    "pvalue": [(0.05, "lt", "statistically significant (p < 0.05)")],
}

# Word-boundary patterns mapping to encoders.
# Each entry: (compiled regex, encoder_name)
# Uses word boundaries (\b) to avoid false positives like "pair" matching "air".
_ENCODER_PATTERNS = [
    (re.compile(r"\bauc\b", re.I), "AUC"),
    (re.compile(r"\baccuracy\b", re.I), "accuracy"),
    (re.compile(r"\b(?:air|adverse.impact.ratio)\b", re.I), "AIR"),
    (re.compile(r"\bfairness\b", re.I), "AIR"),
    (re.compile(r"\bimportance\b", re.I), "importance"),
    (re.compile(r"\b(?:divergence|js_divergence)\b", re.I), "divergence"),
    (re.compile(r"\bbrier\b", re.I), "loss"),
    (re.compile(r"\bloss\b", re.I), "loss"),
    (re.compile(r"\bpvalue\b", re.I), "pvalue"),
    (re.compile(r"\bp[\-_]?value\b", re.I), "pvalue"),
    # capital_ratio: match entity column names like "CET1 Ratio", "Capital Ratio",
    # "Tier 1 Ratio", but NOT category-level keywords that are too broad
    (re.compile(r"\bcet1\b", re.I), "capital_ratio"),
    (re.compile(r"\bcapital.ratio\b", re.I), "capital_ratio"),
    (re.compile(r"\btier.1.ratio\b", re.I), "capital_ratio"),
    (re.compile(r"\bleverage.ratio\b", re.I), "capital_ratio"),
    # rate: only match entity column names containing "rate" (e.g. "NCO Rate",
    # "Denial Rate"), not category-level keywords
    (re.compile(r"\brate\b", re.I), "rate"),
]

# Entity column names that should NOT be matched to capital_ratio encoder
# (they live in capital-related categories but are dollar amounts, not ratios)
_CAPITAL_RATIO_ENTITY_EXCLUDE = re.compile(
    r"(?:dividend|repurchase|distribution|ppnr|impact|income|expense|"
    r"provision|charge.off|rwa|asset|deduction|retained|goodwill|"
    r"intangible|dta|surplus|shortfall|beginning|ending|acl)",
    re.I,
)


def _match_encoder(category: str, entity_id: str, value: float) -> str | None:
    """Try to match an ENM entry to a phase encoder.

    Uses word-boundary regex matching AND value range plausibility to avoid
    applying ratio-based thresholds to dollar amounts or mismatched metrics.
    """
    text = f"{category} {entity_id}"
    for pattern, encoder in _ENCODER_PATTERNS:
        if not pattern.search(text):
            continue

        # Semantic exclusion: capital_ratio encoder should not apply to
        # dollar-denominated fields even if category contains "capital"
        if encoder == "capital_ratio" and _CAPITAL_RATIO_ENTITY_EXCLUDE.search(entity_id):
            continue

        # Value plausibility check
        thresholds = DEFAULT_THRESHOLDS.get(encoder, [])
        if not thresholds:
            return encoder
        max_thresh = max(t for t, _, _ in thresholds)
        # Ratios/percentages stay within a bounded range; reject dollar amounts
        if abs(value) <= max(max_thresh * 100, 1000):
            return encoder
    return None


class ThresholdGenerator(QuestionGenerator):
    category = "threshold"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        for key, entry in list(graph.enm.items()):
            cat, eid = key.type, key.id
            meta = graph.enm_meta.get(key, {})
            value = entry.value

            # Skip internal/synthetic entity names
            base_entity = eid.split("/")[0]
            if _INTERNAL_ID.match(base_entity):
                continue

            encoder_name = _match_encoder(cat, eid, value)

            if encoder_name is None or encoder_name not in graph.phase_encoders:
                continue

            thresholds = DEFAULT_THRESHOLDS.get(encoder_name, [])
            for thresh, op, description in thresholds:
                satisfied, margin = graph.check_threshold(encoder_name, value, thresh, op)
                if satisfied is None:
                    continue

                op_word = "at least" if op == "ge" else ("greater than" if op == "gt"
                          else ("at most" if op == "le" else "less than"))

                def make_fn(c, e, enc, th, o):
                    def fn(g):
                        v = g.lookup(c, e)
                        if v is None:
                            return None
                        ok, _ = g.check_threshold(enc, v, th, o)
                        return ok
                    return fn

                q = GeneratedQuestion(
                    qid=f"threshold_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"Does '{eid}' ({cat}) meet {description}? "
                        f"Is its value {op_word} {thresh}?"
                    ),
                    graph_answer_fn=make_fn(cat, eid, encoder_name, thresh, op),
                    ground_truth=satisfied,
                    llm_prompt=(
                        f"Find the value of '{eid}' in the '{cat}' section. "
                        f"Is that value {op_word} {thresh}? "
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={
                        "encoder": encoder_name, "threshold": thresh, "op": op,
                        "column": meta.get("column", ""),
                        "entity": meta.get("entity", eid),
                        "value": value, "enm_type": cat, "enm_id": eid,
                    },
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

        return questions
