"""Business-relevance filters for benchmark question selection.

Three pluggable strategies that sit between generate() and sample():

  Option A — TemplateFilter:  Regulatory workflow templates (domain knowledge)
  Option B — SemanticFilter:  Heuristic rules (graph topology)
  Option C — LLMFilter:       LLM-scored relevance (expensive, cached)

All implement the same RelevanceFilter interface so the Benchmark runner
can swap them for side-by-side comparison.
"""

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from finstructbench.generators.base import GeneratedQuestion
from finstructbench.graph import DocumentGraph


# ============================================================================
# Base Interface
# ============================================================================

class RelevanceFilter(ABC):
    """Filter questions for business relevance."""

    name: str = "base"

    @abstractmethod
    def filter(
        self,
        questions: list[GeneratedQuestion],
        graph: DocumentGraph,
    ) -> list[GeneratedQuestion]:
        """Return the subset of questions that are business-relevant."""
        ...


# ============================================================================
# Option A — TemplateFilter (Regulatory Workflow Templates)
# ============================================================================

@dataclass
class WorkflowTemplate:
    """A regulatory workflow that defines what questions are meaningful."""

    name: str
    domain: str
    description: str

    # At least one of these must match the question's metadata / text
    column_patterns: list[str] = field(default_factory=list)
    category_keywords: list[str] = field(default_factory=list)
    relation_keywords: list[str] = field(default_factory=list)

    # Which generator categories this template applies to (empty = all)
    applicable_categories: list[str] = field(default_factory=list)

    def _match_score(self, question: GeneratedQuestion) -> float:
        """Score 0-1 for how well a question matches this template."""
        score = 0.0
        checks = 0

        meta = question.metadata
        text = f"{question.natural_language} {meta.get('enm_type', '')} " \
               f"{meta.get('column', '')} {meta.get('col1', '')} " \
               f"{meta.get('col2', '')} {meta.get('type1', '')} " \
               f"{meta.get('type2', '')} {meta.get('relation', '')} " \
               f"{meta.get('tail', '')} {meta.get('encoder', '')} " \
               f"{meta.get('target_type', '')}".lower()

        # Category filter
        if self.applicable_categories:
            checks += 1
            if question.category in self.applicable_categories:
                score += 1.0

        # Column pattern matching
        if self.column_patterns:
            checks += 1
            for pat in self.column_patterns:
                if re.search(pat, text, re.I):
                    score += 1.0
                    break

        # Category keyword matching
        if self.category_keywords:
            checks += 1
            for kw in self.category_keywords:
                if kw.lower() in text:
                    score += 1.0
                    break

        # Relation keyword matching
        if self.relation_keywords:
            checks += 1
            for kw in self.relation_keywords:
                if kw.lower() in text:
                    score += 1.0
                    break

        return score / max(checks, 1)


# ---- Regulatory workflow template library ----

_TEMPLATES = [
    # --- Fair Lending (ECOA / HMDA) ---
    WorkflowTemplate(
        name="disparate_impact_ratio",
        domain="fair_lending",
        description="Adverse impact ratio analysis across demographic groups",
        column_patterns=[r"\bair\b", r"\badverse.impact", r"\bfairness\b"],
        category_keywords=["lending", "fair", "hmda", "ecoa", "adverse",
                           "demographic", "disparate"],
        relation_keywords=["passes", "fails"],
    ),
    WorkflowTemplate(
        name="approval_denial_rates",
        domain="fair_lending",
        description="Compare approval/denial rates across groups or products",
        column_patterns=[r"\brate\b", r"\bapproval\b", r"\bdenial\b",
                         r"\bminority\b", r"\bmajority\b"],
        category_keywords=["lending", "fair", "application", "origination"],
    ),
    WorkflowTemplate(
        name="fair_lending_threshold",
        domain="fair_lending",
        description="Regulatory 4/5ths rule compliance check",
        column_patterns=[r"\bair\b", r"\b4/5"],
        category_keywords=["fair", "lending", "threshold", "regulatory"],
        applicable_categories=["threshold", "multi_hop"],
    ),

    # --- Basel III / Capital Adequacy ---
    WorkflowTemplate(
        name="capital_ratio_adequacy",
        domain="capital_adequacy",
        description="CET1/Tier 1/Total capital ratio vs regulatory minimums",
        column_patterns=[r"\bcet1\b", r"\btier.?1\b", r"\bcapital.ratio\b",
                         r"\bleverage.ratio\b", r"\btotal.capital\b"],
        category_keywords=["capital", "ratio", "adequacy", "buffer",
                           "minimum", "pillar", "basel"],
    ),
    WorkflowTemplate(
        name="rwa_composition",
        domain="capital_adequacy",
        description="Risk-weighted asset breakdown and analysis",
        column_patterns=[r"\brwa\b", r"\brisk.weight", r"\bead\b",
                         r"\bexposure\b"],
        category_keywords=["credit risk", "market risk", "operational risk",
                           "rwa", "risk.weighted"],
    ),
    WorkflowTemplate(
        name="capital_buffer_sufficiency",
        domain="capital_adequacy",
        description="Conservation buffer, countercyclical buffer checks",
        column_patterns=[r"\bbuffer\b", r"\bsurplus\b", r"\bshortfall\b"],
        category_keywords=["buffer", "conservation", "countercyclical",
                           "surplus"],
        applicable_categories=["threshold", "multi_hop", "exact_recall"],
    ),

    # --- Model Validation (SR 11-7) ---
    WorkflowTemplate(
        name="model_discrimination",
        domain="model_validation",
        description="Model accuracy and discrimination metrics (AUC, ACC)",
        column_patterns=[r"\bauc\b", r"\bacc\b", r"\baccuracy\b",
                         r"\bgini\b", r"\bks\b"],
        category_keywords=["model", "validation", "accuracy", "performance",
                           "discrimination"],
    ),
    WorkflowTemplate(
        name="model_calibration",
        domain="model_validation",
        description="Calibration metrics (Brier score, p-value)",
        column_patterns=[r"\bbrier\b", r"\bpvalue\b", r"\bp.value\b",
                         r"\bcalibration\b"],
        category_keywords=["calibration", "brier", "hosmer", "resolution"],
    ),
    WorkflowTemplate(
        name="model_stability",
        domain="model_validation",
        description="Model stability and drift detection",
        column_patterns=[r"\bpsi\b", r"\bdivergence\b", r"\bdrift\b",
                         r"\bstability\b"],
        category_keywords=["stability", "drift", "monitoring", "divergence"],
    ),
    WorkflowTemplate(
        name="segment_weakness",
        domain="model_validation",
        description="Identify weak-performing model segments",
        column_patterns=[r"\bsegment\b", r"\bslicing\b", r"\bweak\b"],
        category_keywords=["segment", "weakness", "slicing", "subgroup"],
        relation_keywords=["fails", "is_weak"],
    ),

    # --- Stress Testing (CCAR / DFAST) ---
    WorkflowTemplate(
        name="stress_scenario_impact",
        domain="stress_testing",
        description="Capital impact under stress scenarios",
        column_patterns=[r"\bbaseline\b", r"\badverse\b", r"\bseverely\b",
                         r"\bstress\b", r"\bscenario\b"],
        category_keywords=["stress", "scenario", "ccar", "dfast",
                           "baseline", "adverse"],
    ),
    WorkflowTemplate(
        name="stress_capital_drawdown",
        domain="stress_testing",
        description="Capital ratio drawdown from baseline to stress",
        column_patterns=[r"\bcet1\b", r"\btier\b", r"\bratio\b",
                         r"\bdrawdown\b", r"\bchange\b"],
        category_keywords=["stress", "drawdown", "projection", "horizon"],
        applicable_categories=["cross_table_aggregation", "multi_hop",
                               "numeric_computation"],
    ),
    WorkflowTemplate(
        name="loss_projection",
        domain="stress_testing",
        description="Projected losses under stress scenarios",
        column_patterns=[r"\bloss\b", r"\bprovision\b", r"\bcharge.off\b",
                         r"\bnco\b", r"\bppnr\b"],
        category_keywords=["loss", "provision", "projection", "stress",
                           "revenue"],
    ),

    # --- Credit Risk / Portfolio ---
    WorkflowTemplate(
        name="portfolio_concentration",
        domain="credit_risk",
        description="Portfolio composition and concentration analysis",
        column_patterns=[r"\bbalance\b", r"\bexposure\b", r"\bshare\b",
                         r"\bconcentration\b", r"\b%.*total\b"],
        category_keywords=["portfolio", "segment", "composition",
                           "concentration", "diversification"],
    ),
    WorkflowTemplate(
        name="credit_quality",
        domain="credit_risk",
        description="Delinquency, charge-off, and recovery metrics",
        column_patterns=[r"\bnco\b", r"\bdelinquen", r"\brecovery\b",
                         r"\bcharge.off\b", r"\bpd\b", r"\blgd\b"],
        category_keywords=["credit quality", "delinquency", "nco",
                           "charge-off", "recovery", "loss"],
    ),
    WorkflowTemplate(
        name="vintage_analysis",
        domain="credit_risk",
        description="Cohort/vintage performance tracking",
        column_patterns=[r"\bvintage\b", r"\bcohort\b", r"\borigination\b",
                         r"\bseasoning\b"],
        category_keywords=["vintage", "cohort", "origination", "seasoning"],
    ),

    # --- Cross-cutting workflows ---
    WorkflowTemplate(
        name="period_comparison",
        domain="general",
        description="Period-over-period changes and trends",
        column_patterns=[r"\bchange\b", r"\bprior\b", r"\bcurrent\b",
                         r"\bbps\b", r"\byoy\b", r"\bqoq\b"],
        category_keywords=["change", "period", "comparison", "prior",
                           "trend"],
        applicable_categories=["cross_table_aggregation",
                               "numeric_computation", "exact_recall"],
    ),
    WorkflowTemplate(
        name="pass_fail_compliance",
        domain="general",
        description="Pass/fail status and contradiction detection",
        relation_keywords=["passes", "fails"],
        applicable_categories=["contradiction", "cross_reference",
                               "counting", "threshold"],
    ),
]


class TemplateFilter(RelevanceFilter):
    """Option A: Keep questions that match at least one regulatory workflow template."""

    name = "template"

    def __init__(self, templates: list[WorkflowTemplate] | None = None,
                 min_score: float = 0.5):
        self.templates = templates or _TEMPLATES
        self.min_score = min_score

    def filter(self, questions, graph):
        kept = []
        for q in questions:
            best = max(
                (t._match_score(q) for t in self.templates),
                default=0.0,
            )
            if best >= self.min_score:
                kept.append(q)
        return kept


# ============================================================================
# Option B — SemanticFilter (Heuristic Rules)
# ============================================================================

# Columns where aggregation (mean/median) is meaningful
_RATIO_COLUMNS = re.compile(
    r"\b(auc|acc|accuracy|air|ratio|rate|psi|brier|gini|ks|"
    r"pvalue|p.value|divergence|importance|share|weight|score|"
    r"cet1|tier|leverage)\b", re.I
)

# Aggregate row names
_AGGREGATE_NAMES = re.compile(
    r"^(total|grand total|all|overall|combined|summary|cumulative|"
    r"aggregate|net|portfolio total|total top \d+)$",
    re.I,
)


class SemanticFilter(RelevanceFilter):
    """Option B: Heuristic rules that reject business-nonsensical questions."""

    name = "semantic"

    def filter(self, questions, graph):
        # Pre-compute: which entities share KG relations (same group)
        entity_groups = self._build_entity_groups(graph)
        kept = []

        for q in questions:
            if self._passes_rules(q, entity_groups, graph):
                kept.append(q)
        return kept

    def _build_entity_groups(self, graph):
        """Map entity → set of group identifiers from KG relations."""
        groups: dict[str, set[str]] = {}
        skip = {"in_section", "has_value"}
        for h, r, t in graph.triples:
            if r in skip:
                continue
            groups.setdefault(h, set()).add(f"{r}:{t}")
        return groups

    def _passes_rules(self, q, entity_groups, graph):
        meta = q.metadata
        cat = q.category

        # Rule 1: Numeric computation pairs should share a KG relation
        if cat == "numeric_computation" and meta.get("op") in (
            "sum", "difference", "ratio", "pct_change"
        ):
            a = meta.get("entity_a", "")
            b = meta.get("entity_b", "")
            groups_a = entity_groups.get(a, set())
            groups_b = entity_groups.get(b, set())
            if groups_a and groups_b and not (groups_a & groups_b):
                return False

        # Rule 2: Mean/median only on ratio-like columns
        if cat == "numeric_computation" and meta.get("op") in (
            "mean", "median"
        ):
            col = meta.get("column", "")
            if not _RATIO_COLUMNS.search(col):
                return False

        # Rule 3: Absence questions need non-trivial entity names
        if cat == "absence":
            entity = meta.get("absent_entity") or meta.get("present_entity", "")
            if len(entity) < 3:
                return False
            # Near-miss suffix distractors with pure appended digits are fine
            # but skip if the base entity looks synthetic
            if re.match(r"^row_\d+$", entity, re.I):
                return False

        # Rule 4: Cross-table aggregation needs sections that share entities
        if cat == "cross_table_aggregation":
            types = meta.get("types") or []
            if len(types) < 2:
                t1 = meta.get("type1", "")
                t2 = meta.get("type2", "")
                types = [t1, t2]
            # Check that the entity actually appears in both
            entity = meta.get("entity", "")
            if not entity:
                return False

        # Rule 5: Threshold questions — encoder must be plausible for section
        if cat == "threshold":
            encoder = meta.get("encoder", "")
            # capital_ratio encoder only in capital-related sections
            enm_type = meta.get("enm_type", "")
            if not enm_type:
                # Extract from question text
                enm_type = q.natural_language.lower()
            if encoder == "capital_ratio":
                capital_kw = {"capital", "cet1", "tier", "leverage", "ratio",
                              "buffer", "stress", "scenario"}
                if not any(kw in enm_type.lower() for kw in capital_kw):
                    return False

        # Rule 6: Multi-hop conditional_filter_count — require non-trivial
        # tail values (not raw float strings like "0.0")
        if cat == "multi_hop" and meta.get("pattern") == "conditional_filter_count":
            tail = meta.get("tail", "")
            try:
                float(tail)
                # tail is a raw number — less business-meaningful
                return False
            except (ValueError, TypeError):
                pass

        return True


# ============================================================================
# Option C — LLMFilter (LLM-Scored Relevance)
# ============================================================================

_LLM_SYSTEM = """You are a senior financial risk analyst reviewing benchmark \
questions for a regulatory compliance evaluation. Score each question 1-5 for \
business relevance:

5 = Critical regulatory question (auditor would ask this)
4 = Important business question (risk manager would ask)
3 = Reasonable analytical question (analyst might explore)
2 = Technically valid but unlikely to be asked in practice
1 = Artificial / no business motivation

Consider: Does this question reflect a real regulatory workflow? Would answering \
it inform a business decision? Is the comparison/aggregation meaningful?

For each question, respond with ONLY the score (1-5), one per line, in order."""


class LLMFilter(RelevanceFilter):
    """Option C: LLM scores questions for business relevance."""

    name = "llm"

    def __init__(self, client=None, model="claude-sonnet-4-20250514",
                 min_score=3, batch_size=20, max_pool=200,
                 cache_dir=None):
        self.client = client
        self.model = model
        self.min_score = min_score
        self.batch_size = batch_size
        self.max_pool = max_pool
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), ".filter_cache"
        )
        self._cache: dict[str, int] = {}
        self._load_cache()

    def _cache_path(self):
        return os.path.join(self.cache_dir, f"llm_scores_{self.model}.json")

    def _load_cache(self):
        path = self._cache_path()
        if os.path.exists(path):
            with open(path) as f:
                self._cache = json.load(f)

    def _save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self._cache_path(), "w") as f:
            json.dump(self._cache, f)

    @staticmethod
    def _question_hash(q: GeneratedQuestion) -> str:
        text = f"{q.category}|{q.natural_language}|{q.answer_type}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _score_batch(self, batch: list[GeneratedQuestion]) -> list[int]:
        """Call LLM to score a batch of questions."""
        if self.client is None:
            raise RuntimeError(
                "LLMFilter requires an Anthropic client. "
                "Pass client=create_client() or set ANTHROPIC_API_KEY."
            )

        numbered = "\n".join(
            f"{i+1}. [{q.category}] {q.natural_language}"
            for i, q in enumerate(batch)
        )
        prompt = (
            f"Score each question 1-5 for business relevance. "
            f"Reply with ONLY the scores, one per line.\n\n{numbered}"
        )

        response = self.client.messages.create(
            model=self.model,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        text = response.content[0].text

        # Parse scores
        scores = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # Handle "1. 4" or just "4" formats
            m = re.search(r"(\d)", line)
            if m:
                s = int(m.group(1))
                scores.append(min(max(s, 1), 5))
        # Pad if LLM returned fewer scores than expected
        while len(scores) < len(batch):
            scores.append(3)  # default to borderline

        return scores[:len(batch)]

    def filter(self, questions, graph):
        import random

        # For large candidate lists, pre-sample to a manageable pool.
        # The LLM scores this pool; unscored questions are dropped.
        max_pool = self.max_pool
        if len(questions) > max_pool:
            rng = random.Random(42)
            pool = rng.sample(questions, max_pool)
        else:
            pool = questions

        # Check cache first, batch uncached questions
        uncached = []
        uncached_indices = []

        scores = {}
        for i, q in enumerate(pool):
            h = self._question_hash(q)
            if h in self._cache:
                scores[i] = self._cache[h]
            else:
                uncached.append(q)
                uncached_indices.append(i)

        # Score uncached in batches
        for batch_start in range(0, len(uncached), self.batch_size):
            batch = uncached[batch_start:batch_start + self.batch_size]
            batch_indices = uncached_indices[
                batch_start:batch_start + self.batch_size
            ]
            batch_scores = self._score_batch(batch)

            for q, idx, s in zip(batch, batch_indices, batch_scores):
                h = self._question_hash(q)
                self._cache[h] = s
                scores[idx] = s

        self._save_cache()

        return [
            q for i, q in enumerate(pool)
            if scores.get(i, 0) >= self.min_score
        ]
