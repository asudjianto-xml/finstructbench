from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS
from finstructbench.filters import RelevanceFilter, TemplateFilter, SemanticFilter, LLMFilter
from finstructbench.generators.exact_recall import ExactRecallGenerator
from finstructbench.generators.threshold import ThresholdGenerator
from finstructbench.generators.cross_reference import CrossReferenceGenerator
from finstructbench.generators.contradiction import ContradictionGenerator
from finstructbench.generators.multi_hop import MultiHopGenerator
from finstructbench.generators.counting import CountingGenerator
from finstructbench.generators.numeric_computation import NumericComputationGenerator
from finstructbench.generators.ranking import RankingGenerator
from finstructbench.generators.absence import AbsenceGenerator
from finstructbench.generators.cross_table_aggregation import CrossTableAggregationGenerator


def default_generators():
    return [
        ExactRecallGenerator(),
        ThresholdGenerator(),
        CrossReferenceGenerator(),
        ContradictionGenerator(),
        MultiHopGenerator(),
        CountingGenerator(),
        NumericComputationGenerator(),
        RankingGenerator(),
        AbsenceGenerator(),
        CrossTableAggregationGenerator(),
    ]
