from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS
from finstructbench.generators.exact_recall import ExactRecallGenerator
from finstructbench.generators.threshold import ThresholdGenerator
from finstructbench.generators.cross_reference import CrossReferenceGenerator
from finstructbench.generators.contradiction import ContradictionGenerator
from finstructbench.generators.multi_hop import MultiHopGenerator
from finstructbench.generators.counting import CountingGenerator


def default_generators():
    return [
        ExactRecallGenerator(),
        ThresholdGenerator(),
        CrossReferenceGenerator(),
        ContradictionGenerator(),
        MultiHopGenerator(),
        CountingGenerator(),
    ]
