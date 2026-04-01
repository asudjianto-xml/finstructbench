"""
FinStructBench — Benchmark Runner.

Usage:
    source ~/jupyterlab/ga_verify/venv/bin/activate
    python -m finstructbench.benchmark instances/model_validation.md --max-per-category 10
"""

import json
import time
import sys
import os
from dataclasses import dataclass, field

from finstructbench.ingest import ingest_markdown
from finstructbench.generators import default_generators
from finstructbench.scorers import score_answer, PARSERS
from finstructbench.llm_caller import call_llm, create_client
from finstructbench.paraphrase import paraphrase_questions


@dataclass
class QuestionResult:
    qid: str
    category: str
    question: str
    ground_truth: str
    graph_answer: str
    graph_correct: bool
    graph_detail: str
    llm_answer: str = None
    llm_correct: bool = False
    llm_detail: str = ""
    llm_raw: str = ""


@dataclass
class BenchmarkResult:
    instance_name: str = ""
    total_questions: int = 0
    graph_score: int = 0
    llm_score: int = 0
    by_category: dict = field(default_factory=dict)
    questions: list = field(default_factory=list)
    model: str = ""
    timestamp: str = ""
    graph_stats: dict = field(default_factory=dict)


class Benchmark:
    def __init__(self, markdown_path, generators=None,
                 max_per_category=10, seed=42, relevance_filter=None,
                 ingest_mode="default", llm_client=None,
                 llm_model="claude-sonnet-4-20250514",
                 question_mode="deterministic"):
        """Initialize the benchmark.

        Args:
            markdown_path: Path to the markdown financial report.
            generators: List of QuestionGenerator instances (default: all 10).
            max_per_category: Max questions sampled per category.
            seed: Random seed for reproducible sampling.
            relevance_filter: Optional RelevanceFilter for business relevance.
            ingest_mode: Document ingestion mode.
            llm_client: Anthropic client (needed for hybrid mode and LLM eval).
            llm_model: Model for paraphrasing in hybrid mode.
            question_mode: "deterministic" (default) or "hybrid".
                - deterministic: Template-based questions. Fully reproducible,
                  zero cost, no LLM involved in question generation.
                - hybrid: Deterministic logic + LLM paraphrase. Preserves
                  ground truth and evaluation prompts; only the natural
                  language presentation is paraphrased for linguistic diversity.
        """
        if question_mode not in ("deterministic", "hybrid"):
            raise ValueError(
                f"question_mode must be 'deterministic' or 'hybrid', "
                f"got '{question_mode}'"
            )

        self.markdown_path = markdown_path
        self.instance_name = os.path.splitext(os.path.basename(markdown_path))[0]
        self.question_mode = question_mode
        self._llm_client = llm_client
        self._llm_model = llm_model

        print(f"Ingesting: {markdown_path} (mode={ingest_mode})")
        self.graph = ingest_markdown(
            markdown_path, mode=ingest_mode,
            llm_client=llm_client, llm_model=llm_model)
        stats = self.graph.stats()
        print(f"  ENM entries: {stats['enm_entries']}")
        print(f"  KG triples: {stats['triples']}")
        print(f"  Phase encoders: {stats['phase_encoders']}")

        with open(markdown_path) as f:
            self.markdown = f.read()

        self.generators = generators or default_generators()
        self.max_per_category = max_per_category
        self.seed = seed
        self.relevance_filter = relevance_filter

    def generate_questions(self):
        all_qs = []
        filter_name = self.relevance_filter.name if self.relevance_filter else "none"
        print(f"\nGenerating questions from graph topology "
              f"(filter={filter_name}, mode={self.question_mode})...")
        for gen in self.generators:
            candidates = gen.generate(self.graph)
            if self.relevance_filter:
                filtered = self.relevance_filter.filter(candidates, self.graph)
                sampled = gen.sample(filtered, self.max_per_category, self.seed)
                print(f"  {gen.category}: {len(candidates)} -> "
                      f"{len(filtered)} filtered -> {len(sampled)} selected")
            else:
                sampled = gen.sample(candidates, self.max_per_category, self.seed)
                print(f"  {gen.category}: {len(candidates)} candidates -> "
                      f"{len(sampled)} selected")
            all_qs.extend(sampled)
        print(f"  Total: {len(all_qs)}")

        # Hybrid mode: paraphrase natural language while preserving ground truth
        if self.question_mode == "hybrid":
            if self._llm_client is None:
                raise ValueError(
                    "Hybrid question mode requires an LLM client. "
                    "Pass llm_client= to Benchmark() or set ANTHROPIC_API_KEY."
                )
            print(f"\nParaphrasing {len(all_qs)} questions (hybrid mode)...")
            all_qs = paraphrase_questions(
                all_qs,
                self._llm_client,
                model=self._llm_model,
                use_cache=True,
                verbose=True,
            )

        return all_qs

    def run(self, llm_client=None, model="claude-sonnet-4-20250514"):
        questions = self.generate_questions()

        result = BenchmarkResult(
            instance_name=self.instance_name,
            total_questions=len(questions),
            model=model,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            graph_stats=self.graph.stats(),
        )

        print(f"\nEvaluating {len(questions)} questions...")
        for i, q in enumerate(questions):
            graph_answer = q.graph_answer_fn(self.graph)
            graph_score = score_answer(graph_answer, q.ground_truth)
            if graph_score.correct:
                result.graph_score += 1

            qr = QuestionResult(
                qid=q.qid,
                category=q.category,
                question=q.natural_language,
                ground_truth=str(q.ground_truth),
                graph_answer=str(graph_answer),
                graph_correct=graph_score.correct,
                graph_detail=graph_score.detail,
            )

            if llm_client:
                try:
                    tag = f"[{i+1}/{len(questions)}] {q.qid}"
                    print(f"  {tag}...", end=" ", flush=True)
                    raw = call_llm(llm_client, self.markdown, q.llm_prompt, model)
                    parser = PARSERS.get(q.answer_type, PARSERS["str"])
                    llm_answer = parser(raw)
                    llm_score = score_answer(llm_answer, q.ground_truth)
                    if llm_score.correct:
                        result.llm_score += 1
                    qr.llm_answer = str(llm_answer)
                    qr.llm_correct = llm_score.correct
                    qr.llm_detail = llm_score.detail
                    qr.llm_raw = raw[:500]
                    status = "PASS" if llm_score.correct else "FAIL"
                    print(f"{status}")
                except Exception as e:
                    qr.llm_answer = f"ERROR: {e}"
                    print(f"ERROR: {e}")

            cat = q.category
            if cat not in result.by_category:
                result.by_category[cat] = {"graph": 0, "llm": 0, "total": 0}
            result.by_category[cat]["total"] += 1
            if graph_score.correct:
                result.by_category[cat]["graph"] += 1
            if qr.llm_correct:
                result.by_category[cat]["llm"] += 1

            result.questions.append(qr)

        return result

    @staticmethod
    def print_results(result):
        n = result.total_questions
        print(f"\n{'=' * 70}")
        print(f"FINSTRUCTBENCH — {result.instance_name}")
        print(f"{'=' * 70}")
        print(f"  Model:     {result.model}")
        print(f"  Questions: {n}")
        print(f"  Graph:     {result.graph_stats.get('enm_entries', 0)} ENM, "
              f"{result.graph_stats.get('triples', 0)} triples")
        print(f"\n  {'Graph Baseline:':<20} {result.graph_score}/{n} "
              f"({result.graph_score/n*100:.0f}%)")
        print(f"  {'LLM + Markdown:':<20} {result.llm_score}/{n} "
              f"({result.llm_score/n*100:.0f}%)")

        print(f"\n  {'Category':<20} {'Graph':>8} {'LLM':>8}")
        print(f"  {'─' * 40}")
        for cat in sorted(result.by_category):
            c = result.by_category[cat]
            print(f"  {cat:<20} {c['graph']:>3}/{c['total']:<3}   {c['llm']:>3}/{c['total']:<3}")

        failures = [q for q in result.questions if not q.llm_correct]
        if failures:
            print(f"\n  LLM Failures ({len(failures)}):")
            for f in failures[:15]:
                print(f"    {f.qid}: {f.question[:65]}")
                print(f"      Expected: {f.ground_truth[:55]}")
                print(f"      Got:      {(f.llm_answer or 'None')[:55]}")

    @staticmethod
    def save_results(result, path):
        data = {
            "instance": result.instance_name,
            "total": result.total_questions,
            "graph_score": result.graph_score,
            "llm_score": result.llm_score,
            "model": result.model,
            "timestamp": result.timestamp,
            "graph_stats": result.graph_stats,
            "by_category": result.by_category,
            "questions": [
                {
                    "qid": q.qid, "category": q.category,
                    "question": q.question, "ground_truth": q.ground_truth,
                    "graph_answer": q.graph_answer, "graph_correct": q.graph_correct,
                    "llm_answer": q.llm_answer, "llm_correct": q.llm_correct,
                    "llm_detail": q.llm_detail,
                }
                for q in result.questions
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinStructBench")
    parser.add_argument("markdown", help="Path to financial report markdown")
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--question-mode",
        choices=["deterministic", "hybrid"],
        default="deterministic",
        help=(
            "Question generation mode. 'deterministic' (default): "
            "template-based, fully reproducible, no LLM cost. "
            "'hybrid': deterministic logic + LLM paraphrase for "
            "linguistic diversity (requires API key)."
        ),
    )
    parser.add_argument(
        "--paraphrase-model",
        default="claude-sonnet-4-20250514",
        help="Model to use for paraphrasing in hybrid mode.",
    )
    args = parser.parse_args()

    client = None if args.no_llm else create_client()

    bench = Benchmark(
        args.markdown,
        max_per_category=args.max_per_category,
        seed=args.seed,
        question_mode=args.question_mode,
        llm_client=client,
        llm_model=args.paraphrase_model,
    )

    result = bench.run(llm_client=client, model=args.model)
    bench.print_results(result)

    output = args.output or f"finstructbench/results/{bench.instance_name}.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    bench.save_results(result, output)


if __name__ == "__main__":
    main()
