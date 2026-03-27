"""
FinStructBench — Benchmark for Structured Retrieval from Financial Documents.

Auto-generates questions from document graph topology and evaluates
LLM accuracy against a provably correct graph traversal baseline.

Quick start::

    from finstructbench import Benchmark
    from finstructbench.instances import get_instance_path

    bench = Benchmark(get_instance_path("model_validation"))
    result = bench.run()            # graph-only (no LLM)
    bench.print_results(result)

With LLM evaluation::

    from finstructbench.llm_caller import create_client

    client = create_client()        # requires ANTHROPIC_API_KEY
    result = bench.run(llm_client=client)
"""

__version__ = "0.1.0"

from finstructbench.graph import DocumentGraph, ENMKey, ENMEntry, PhaseEncoder
from finstructbench.ingest import ingest_markdown
from finstructbench.benchmark import Benchmark, BenchmarkResult
from finstructbench.generators import default_generators
from finstructbench.instances import get_instance_path, list_instances
