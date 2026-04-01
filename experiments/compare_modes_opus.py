"""Compare deterministic vs hybrid (LLM-paraphrased) question modes.

Runs both modes on all 5 instances, evaluates with graph baseline and
Claude Opus 4.6, then prints a side-by-side comparison.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finstructbench import Benchmark, get_instance_path
from finstructbench.llm_caller import create_client


def run_instance(instance_name, client, question_mode, model):
    """Run benchmark on one instance with the given question mode."""
    path = get_instance_path(instance_name)
    bench = Benchmark(
        path,
        max_per_category=10,
        seed=42,
        question_mode=question_mode,
        llm_client=client,
        llm_model=model,
    )
    result = bench.run(llm_client=client, model=model)
    return result


def main():
    model = "claude-opus-4-6"
    client = create_client()

    instances = ["model_validation", "fair_lending", "stress_test",
                 "credit_portfolio", "basel_capital"]

    results = {}

    for mode in ["deterministic", "hybrid"]:
        results[mode] = {}
        print(f"\n{'=' * 70}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'=' * 70}")

        for inst in instances:
            print(f"\n--- {inst} ({mode}) ---")
            result = run_instance(inst, client, mode, model)
            results[mode][inst] = result
            Benchmark.print_results(result)

            # Save per-run results
            out_dir = f"experiments/results/opus_{mode}"
            os.makedirs(out_dir, exist_ok=True)
            Benchmark.save_results(result, f"{out_dir}/{inst}.json")

    # ================================================================
    # Side-by-side comparison
    # ================================================================
    print(f"\n\n{'=' * 80}")
    print("COMPARISON: DETERMINISTIC vs HYBRID (LLM-Paraphrased)")
    print(f"{'=' * 80}")
    print(f"Model: {model}")
    print(f"Questions per category: 10, Seed: 42\n")

    # Overall table
    print(f"{'Instance':<22} {'Det Graph':>10} {'Det LLM':>10} "
          f"{'Hyb Graph':>10} {'Hyb LLM':>10} {'Delta LLM':>10}")
    print(f"{'-' * 72}")

    total_det_graph = total_det_llm = 0
    total_hyb_graph = total_hyb_llm = 0
    total_qs = 0

    for inst in instances:
        det = results["deterministic"][inst]
        hyb = results["hybrid"][inst]
        n = det.total_questions

        det_g = f"{det.graph_score}/{n}"
        det_l = f"{det.llm_score}/{n} ({det.llm_score/n*100:.0f}%)"
        hyb_g = f"{hyb.graph_score}/{n}"
        hyb_l = f"{hyb.llm_score}/{n} ({hyb.llm_score/n*100:.0f}%)"
        delta = hyb.llm_score - det.llm_score
        delta_s = f"{'+' if delta >= 0 else ''}{delta}"

        print(f"{inst:<22} {det_g:>10} {det_l:>10} "
              f"{hyb_g:>10} {hyb_l:>10} {delta_s:>10}")

        total_det_graph += det.graph_score
        total_det_llm += det.llm_score
        total_hyb_graph += hyb.graph_score
        total_hyb_llm += hyb.llm_score
        total_qs += n

    print(f"{'-' * 72}")
    print(f"{'TOTAL':<22} "
          f"{total_det_graph}/{total_qs}:>10 "
          f"{total_det_llm}/{total_qs} ({total_det_llm/total_qs*100:.0f}%):>10 "
          f"{total_hyb_graph}/{total_qs}:>10 "
          f"{total_hyb_llm}/{total_qs} ({total_hyb_llm/total_qs*100:.0f}%):>10 "
          f"{'+' if (total_hyb_llm - total_det_llm) >= 0 else ''}"
          f"{total_hyb_llm - total_det_llm}:>10")

    # Per-category comparison
    print(f"\n{'Category':<25} {'Det LLM':>12} {'Hyb LLM':>12} {'Delta':>8}")
    print(f"{'-' * 57}")

    all_cats = set()
    for mode in ["deterministic", "hybrid"]:
        for inst in instances:
            all_cats.update(results[mode][inst].by_category.keys())

    for cat in sorted(all_cats):
        det_correct = det_total = hyb_correct = hyb_total = 0
        for inst in instances:
            if cat in results["deterministic"][inst].by_category:
                c = results["deterministic"][inst].by_category[cat]
                det_correct += c["llm"]
                det_total += c["total"]
            if cat in results["hybrid"][inst].by_category:
                c = results["hybrid"][inst].by_category[cat]
                hyb_correct += c["llm"]
                hyb_total += c["total"]

        if det_total == 0:
            continue

        det_pct = det_correct / det_total * 100 if det_total else 0
        hyb_pct = hyb_correct / hyb_total * 100 if hyb_total else 0
        delta = hyb_correct - det_correct

        print(f"{cat:<25} {det_correct:>3}/{det_total:<3} ({det_pct:>4.0f}%) "
              f"{hyb_correct:>3}/{hyb_total:<3} ({hyb_pct:>4.0f}%) "
              f"{'+'if delta >= 0 else ''}{delta:>4}")

    # Save combined results
    combined = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "instances": instances,
        "deterministic": {
            inst: {
                "total": results["deterministic"][inst].total_questions,
                "graph_score": results["deterministic"][inst].graph_score,
                "llm_score": results["deterministic"][inst].llm_score,
                "by_category": results["deterministic"][inst].by_category,
            }
            for inst in instances
        },
        "hybrid": {
            inst: {
                "total": results["hybrid"][inst].total_questions,
                "graph_score": results["hybrid"][inst].graph_score,
                "llm_score": results["hybrid"][inst].llm_score,
                "by_category": results["hybrid"][inst].by_category,
            }
            for inst in instances
        },
    }
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/comparison_opus.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved: experiments/results/comparison_opus.json")


if __name__ == "__main__":
    main()
