"""
FinStructBench evaluation: Qwen2.5-7B (vLLM) and Claude Sonnet 4 (Anthropic API).

Follows the paper methodology:
  - 10 questions per category, seed 42
  - Full markdown document as context
  - All 5 bundled instances
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finstructbench import Benchmark, get_instance_path
from finstructbench.scorers import PARSERS, score_answer
from finstructbench.llm_caller import SYSTEM_PROMPT

INSTANCES = ["model_validation", "fair_lending", "stress_test",
             "credit_portfolio", "basel_capital"]
SEED = 42
MAX_PER_CATEGORY = 10
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── vLLM caller (Qwen) ─────────────────────────────────────────────────────

def create_vllm_engine(model_name="Qwen/Qwen2.5-7B-Instruct"):
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_name,
        max_model_len=32768,
        gpu_memory_utilization=0.25,
        trust_remote_code=True,
    )
    return llm


def call_vllm(engine, markdown_context, question_prompt):
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=1024)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"<report>\n{markdown_context}\n</report>\n\n"
            f"{question_prompt}"
        )},
    ]

    outputs = engine.chat(messages=[messages], sampling_params=params)
    return outputs[0].outputs[0].text


# ── Anthropic caller (Sonnet) ──────────────────────────────────────────────

def create_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def call_anthropic(client, markdown_context, question_prompt,
                   model="claude-sonnet-4-20250514"):
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"<report>\n{markdown_context}\n</report>\n\n"
                f"{question_prompt}"
            ),
        }],
    )
    return response.content[0].text


# ── Evaluation loop ────────────────────────────────────────────────────────

def evaluate_model(model_name, call_fn, instances=INSTANCES):
    """Run the benchmark for one model across all instances."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for inst_name in instances:
        print(f"\n{'='*70}")
        print(f"  {model_name} — {inst_name}")
        print(f"{'='*70}")

        bench = Benchmark(
            get_instance_path(inst_name),
            max_per_category=MAX_PER_CATEGORY,
            seed=SEED,
        )
        questions = bench.generate_questions()

        inst_result = {
            "instance": inst_name,
            "model": model_name,
            "total": len(questions),
            "graph_score": 0,
            "llm_score": 0,
            "by_category": {},
            "questions": [],
        }

        for i, q in enumerate(questions):
            # Graph baseline
            graph_answer = q.graph_answer_fn(bench.graph)
            graph_sc = score_answer(graph_answer, q.ground_truth)
            if graph_sc.correct:
                inst_result["graph_score"] += 1

            # LLM evaluation
            tag = f"[{i+1}/{len(questions)}] {q.qid}"
            print(f"  {tag}...", end=" ", flush=True)
            try:
                raw = call_fn(bench.markdown, q.llm_prompt)
                parser = PARSERS.get(q.answer_type, PARSERS["str"])
                llm_answer = parser(raw)
                llm_sc = score_answer(llm_answer, q.ground_truth)
                correct = llm_sc.correct
                if correct:
                    inst_result["llm_score"] += 1
                print("PASS" if correct else "FAIL")
            except Exception as e:
                raw = f"ERROR: {e}"
                llm_answer = None
                llm_sc = score_answer(None, q.ground_truth)
                correct = False
                print(f"ERROR: {e}")

            cat = q.category
            if cat not in inst_result["by_category"]:
                inst_result["by_category"][cat] = {"graph": 0, "llm": 0, "total": 0}
            inst_result["by_category"][cat]["total"] += 1
            if graph_sc.correct:
                inst_result["by_category"][cat]["graph"] += 1
            if correct:
                inst_result["by_category"][cat]["llm"] += 1

            inst_result["questions"].append({
                "qid": q.qid,
                "category": q.category,
                "question": q.natural_language,
                "ground_truth": str(q.ground_truth),
                "graph_answer": str(graph_answer),
                "graph_correct": graph_sc.correct,
                "llm_answer": str(llm_answer),
                "llm_correct": correct,
                "llm_raw": str(raw)[:500],
            })

        all_results[inst_name] = inst_result

        # Print instance summary
        n = inst_result["total"]
        gs = inst_result["graph_score"]
        ls = inst_result["llm_score"]
        print(f"\n  {inst_name}: Graph {gs}/{n} ({gs/n*100:.0f}%) | "
              f"{model_name} {ls}/{n} ({ls/n*100:.0f}%)")
        for cat in sorted(inst_result["by_category"]):
            c = inst_result["by_category"][cat]
            print(f"    {cat:<20} {c['graph']:>3}/{c['total']:<3}  "
                  f"{c['llm']:>3}/{c['total']:<3}")

    # Save all results
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    return all_results


def print_summary(results_by_model):
    """Print combined results table matching paper format."""
    print(f"\n{'='*70}")
    print("FINSTRUCTBENCH — COMBINED RESULTS")
    print(f"{'='*70}")

    # Table 1: Overall by instance
    models = list(results_by_model.keys())
    header = f"{'Instance':<22} {'Questions':>9} {'Graph':>8}"
    for m in models:
        header += f" {m:>16}"
    print(f"\n{header}")
    print("─" * len(header))

    totals = {m: {"q": 0, "g": 0, "l": 0} for m in models}
    for inst in INSTANCES:
        row = f"{inst:<22}"
        first_model = models[0]
        n = results_by_model[first_model][inst]["total"]
        gs = results_by_model[first_model][inst]["graph_score"]
        row += f" {n:>9} {gs}/{n:>5}"
        for m in models:
            r = results_by_model[m][inst]
            ls = r["llm_score"]
            row += f" {ls}/{n} ({ls/n*100:4.0f}%)"
            totals[m]["q"] += n
            totals[m]["g"] += gs
            totals[m]["l"] += ls
        print(row)

    total_q = totals[models[0]]["q"]
    total_g = totals[models[0]]["g"]
    row = f"{'TOTAL':<22} {total_q:>9} {total_g}/{total_q:>5}"
    for m in models:
        tl = totals[m]["l"]
        row += f" {tl}/{total_q} ({tl/total_q*100:4.0f}%)"
    print("─" * len(header))
    print(row)

    # Table 2: By category
    all_cats = set()
    for m in models:
        for inst in INSTANCES:
            all_cats.update(results_by_model[m][inst]["by_category"].keys())
    all_cats = sorted(all_cats)

    print(f"\n{'Category':<22} {'Total':>6} {'Graph':>6}", end="")
    for m in models:
        print(f" {m:>16}", end="")
    print()
    print("─" * (22 + 6 + 6 + 16 * len(models) + len(models) + 2))

    for cat in all_cats:
        cat_total = 0
        cat_graph = 0
        cat_llm = {m: 0 for m in models}
        for inst in INSTANCES:
            for m in models:
                c = results_by_model[m][inst]["by_category"].get(cat, {})
                if m == models[0]:
                    cat_total += c.get("total", 0)
                    cat_graph += c.get("graph", 0)
                cat_llm[m] += c.get("llm", 0)
        if cat_total == 0:
            continue
        row = f"{cat:<22} {cat_total:>6} {cat_graph:>6}"
        for m in models:
            cl = cat_llm[m]
            row += f" {cl:>4} ({cl/cat_total*100:4.0f}%)"
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "sonnet", "both"],
                        default="both")
    parser.add_argument("--instances", nargs="+", default=INSTANCES)
    args = parser.parse_args()

    results = {}

    if args.model in ("qwen", "both"):
        print("\n" + "="*70)
        print("  Loading Qwen2.5-7B-Instruct via vLLM...")
        print("="*70)
        engine = create_vllm_engine()
        call_fn = lambda md, prompt: call_vllm(engine, md, prompt)
        results["Qwen2.5-7B"] = evaluate_model(
            "Qwen2.5-7B", call_fn, args.instances)
        # Free GPU memory before Sonnet run
        del engine
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.model in ("sonnet", "both"):
        print("\n" + "="*70)
        print("  Running Claude Sonnet 4 via Anthropic API...")
        print("="*70)
        client = create_anthropic_client()
        call_fn = lambda md, prompt: call_anthropic(client, md, prompt)
        results["Claude Sonnet 4"] = evaluate_model(
            "Claude Sonnet 4", call_fn, args.instances)

    if len(results) > 0:
        print_summary(results)
