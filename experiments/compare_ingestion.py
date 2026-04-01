"""
Three-way ingestion comparison: Regex vs LLM-Only vs Hybrid.

Compares DocumentGraphs produced by each mode across all 5 instances:
  1. Regex (default): Pure deterministic regex + heuristic extraction
  2. LLM-Only: LLM extracts all ENM entries and triples from tables and prose
  3. Hybrid: Regex base + LLM fallback for ambiguous columns + LLM prose extraction

Metrics:
  - ENM counts, overlap (Jaccard), value agreement
  - Triple counts, overlap (Jaccard)
  - Regex recovery rate: what fraction of regex ENM does each LLM mode recover?
  - LLM value-add: extra entries beyond regex
  - Numeric precision: does LLM preserve exact values?
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finstructbench.ingest import ingest_markdown
from finstructbench.instances import get_instance_path, list_instances
from finstructbench.llm_caller import create_client

INSTANCES = list_instances()
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
LLM_MODEL = "claude-opus-4-6"
MODES = ["default", "llm_only", "hybrid"]


def pairwise_enm(g1, g2, name1, name2):
    """Compare ENM entries between two graphs."""
    k1 = set(g1.enm.keys())
    k2 = set(g2.enm.keys())
    shared = k1 & k2

    exact_match = 0
    value_errors = []
    for k in shared:
        v1 = g1.enm[k].value
        v2 = g2.enm[k].value
        if v1 == v2:
            exact_match += 1
        else:
            value_errors.append({
                "key": f"{k.type}/{k.id}",
                f"{name1}_value": v1,
                f"{name2}_value": v2,
                "abs_diff": abs(v1 - v2),
                "rel_diff": abs(v1 - v2) / max(abs(v1), 1e-15),
            })

    return {
        f"{name1}_count": len(k1),
        f"{name2}_count": len(k2),
        "shared": len(shared),
        f"only_{name1}": len(k1 - k2),
        f"only_{name2}": len(k2 - k1),
        "exact_match": exact_match,
        "match_rate": exact_match / max(len(shared), 1),
        "jaccard": len(shared) / max(len(k1 | k2), 1),
        "recovery_rate": len(shared) / max(len(k1), 1),
        "value_errors": value_errors[:10],
    }


def pairwise_triples(g1, g2, name1, name2):
    """Compare triples between two graphs."""
    t1 = set(g1.triples)
    t2 = set(g2.triples)
    shared = t1 & t2

    return {
        f"{name1}_count": len(t1),
        f"{name2}_count": len(t2),
        "shared": len(shared),
        f"only_{name1}": len(t1 - t2),
        f"only_{name2}": len(t2 - t1),
        "jaccard": len(shared) / max(len(t1 | t2), 1),
        "recovery_rate": len(shared) / max(len(t1), 1),
    }


def enm_by_category(graphs, mode_names):
    """Per-category ENM counts across modes."""
    all_cats = set()
    for g in graphs.values():
        all_cats.update(k.type for k in g.enm.keys())

    by_cat = {}
    for cat in sorted(all_cats):
        row = {}
        for mode in mode_names:
            keys = {k for k in graphs[mode].enm if k.type == cat}
            row[mode] = len(keys)
        by_cat[cat] = row

    return by_cat


def numeric_precision_analysis(g_regex, g_llm, label):
    """Detailed numeric precision analysis: does LLM match regex values exactly?"""
    k_regex = set(g_regex.enm.keys())
    k_llm = set(g_llm.enm.keys())
    shared = k_regex & k_llm

    if not shared:
        return {"n": 0}

    diffs = []
    exact = 0
    close_1e6 = 0
    close_1e3 = 0
    for k in shared:
        vr = g_regex.enm[k].value
        vl = g_llm.enm[k].value
        d = abs(vr - vl)
        if d == 0:
            exact += 1
        if d < 1e-6:
            close_1e6 += 1
        if d < 1e-3:
            close_1e3 += 1
        if d > 0:
            diffs.append(d)

    arr = np.array(diffs) if diffs else np.array([0.0])
    return {
        "n_shared": len(shared),
        "exact_match": exact,
        "exact_rate": exact / len(shared),
        "within_1e-6": close_1e6,
        "within_1e-3": close_1e3,
        "n_errors": len(diffs),
        "error_mean": float(arr.mean()) if diffs else 0,
        "error_max": float(arr.max()) if diffs else 0,
        "error_median": float(np.median(arr)) if diffs else 0,
    }


def main():
    client = create_client()
    all_results = {}

    # Aggregate accumulators
    agg = {mode: {"enm": 0, "triples": 0} for mode in MODES}
    pair_agg = {}
    PAIRS = [
        ("default", "llm_only"),
        ("default", "hybrid"),
        ("llm_only", "hybrid"),
    ]
    for m1, m2 in PAIRS:
        pair_agg[f"{m1}_vs_{m2}"] = {
            "enm_shared": 0, "enm_union": 0,
            "trip_shared": 0, "trip_union": 0,
            "enm_exact": 0, "enm_shared_total": 0,
        }

    for name in INSTANCES:
        print(f"\n{'='*70}")
        print(f"  INSTANCE: {name}")
        print(f"{'='*70}")

        path = get_instance_path(name)
        graphs = {}

        # Ingest in all three modes
        for mode in MODES:
            label = f"{mode}" + (f" ({LLM_MODEL})" if mode != "default" else "")
            print(f"  Ingesting ({label})...", end=" ", flush=True)
            t0 = time.time()
            if mode == "default":
                graphs[mode] = ingest_markdown(path, mode="default")
            else:
                graphs[mode] = ingest_markdown(
                    path, mode=mode, llm_client=client, llm_model=LLM_MODEL)
            elapsed = time.time() - t0
            g = graphs[mode]
            print(f"done ({elapsed:.1f}s) — "
                  f"{len(g.enm)} ENM, {len(set(g.triples))} triples")

        # Pairwise comparisons
        result = {"instance": name, "graphs": {}, "pairwise_enm": {},
                  "pairwise_triples": {}, "precision": {},
                  "enm_by_category": {}}

        for mode in MODES:
            g = graphs[mode]
            result["graphs"][mode] = {
                "enm_count": len(g.enm),
                "triple_count_unique": len(set(g.triples)),
                "triple_count_total": len(g.triples),
                "stats": g.stats(),
            }
            agg[mode]["enm"] += len(g.enm)
            agg[mode]["triples"] += len(set(g.triples))

        for m1, m2 in PAIRS:
            key = f"{m1}_vs_{m2}"
            enm_cmp = pairwise_enm(graphs[m1], graphs[m2], m1, m2)
            trip_cmp = pairwise_triples(graphs[m1], graphs[m2], m1, m2)
            result["pairwise_enm"][key] = enm_cmp
            result["pairwise_triples"][key] = trip_cmp

            # Aggregate
            k1 = set(graphs[m1].enm.keys())
            k2 = set(graphs[m2].enm.keys())
            t1 = set(graphs[m1].triples)
            t2 = set(graphs[m2].triples)
            pair_agg[key]["enm_shared"] += len(k1 & k2)
            pair_agg[key]["enm_union"] += len(k1 | k2)
            pair_agg[key]["trip_shared"] += len(t1 & t2)
            pair_agg[key]["trip_union"] += len(t1 | t2)
            pair_agg[key]["enm_exact"] += enm_cmp["exact_match"]
            pair_agg[key]["enm_shared_total"] += enm_cmp["shared"]

        # Precision analysis: LLM vs regex
        result["precision"]["llm_only_vs_regex"] = numeric_precision_analysis(
            graphs["default"], graphs["llm_only"], "llm_only")
        result["precision"]["hybrid_vs_regex"] = numeric_precision_analysis(
            graphs["default"], graphs["hybrid"], "hybrid")

        # Per-category
        result["enm_by_category"] = enm_by_category(graphs, MODES)

        all_results[name] = result

        # Print instance summary
        print(f"\n  --- {name} Summary ---")
        print(f"  {'Mode':<12} {'ENM':>6} {'Triples':>8}")
        print(f"  {'-'*28}")
        for mode in MODES:
            g = graphs[mode]
            print(f"  {mode:<12} {len(g.enm):>6} {len(set(g.triples)):>8}")

        print(f"\n  {'Pair':<25} {'ENM Jac':>8} {'Trip Jac':>9} "
              f"{'Val Match':>10} {'Recovery':>9}")
        print(f"  {'-'*63}")
        for m1, m2 in PAIRS:
            key = f"{m1}_vs_{m2}"
            e = result["pairwise_enm"][key]
            t = result["pairwise_triples"][key]
            print(f"  {key:<25} {e['jaccard']:>8.3f} {t['jaccard']:>9.3f} "
                  f"{e['match_rate']:>9.1%} {e['recovery_rate']:>9.1%}")

    # ================================================================
    # AGGREGATE TABLES
    # ================================================================
    print(f"\n\n{'='*90}")
    print(f"AGGREGATE RESULTS: REGEX vs LLM-ONLY vs HYBRID")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"{'='*90}")

    # Table 1: Counts per mode per instance
    print(f"\n--- ENM Counts ---")
    print(f"{'Instance':<22}", end="")
    for mode in MODES:
        print(f" {mode:>12}", end="")
    print()
    print("-" * (22 + 13 * len(MODES)))
    for name in INSTANCES:
        print(f"{name:<22}", end="")
        for mode in MODES:
            n = all_results[name]["graphs"][mode]["enm_count"]
            print(f" {n:>12}", end="")
        print()
    print("-" * (22 + 13 * len(MODES)))
    print(f"{'TOTAL':<22}", end="")
    for mode in MODES:
        print(f" {agg[mode]['enm']:>12}", end="")
    print()

    print(f"\n--- Triple Counts (unique) ---")
    print(f"{'Instance':<22}", end="")
    for mode in MODES:
        print(f" {mode:>12}", end="")
    print()
    print("-" * (22 + 13 * len(MODES)))
    for name in INSTANCES:
        print(f"{name:<22}", end="")
        for mode in MODES:
            n = all_results[name]["graphs"][mode]["triple_count_unique"]
            print(f" {n:>12}", end="")
        print()
    print("-" * (22 + 13 * len(MODES)))
    print(f"{'TOTAL':<22}", end="")
    for mode in MODES:
        print(f" {agg[mode]['triples']:>12}", end="")
    print()

    # Table 2: Pairwise Jaccard
    print(f"\n--- Pairwise ENM Jaccard ---")
    print(f"{'Instance':<22} {'Regex↔LLM':>12} {'Regex↔Hybrid':>14} {'LLM↔Hybrid':>12}")
    print("-" * 62)
    jaccard_enm = {p: [] for p in ["default_vs_llm_only", "default_vs_hybrid", "llm_only_vs_hybrid"]}
    jaccard_trip = {p: [] for p in jaccard_enm}
    val_match = {p: [] for p in jaccard_enm}
    recovery = {p: [] for p in jaccard_enm}

    for name in INSTANCES:
        print(f"{name:<22}", end="")
        for m1, m2 in PAIRS:
            key = f"{m1}_vs_{m2}"
            j = all_results[name]["pairwise_enm"][key]["jaccard"]
            jaccard_enm[key].append(j)
            val_match[key].append(all_results[name]["pairwise_enm"][key]["match_rate"])
            recovery[key].append(all_results[name]["pairwise_enm"][key]["recovery_rate"])
            jaccard_trip[key].append(all_results[name]["pairwise_triples"][key]["jaccard"])
            print(f" {j:>12.3f}", end="")
        print()

    print(f"\n--- Pairwise Triple Jaccard ---")
    print(f"{'Instance':<22} {'Regex↔LLM':>12} {'Regex↔Hybrid':>14} {'LLM↔Hybrid':>12}")
    print("-" * 62)
    for name in INSTANCES:
        print(f"{name:<22}", end="")
        for m1, m2 in PAIRS:
            key = f"{m1}_vs_{m2}"
            j = all_results[name]["pairwise_triples"][key]["jaccard"]
            print(f" {j:>12.3f}", end="")
        print()

    # Table 3: Value match rate (on shared ENM keys)
    print(f"\n--- ENM Value Match Rate (shared keys, exact) ---")
    print(f"{'Instance':<22} {'Regex↔LLM':>12} {'Regex↔Hybrid':>14} {'LLM↔Hybrid':>12}")
    print("-" * 62)
    for name in INSTANCES:
        print(f"{name:<22}", end="")
        for m1, m2 in PAIRS:
            key = f"{m1}_vs_{m2}"
            r = all_results[name]["pairwise_enm"][key]["match_rate"]
            print(f" {r:>11.1%}", end="")
        print()

    # Table 4: Recovery rate (how much of regex does LLM recover?)
    print(f"\n--- Regex ENM Recovery Rate ---")
    print(f"{'Instance':<22} {'LLM recovers':>14} {'Hybrid recovers':>16}")
    print("-" * 54)
    for name in INSTANCES:
        print(f"{name:<22}", end="")
        for key in ["default_vs_llm_only", "default_vs_hybrid"]:
            r = all_results[name]["pairwise_enm"][key]["recovery_rate"]
            print(f" {r:>14.1%}", end="")
        print()

    # Table 5: Numeric precision
    print(f"\n--- Numeric Precision (LLM vs Regex on shared keys) ---")
    print(f"{'Instance':<22} {'LLM-Only':>30} {'Hybrid':>30}")
    print(f"{'':22} {'Exact':>8} {'<1e-6':>8} {'Errors':>8}   {'Exact':>8} {'<1e-6':>8} {'Errors':>8}")
    print("-" * 84)
    for name in INSTANCES:
        p_llm = all_results[name]["precision"]["llm_only_vs_regex"]
        p_hyb = all_results[name]["precision"]["hybrid_vs_regex"]
        print(f"{name:<22} "
              f"{p_llm.get('exact_match',0):>8} {p_llm.get('within_1e-6',0):>8} "
              f"{p_llm.get('n_errors',0):>8}   "
              f"{p_hyb.get('exact_match',0):>8} {p_hyb.get('within_1e-6',0):>8} "
              f"{p_hyb.get('n_errors',0):>8}")

    # Statistical summary
    print(f"\n{'='*90}")
    print("STATISTICAL SUMMARY (n=5 instances)")
    print(f"{'='*90}")

    def stats_line(label, vals):
        arr = np.array(vals)
        return (f"  {label:<35} mean={arr.mean():.4f}  std={arr.std():.4f}  "
                f"min={arr.min():.4f}  max={arr.max():.4f}")

    for key, label in [("default_vs_llm_only", "Regex↔LLM"),
                       ("default_vs_hybrid", "Regex↔Hybrid"),
                       ("llm_only_vs_hybrid", "LLM↔Hybrid")]:
        print(f"\n  {label}:")
        print(stats_line("  ENM Jaccard", jaccard_enm[key]))
        print(stats_line("  Triple Jaccard", jaccard_trip[key]))
        print(stats_line("  Value Match Rate", val_match[key]))
        print(stats_line("  Recovery Rate", recovery[key]))

    # Aggregate Jaccard
    print(f"\n  Aggregate (pooled across all instances):")
    for key, label in [("default_vs_llm_only", "Regex↔LLM"),
                       ("default_vs_hybrid", "Regex↔Hybrid"),
                       ("llm_only_vs_hybrid", "LLM↔Hybrid")]:
        pa = pair_agg[key]
        enm_j = pa["enm_shared"] / max(pa["enm_union"], 1)
        trip_j = pa["trip_shared"] / max(pa["trip_union"], 1)
        val_r = pa["enm_exact"] / max(pa["enm_shared_total"], 1)
        print(f"    {label:<25} ENM Jaccard={enm_j:.4f}  "
              f"Triple Jaccard={trip_j:.4f}  Value Match={val_r:.4f}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm_model": LLM_MODEL,
        "instances": INSTANCES,
        "modes": MODES,
        "aggregate_counts": agg,
        "pair_aggregate": pair_agg,
        "per_instance": all_results,
    }
    safe_model = LLM_MODEL.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"ingestion_3way_{safe_model}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved: {out_path}")


if __name__ == "__main__":
    main()
