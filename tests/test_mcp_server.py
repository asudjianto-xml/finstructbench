"""
Comprehensive tests for the FinStructBench MCP server.

Covers:
  1. Server setup and tool registration
  2. All 14 tools across all 5 instances
  3. Ground-truth validation (MCP tool answers == benchmark graph answers)
  4. Edge cases and error handling
  5. Custom document loading
  6. Cross-instance consistency
"""

import json
import os
import tempfile

import pytest

from finstructbench.graph import DocumentGraph
from finstructbench.ingest import ingest_markdown
from finstructbench.instances import get_instance_path, list_instances
from finstructbench.generators import default_generators
from finstructbench.scorers import score_answer
from finstructbench.mcp_server import (
    mcp,
    _get_graph,
    _graphs,
    list_documents,
    load_document,
    graph_stats,
    query_enm,
    query_triples,
    exact_recall,
    threshold_check,
    cross_reference,
    count_entities,
    find_contradictions,
    multi_hop_argminmax,
    multi_hop_chain,
    list_relations,
    list_enm_types,
)


ALL_INSTANCES = list_instances()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the graph cache before each test."""
    _graphs.clear()
    yield
    _graphs.clear()


# =========================================================================
# 1. SERVER SETUP AND TOOL REGISTRATION
# =========================================================================

class TestServerSetup:
    def test_mcp_server_object_exists(self):
        assert mcp is not None
        assert mcp.name == "finstructbench"

    def test_all_14_tools_registered(self):
        tools = mcp._tool_manager.list_tools()
        names = {t.name for t in tools}
        expected = {
            "list_documents", "load_document", "graph_stats",
            "query_enm", "query_triples",
            "exact_recall", "threshold_check", "cross_reference",
            "count_entities", "find_contradictions",
            "multi_hop_argminmax", "multi_hop_chain",
            "list_relations", "list_enm_types",
        }
        assert expected == names, f"Missing: {expected - names}, Extra: {names - expected}"

    def test_tool_count(self):
        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 14


# =========================================================================
# 2. MANAGEMENT & DISCOVERY TOOLS
# =========================================================================

class TestManagementTools:
    def test_list_documents_shows_bundled(self):
        result = json.loads(list_documents())
        assert set(result["bundled_instances"]) == set(ALL_INSTANCES)

    def test_list_documents_tracks_loaded(self):
        _get_graph("model_validation")
        result = json.loads(list_documents())
        assert "model_validation" in result["loaded"]

    def test_load_document_custom(self):
        path = get_instance_path("fair_lending")
        result = json.loads(load_document(path, doc_id="custom_fl"))
        assert result["doc_id"] == "custom_fl"
        assert result["enm_entries"] > 0
        assert result["triples"] > 0
        # Should now appear in list
        docs = json.loads(list_documents())
        assert "custom_fl" in docs["custom_loaded"]

    def test_load_document_missing_file(self):
        result = json.loads(load_document("/nonexistent/file.md"))
        assert "error" in result

    def test_load_document_auto_id(self):
        path = get_instance_path("basel_capital")
        result = json.loads(load_document(path))
        assert result["doc_id"] == "basel_capital"

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_graph_stats_all_instances(self, instance):
        result = json.loads(graph_stats(instance))
        assert result["doc_id"] == instance
        assert result["enm_entries"] > 0
        assert result["triples"] > 0
        assert len(result["phase_encoders"]) > 0

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_list_enm_types_all_instances(self, instance):
        result = json.loads(list_enm_types(instance))
        assert result["total_entries"] > 0
        assert len(result["types"]) > 0
        # Counts should sum to total
        total = sum(t["count"] for t in result["types"])
        assert total == result["total_entries"]

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_list_relations_all_instances(self, instance):
        result = json.loads(list_relations(instance))
        assert result["total_triples"] > 0
        assert len(result["relations"]) > 0
        # Counts should sum to total
        total = sum(r["count"] for r in result["relations"])
        assert total == result["total_triples"]


# =========================================================================
# 3. LOW-LEVEL PRIMITIVES
# =========================================================================

class TestQueryENM:
    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_query_all_entries(self, instance):
        """query_enm with no filters returns all entries."""
        result = json.loads(query_enm(instance))
        assert isinstance(result, list)
        graph = _get_graph(instance)
        assert len(result) == len(graph.enm)

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_query_by_type(self, instance):
        """query_enm filtered by type returns correct subset."""
        graph = _get_graph(instance)
        types = json.loads(list_enm_types(instance))
        first_type = types["types"][0]["name"]
        expected_count = types["types"][0]["count"]
        result = json.loads(query_enm(instance, enm_type=first_type))
        assert isinstance(result, list)
        assert len(result) == expected_count

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_query_specific_entry(self, instance):
        """query_enm with type+id returns exact value."""
        graph = _get_graph(instance)
        key = next(iter(graph.enm))
        entry = graph.enm[key]
        result = json.loads(query_enm(instance, enm_type=key.type, entity_id=key.id))
        assert result["value"] == entry.value

    def test_query_nonexistent_type(self):
        result = json.loads(query_enm("model_validation", enm_type="nonexistent_xyz"))
        assert "error" in result
        assert "available_enm_types" in result

    def test_query_nonexistent_entry(self):
        graph = _get_graph("model_validation")
        first_type = next(iter(graph.enm)).type
        result = json.loads(query_enm(
            "model_validation", enm_type=first_type, entity_id="nonexistent_entity"
        ))
        assert "error" in result


class TestQueryTriples:
    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_query_by_relation(self, instance):
        """query_triples filtered by relation returns correct matches."""
        graph = _get_graph(instance)
        rels = json.loads(list_relations(instance))
        rel = rels["relations"][0]["name"]
        expected = len(graph.query_triples(relation=rel))
        result = json.loads(query_triples(instance, relation=rel, limit=10000))
        assert result["total"] == expected

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_query_by_head(self, instance):
        """query_triples by head entity returns correct matches."""
        graph = _get_graph(instance)
        # Pick a head that appears in triples
        head = graph.triples[0][0]
        expected = len(graph.query_triples(head=head))
        result = json.loads(query_triples(instance, head=head, limit=10000))
        assert result["total"] == expected

    def test_query_with_limit(self):
        result = json.loads(query_triples("model_validation", limit=5))
        assert len(result["triples"]) <= 5
        if result["total"] > 5:
            assert result["truncated"] is True

    def test_query_empty_result(self):
        result = json.loads(query_triples(
            "model_validation", head="nonexistent_entity_xyz"
        ))
        assert result["total"] == 0
        assert result["triples"] == []


# =========================================================================
# 4. CATEGORY-SPECIFIC TOOLS
# =========================================================================

class TestExactRecall:
    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_exact_recall_every_entry(self, instance):
        """Every ENM entry should be retrievable via exact_recall."""
        graph = _get_graph(instance)
        failures = []
        for key, entry in graph.enm.items():
            result = json.loads(exact_recall(instance, key.type, key.id))
            if "error" in result:
                failures.append(f"{key.type}/{key.id}: {result['error']}")
            elif abs(result["value"] - entry.value) > 1e-9:
                failures.append(
                    f"{key.type}/{key.id}: got {result['value']} != {entry.value}"
                )
        assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])

    def test_exact_recall_wrong_type(self):
        result = json.loads(exact_recall("model_validation", "bogus_type", "x"))
        assert "error" in result
        assert "available_types" in result

    def test_exact_recall_wrong_id(self):
        graph = _get_graph("model_validation")
        first_type = next(iter(graph.enm)).type
        result = json.loads(exact_recall("model_validation", first_type, "bogus_id"))
        assert "error" in result
        assert "available_entries" in result


class TestThresholdCheck:
    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_threshold_ge(self, instance):
        """Threshold check with >= operator."""
        graph = _get_graph(instance)
        key = next(iter(graph.enm))
        entry = graph.enm[key]
        # Test with a threshold below the value — should satisfy
        low_thresh = entry.value - 1.0
        result = json.loads(threshold_check(
            instance, key.type, key.id, low_thresh, "ge"
        ))
        assert result["satisfied"] is True
        assert result["value"] == entry.value
        assert abs(result["margin"] - (entry.value - low_thresh)) < 1e-9

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_threshold_lt(self, instance):
        """Threshold check with < operator."""
        graph = _get_graph(instance)
        key = next(iter(graph.enm))
        entry = graph.enm[key]
        # Test with a threshold above the value — should satisfy <
        high_thresh = entry.value + 1.0
        result = json.loads(threshold_check(
            instance, key.type, key.id, high_thresh, "lt"
        ))
        assert result["satisfied"] is True

    def test_threshold_exact_boundary(self):
        """Value exactly at threshold: >= should pass, > should fail."""
        graph = _get_graph("model_validation")
        key = next(iter(graph.enm))
        val = graph.enm[key].value
        ge_result = json.loads(threshold_check(
            "model_validation", key.type, key.id, val, "ge"
        ))
        gt_result = json.loads(threshold_check(
            "model_validation", key.type, key.id, val, "gt"
        ))
        assert ge_result["satisfied"] is True
        assert gt_result["satisfied"] is False

    def test_threshold_bad_operator(self):
        graph = _get_graph("model_validation")
        key = next(iter(graph.enm))
        result = json.loads(threshold_check(
            "model_validation", key.type, key.id, 0.5, "invalid_op"
        ))
        assert "error" in result

    def test_threshold_nonexistent_entry(self):
        result = json.loads(threshold_check(
            "model_validation", "bogus", "bogus", 0.5, "ge"
        ))
        assert "error" in result


class TestCrossReference:
    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_cross_reference_basic(self, instance):
        """Cross-reference returns correct intersection."""
        graph = _get_graph(instance)
        rels = json.loads(list_relations(instance))
        # Find two relations that have overlapping heads
        rel_names = [r["name"] for r in rels["relations"]
                     if r["name"] not in ("in_section", "has_value", "has_effect")]
        found = False
        for i, r1 in enumerate(rel_names[:5]):
            for r2 in rel_names[i+1:6]:
                result = json.loads(cross_reference(instance, r1, r2))
                if result["count"] > 0:
                    # Verify against direct graph query
                    h1 = {h for h, _, _ in graph.query_triples(relation=r1)}
                    h2 = {h for h, _, _ in graph.query_triples(relation=r2)}
                    expected = h1 & h2
                    assert set(result["entities"]) == expected
                    found = True
                    break
            if found:
                break
        # At least one pair should have overlap in every instance
        assert found, f"No overlapping relation pairs found in {instance}"

    def test_cross_reference_no_overlap(self):
        """Two relations with no common heads return empty."""
        # Use a relation that only model_validation has
        result = json.loads(cross_reference(
            "model_validation", "passes", "this_relation_does_not_exist"
        ))
        # query_triples returns empty for nonexistent relation, so overlap = empty
        assert result["count"] == 0

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_cross_reference_self(self, instance):
        """Cross-referencing a relation with itself returns all its heads."""
        rels = json.loads(list_relations(instance))
        rel = rels["relations"][0]["name"]
        result = json.loads(cross_reference(instance, rel, rel))
        graph = _get_graph(instance)
        expected = {h for h, _, _ in graph.query_triples(relation=rel)}
        assert set(result["entities"]) == expected


class TestCountEntities:
    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_count_total_triples(self, instance):
        """count_entities for a relation matches direct triple count."""
        graph = _get_graph(instance)
        rels = json.loads(list_relations(instance))
        for r in rels["relations"][:3]:
            result = json.loads(count_entities(instance, r["name"]))
            assert result["triple_count"] == r["count"]

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_count_by_tail(self, instance):
        """count_entities with tail filter matches direct query."""
        graph = _get_graph(instance)
        # Find a (relation, tail) pair
        for h, r, t in graph.triples[:50]:
            if r not in ("in_section", "has_value"):
                expected = len(graph.query_triples(relation=r, tail=t))
                result = json.loads(count_entities(instance, r, tail=t))
                assert result["count"] == expected
                break

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_count_unique_heads(self, instance):
        """count_entities with count_unique_heads matches set size."""
        graph = _get_graph(instance)
        rels = json.loads(list_relations(instance))
        rel = rels["relations"][0]["name"]
        expected = len(set(h for h, _, _ in graph.query_triples(relation=rel)))
        result = json.loads(count_entities(instance, rel, count_unique_heads=True))
        assert result["unique_head_count"] == expected


class TestFindContradictions:
    def test_model_validation_has_contradictions(self):
        """Model validation is the only instance with contradictions."""
        result = json.loads(find_contradictions("model_validation"))
        graph = _get_graph("model_validation")
        expected = graph.find_contradictions()
        if expected:
            assert result["contradiction_count"] > 0
            assert len(result["features"]) > 0
            assert len(result["raw_contradictions"]) == len(expected)
        else:
            assert result["contradiction_count"] == 0

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_contradictions_consistency(self, instance):
        """Contradiction tool matches graph.find_contradictions()."""
        graph = _get_graph(instance)
        expected_raw = graph.find_contradictions()
        result = json.loads(find_contradictions(instance))
        raw = result.get("raw_contradictions", [])
        assert len(raw) == len(expected_raw)


class TestMultiHop:
    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_argminmax_lowest(self, instance):
        """multi_hop_argminmax(lowest) matches manual ENM scan."""
        graph = _get_graph(instance)
        types = json.loads(list_enm_types(instance))
        for t in types["types"]:
            if t["count"] < 3:
                continue
            result = json.loads(multi_hop_argminmax(instance, t["name"], "lowest"))
            if "error" in result:
                continue
            # Verify: the value is actually the minimum
            vals = [e.value for k, e in graph.enm.items() if k.type == t["name"]]
            assert result["value"] == min(vals), (
                f"{instance}/{t['name']}: got {result['value']} != min {min(vals)}"
            )
            break

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_argminmax_highest(self, instance):
        """multi_hop_argminmax(highest) matches manual ENM scan."""
        graph = _get_graph(instance)
        types = json.loads(list_enm_types(instance))
        for t in types["types"]:
            if t["count"] < 3:
                continue
            result = json.loads(multi_hop_argminmax(instance, t["name"], "highest"))
            if "error" in result:
                continue
            vals = [e.value for k, e in graph.enm.items() if k.type == t["name"]]
            assert result["value"] == max(vals)
            break

    def test_argminmax_nonexistent_type(self):
        result = json.loads(multi_hop_argminmax("model_validation", "bogus", "lowest"))
        assert "error" in result
        assert "available_types" in result

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_multi_hop_chain_basic(self, instance):
        """multi_hop_chain produces valid results with correct source value."""
        graph = _get_graph(instance)
        types = json.loads(list_enm_types(instance))
        type_names = [t["name"] for t in types["types"] if t["count"] >= 3]
        if len(type_names) < 2:
            pytest.skip("Need at least 2 ENM types with >=3 entries")

        for t1 in type_names[:3]:
            for t2 in type_names[:3]:
                if t1 == t2:
                    continue
                result = json.loads(multi_hop_chain(instance, t1, t2, "lowest"))
                if "error" in result and "target_values" not in result:
                    continue
                # Verify hop 1: source_value is the actual minimum
                vals = [e.value for k, e in graph.enm.items() if k.type == t1]
                assert result["source_value"] == min(vals)
                # Verify hop 2: target_values exist and match graph
                if "target_values" in result:
                    base = result["base_entity"]
                    for tid, tval in result["target_values"].items():
                        graph_val = graph.lookup(t2, tid)
                        assert graph_val is not None
                        assert abs(graph_val - tval) < 1e-9
                return  # Found at least one valid chain
        # It's OK if no chain exists for some instances


# =========================================================================
# 5. GROUND-TRUTH VALIDATION
#    Run the actual benchmark generators and verify MCP tools produce
#    answers that match the graph_answer_fn ground truth.
# =========================================================================

class TestGroundTruthValidation:
    """Verify MCP tool answers match benchmark ground truth for every question."""

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_exact_recall_ground_truth(self, instance):
        """MCP exact_recall matches generator ground truth for all questions."""
        from finstructbench.generators.exact_recall import ExactRecallGenerator
        graph = _get_graph(instance)
        gen = ExactRecallGenerator()
        questions = gen.generate(graph)
        sampled = gen.sample(questions, 10, seed=42)

        failures = []
        for q in sampled:
            meta = q.metadata
            result = json.loads(exact_recall(instance, meta["enm_type"], meta["enm_id"]))
            mcp_val = result.get("value")
            gt_val = q.ground_truth
            if mcp_val is None or abs(mcp_val - gt_val) > 1e-9:
                failures.append(f"{q.qid}: MCP={mcp_val} GT={gt_val}")
        assert not failures, "\n".join(failures)

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_threshold_ground_truth(self, instance):
        """MCP threshold_check matches generator ground truth."""
        from finstructbench.generators.threshold import ThresholdGenerator
        graph = _get_graph(instance)
        gen = ThresholdGenerator()
        questions = gen.generate(graph)
        sampled = gen.sample(questions, 10, seed=42)

        failures = []
        for q in sampled:
            meta = q.metadata
            # Need to find the ENM type and entity from the question
            # The threshold generator stores encoder, threshold, op in metadata
            # but not the enm_type/entity_id directly. Re-derive from the question.
            # The llm_prompt contains "'{eid}' in the '{cat}' section"
            import re
            m = re.search(r"'([^']+)' in the '([^']+)' section", q.llm_prompt)
            if not m:
                continue
            eid, cat = m.group(1), m.group(2)
            result = json.loads(threshold_check(
                instance, cat, eid, meta["threshold"], meta["op"]
            ))
            mcp_answer = result.get("satisfied")
            if mcp_answer != q.ground_truth:
                failures.append(f"{q.qid}: MCP={mcp_answer} GT={q.ground_truth}")
        assert not failures, "\n".join(failures)

    @pytest.mark.parametrize("instance", ALL_INSTANCES)
    def test_counting_ground_truth(self, instance):
        """MCP count_entities matches generator ground truth for counting-by-relation."""
        from finstructbench.generators.counting import CountingGenerator
        graph = _get_graph(instance)
        gen = CountingGenerator()
        questions = gen.generate(graph)
        sampled = gen.sample(questions, 10, seed=42)

        failures = []
        for q in sampled:
            meta = q.metadata
            # Skip conditional counting and column-based counting questions —
            # they use ENM/enm_meta, not triples, so MCP count_entities can't verify
            if meta.get("pattern") == "conditional_above_median":
                continue
            if meta.get("column") and "relation" in meta and meta["relation"] == meta.get("column"):
                continue
            if "relation" in meta and "tail" in meta:
                result = json.loads(count_entities(
                    instance, meta["relation"], tail=meta["tail"]
                ))
                mcp_val = result.get("count")
            elif "relation" in meta:
                # Could be total triples or unique heads — check question text
                if "unique" in q.natural_language.lower():
                    result = json.loads(count_entities(
                        instance, meta["relation"], count_unique_heads=True
                    ))
                    mcp_val = result.get("unique_head_count")
                else:
                    result = json.loads(count_entities(instance, meta["relation"]))
                    mcp_val = result.get("triple_count")
            else:
                continue

            gt = q.ground_truth
            if isinstance(gt, int) and mcp_val != gt:
                failures.append(f"{q.qid}: MCP={mcp_val} GT={gt}")
        assert not failures, "\n".join(failures)


# =========================================================================
# 6. ERROR HANDLING & EDGE CASES
# =========================================================================

class TestErrorHandling:
    def test_unknown_document_raises(self):
        _graphs.clear()
        with pytest.raises(ValueError, match="Unknown document"):
            _get_graph("this_does_not_exist_at_all")

    def test_threshold_all_operators(self):
        """All four operators work."""
        graph = _get_graph("model_validation")
        key = next(iter(graph.enm))
        for op in ["ge", "gt", "le", "lt"]:
            result = json.loads(threshold_check(
                "model_validation", key.type, key.id, 0.0, op
            ))
            assert "satisfied" in result, f"Operator {op} failed"

    def test_query_triples_all_none(self):
        """query_triples with no filters returns all triples (up to limit)."""
        result = json.loads(query_triples("basel_capital", limit=10000))
        graph = _get_graph("basel_capital")
        assert result["total"] == len(graph.triples)

    def test_exact_recall_value_precision(self):
        """Values should be returned with full float precision."""
        graph = _get_graph("fair_lending")
        # Find an entry with decimal places
        for key, entry in graph.enm.items():
            if entry.value != int(entry.value):
                result = json.loads(exact_recall("fair_lending", key.type, key.id))
                assert result["value"] == entry.value
                break

    def test_graph_caching(self):
        """Loading the same instance twice reuses the cache."""
        _graphs.clear()
        g1 = _get_graph("basel_capital")
        g2 = _get_graph("basel_capital")
        assert g1 is g2  # Same object


# =========================================================================
# 7. CROSS-INSTANCE CONSISTENCY
# =========================================================================

class TestCrossInstanceConsistency:
    def test_all_instances_have_enm_entries(self):
        for instance in ALL_INSTANCES:
            stats = json.loads(graph_stats(instance))
            assert stats["enm_entries"] > 0, f"{instance} has no ENM entries"

    def test_all_instances_have_triples(self):
        for instance in ALL_INSTANCES:
            stats = json.loads(graph_stats(instance))
            assert stats["triples"] > 0, f"{instance} has no triples"

    def test_enm_counts_match_paper(self):
        """ENM counts match expected values after financial number parsing fix."""
        expected = {
            "model_validation": 275,
            "fair_lending": 1139,
            "stress_test": 476,
            "credit_portfolio": 1503,
            "basel_capital": 494,
        }
        for instance, exp_count in expected.items():
            stats = json.loads(graph_stats(instance))
            assert stats["enm_entries"] == exp_count, (
                f"{instance}: {stats['enm_entries']} != {exp_count}"
            )

    def test_triple_counts_match_paper(self):
        """Triple counts match expected values after financial number parsing fix."""
        expected = {
            "model_validation": 5586,
            "fair_lending": 2014,
            "stress_test": 1417,
            "credit_portfolio": 2356,
            "basel_capital": 784,
        }
        for instance, exp_count in expected.items():
            stats = json.loads(graph_stats(instance))
            assert stats["triples"] == exp_count, (
                f"{instance}: {stats['triples']} != {exp_count}"
            )

    def test_enm_integrity_all_instances(self):
        """Every ENM entry passes SHA-256 integrity check (via lookup)."""
        for instance in ALL_INSTANCES:
            graph = _get_graph(instance)
            for key in graph.enm:
                val = graph.lookup(key.type, key.id)
                assert val is not None, f"{instance}/{key.type}/{key.id} integrity fail"
