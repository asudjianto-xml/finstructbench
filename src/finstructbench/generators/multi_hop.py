"""Multi-hop — chained queries that require 2+ precise lookups.

Patterns modeled on real financial analyst workflows:
  1. Column-specific argmax/min → cross-category lookup (single value)
  2. Spread/range within a column
  3. Argmax/min entity name (string)
  4. Conditional filter → count
  5. Argmax/min → cross-category threshold check (bool)
"""

import re
from finstructbench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS
from finstructbench.generators.threshold import _match_encoder, DEFAULT_THRESHOLDS

# Aggregate / summary row names that are trivially the max — skip these
_AGGREGATE_NAMES = re.compile(
    r"^(total|grand total|all|overall|combined|summary|cumulative|"
    r"aggregate|net|portfolio total|total top \d+)$",
    re.I,
)


def _parse_column(entity_id: str) -> tuple[str, str | None]:
    """Split 'entity/.../Column' into (base_entity, column_name)."""
    parts = entity_id.split("/")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return entity_id, None


def _is_aggregate(name: str) -> bool:
    """Return True if name looks like a summary/total row."""
    return bool(_AGGREGATE_NAMES.match(name.strip()))


def _group_by_column(entries: list[tuple[str, float]]) -> dict[str, list[tuple[str, float]]]:
    """Group ENM entries by their column (last path component).

    Returns {column_name: [(base_entity, value, full_eid), ...]}.
    Only includes columns with 3+ non-aggregate entries.
    """
    by_col = {}
    for eid, val in entries:
        base, col = _parse_column(eid)
        if col is None:
            continue
        if _is_aggregate(base):
            continue
        by_col.setdefault(col, []).append((base, val, eid))

    return {c: vs for c, vs in by_col.items() if len(vs) >= 3}


class MultiHopGenerator(QuestionGenerator):
    category = "multi_hop"

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions = []
        idx = 0

        # Group ENM entries by type
        enm_by_type = {}
        for key, entry in list(graph.enm.items()):
            enm_by_type.setdefault(key.type, []).append((key.id, entry.value))

        # ================================================================
        # Pattern 1: Column-specific argmax/min → cross-category lookup
        #
        # "Which entity has the highest AUC in segmentation analysis?
        #  What is that entity's feature importance?"
        #
        # Answer: a single float from the second lookup.
        # ================================================================
        for type1, entries1 in enm_by_type.items():
            cols1 = _group_by_column(entries1)

            for col1, col1_entries in cols1.items():
                sorted_col = sorted(col1_entries, key=lambda x: x[1])

                for which, target_base, target_val, target_eid in [
                    ("lowest", sorted_col[0][0], sorted_col[0][1], sorted_col[0][2]),
                    ("highest", sorted_col[-1][0], sorted_col[-1][1], sorted_col[-1][2]),
                ]:
                    # Look for this entity in a different category
                    for type2, entries2 in enm_by_type.items():
                        if type1 == type2:
                            continue
                        cols2 = _group_by_column(entries2)

                        for col2, col2_entries in cols2.items():
                            # Find matching entity in type2/col2
                            match = [
                                (base, val, eid) for base, val, eid in col2_entries
                                if base == target_base
                            ]
                            if not match:
                                continue

                            lookup_val = match[0][1]
                            t1_label = type1.replace("_", " ")
                            t2_label = type2.replace("_", " ")

                            def make_fn(tp1, c1, w, tp2, c2):
                                def fn(g):
                                    # Step 1: find argmax/min in type1/col1
                                    best_base = None
                                    best_val = None
                                    for k, entry in g.enm.items():
                                        if k.type != tp1:
                                            continue
                                        _, ec = _parse_column(k.id)
                                        if ec != c1:
                                            continue
                                        v = entry.value
                                        b = k.id.split("/")[0]
                                        if best_val is None or \
                                           (w == "lowest" and v < best_val) or \
                                           (w == "highest" and v > best_val):
                                            best_base = b
                                            best_val = v
                                    if best_base is None:
                                        return None
                                    # Step 2: lookup in type2/col2
                                    for k, entry in g.enm.items():
                                        if k.type != tp2:
                                            continue
                                        b2, ec2 = _parse_column(k.id)
                                        if b2 == best_base and ec2 == c2:
                                            return entry.value
                                    return None
                                return fn

                            q = GeneratedQuestion(
                                qid=f"multi_hop_{idx:03d}",
                                category=self.category,
                                natural_language=(
                                    f"Which entity has the {which} '{col1}' "
                                    f"in '{t1_label}'? "
                                    f"What is that entity's '{col2}' "
                                    f"in '{t2_label}'?"
                                ),
                                graph_answer_fn=make_fn(
                                    type1, col1, which, type2, col2
                                ),
                                ground_truth=lookup_val,
                                llm_prompt=(
                                    f"Step 1: In the '{t1_label}' section, find "
                                    f"the entity with the {which} '{col1}' value.\n"
                                    f"Step 2: Look up that same entity's '{col2}' "
                                    f"in the '{t2_label}' section.\n"
                                    f"Report ONLY the final numeric value from Step 2.\n"
                                    f"{ANSWER_FORMATS['float']}"
                                ),
                                answer_type="float",
                                metadata={
                                    "pattern": "argmax_cross_lookup",
                                    "type1": type1, "col1": col1,
                                    "which": which,
                                    "type2": type2, "col2": col2,
                                    "bridge_entity": target_base,
                                },
                            )

                            if self._validate(graph, q):
                                questions.append(q)
                                idx += 1

        # ================================================================
        # Pattern 2: Spread — difference between max and min in a column
        #
        # "What is the range (max − min) of AUC values across all segments?"
        #
        # Answer: a float.
        # ================================================================
        for enm_type, entries in enm_by_type.items():
            cols = _group_by_column(entries)

            for col, col_entries in cols.items():
                vals = [v for _, v, _ in col_entries]
                spread = max(vals) - min(vals)
                if spread < 1e-9:
                    continue

                type_label = enm_type.replace("_", " ")

                def make_spread_fn(tp, c):
                    def fn(g):
                        values = []
                        for k, entry in g.enm.items():
                            if k.type != tp:
                                continue
                            _, ec = _parse_column(k.id)
                            if ec == c:
                                values.append(entry.value)
                        if len(values) < 2:
                            return None
                        return max(values) - min(values)
                    return fn

                q = GeneratedQuestion(
                    qid=f"multi_hop_{idx:03d}",
                    category=self.category,
                    natural_language=(
                        f"What is the range (highest minus lowest) of "
                        f"'{col}' values in '{type_label}'?"
                    ),
                    graph_answer_fn=make_spread_fn(enm_type, col),
                    ground_truth=spread,
                    llm_prompt=(
                        f"In the '{type_label}' section, find ALL '{col}' values. "
                        f"Compute: highest value minus lowest value.\n"
                        f"Report the exact difference.\n"
                        f"{ANSWER_FORMATS['float']}"
                    ),
                    answer_type="float",
                    metadata={
                        "pattern": "spread",
                        "enm_type": enm_type, "column": col,
                    },
                )

                if self._validate(graph, q):
                    questions.append(q)
                    idx += 1

        # ================================================================
        # Pattern 3: Argmax/min entity name (string answer)
        #
        # "Which entity has the highest AUC in segmentation analysis?"
        #
        # Answer: a string (the entity name).
        # ================================================================
        for enm_type, entries in enm_by_type.items():
            cols = _group_by_column(entries)

            for col, col_entries in cols.items():
                sorted_col = sorted(col_entries, key=lambda x: x[1])
                type_label = enm_type.replace("_", " ")

                for which, target_base in [
                    ("lowest", sorted_col[0][0]),
                    ("highest", sorted_col[-1][0]),
                ]:
                    # Only generate if answer is unambiguous
                    if which == "lowest":
                        ties = [b for b, v, _ in col_entries if v == sorted_col[0][1]]
                    else:
                        ties = [b for b, v, _ in col_entries if v == sorted_col[-1][1]]
                    if len(ties) > 1:
                        continue

                    def make_name_fn(tp, c, w):
                        def fn(g):
                            best_base = None
                            best_val = None
                            for k, entry in g.enm.items():
                                if k.type != tp:
                                    continue
                                _, ec = _parse_column(k.id)
                                if ec != c:
                                    continue
                                b = k.id.split("/")[0]
                                v = entry.value
                                if best_val is None or \
                                   (w == "lowest" and v < best_val) or \
                                   (w == "highest" and v > best_val):
                                    best_base = b
                                    best_val = v
                            return best_base
                        return fn

                    q = GeneratedQuestion(
                        qid=f"multi_hop_{idx:03d}",
                        category=self.category,
                        natural_language=(
                            f"Which entity has the {which} '{col}' "
                            f"in '{type_label}'?"
                        ),
                        graph_answer_fn=make_name_fn(enm_type, col, which),
                        ground_truth=target_base,
                        llm_prompt=(
                            f"In the '{type_label}' section, find the entity "
                            f"with the {which} '{col}' value.\n"
                            f"Report ONLY the entity name.\n"
                            f"{ANSWER_FORMATS['str']}"
                        ),
                        answer_type="str",
                        metadata={
                            "pattern": "argmax_name",
                            "enm_type": enm_type, "column": col,
                            "which": which,
                        },
                    )

                    if self._validate(graph, q):
                        questions.append(q)
                        idx += 1

        # ================================================================
        # Pattern 4: Conditional filter → count
        #
        # "How many entities that pass [test] also have [column] >= [threshold]?"
        #
        # Chains: KG triple query → ENM lookup → threshold check → count.
        # Answer: an integer.
        # ================================================================
        # Collect (relation, tail) → set of head entities from KG
        rel_tail_heads: dict[tuple[str, str], set[str]] = {}
        skip_rels = {"in_section", "has_value", "has_effect"}
        for h, r, t in graph.triples:
            if r in skip_rels:
                continue
            rel_tail_heads.setdefault((r, t), set()).add(h)

        for (rel, tail), heads in rel_tail_heads.items():
            if len(heads) < 3:
                continue

            # For each ENM type/column, see how many of these heads have
            # a value that meets a threshold
            for enm_type, entries in enm_by_type.items():
                cols = _group_by_column(entries)
                for col, col_entries in cols.items():
                    # Find entities in both the triple set and this column
                    col_entity_map = {base: eid for base, _, eid in col_entries}
                    overlap = heads & set(col_entity_map.keys())
                    if len(overlap) < 2:
                        continue

                    # Try to find a matching encoder for this column
                    sample_val = next(
                        v for b, v, _ in col_entries if b in overlap
                    )
                    encoder_name = _match_encoder(enm_type, col, sample_val)
                    if encoder_name is None or encoder_name not in graph.phase_encoders:
                        continue

                    thresholds = DEFAULT_THRESHOLDS.get(encoder_name, [])
                    if not thresholds:
                        continue
                    thresh, op, description = thresholds[0]

                    # Count how many overlapping entities meet the threshold
                    count = 0
                    for entity in overlap:
                        eid = col_entity_map[entity]
                        val = graph.lookup(enm_type, eid)
                        if val is not None:
                            sat, _ = graph.check_threshold(
                                encoder_name, val, thresh, op
                            )
                            if sat:
                                count += 1

                    if count < 1:
                        continue

                    op_word = (
                        "at least" if op == "ge" else
                        "greater than" if op == "gt" else
                        "at most" if op == "le" else "less than"
                    )
                    type_label = enm_type.replace("_", " ")

                    def make_cond_fn(rl, tl, tp, cl, col_ents, enc, th, o):
                        def fn(g):
                            # Step 1: get entities with this relation
                            triple_heads = set(
                                h for h, _, _ in g.query_triples(
                                    relation=rl, tail=tl
                                )
                            )
                            # Step 2: for each, check threshold on column
                            n = 0
                            for ent in triple_heads:
                                eid = col_ents.get(ent)
                                if eid is None:
                                    continue
                                v = g.lookup(tp, eid)
                                if v is None:
                                    continue
                                sat, _ = g.check_threshold(enc, v, th, o)
                                if sat:
                                    n += 1
                            return n
                        return fn

                    q = GeneratedQuestion(
                        qid=f"multi_hop_{idx:03d}",
                        category=self.category,
                        natural_language=(
                            f"How many entities that have '{rel}' = '{tail}' "
                            f"also have '{col}' {op_word} {thresh} "
                            f"in '{type_label}'?"
                        ),
                        graph_answer_fn=make_cond_fn(
                            rel, tail, enm_type, col,
                            col_entity_map, encoder_name, thresh, op,
                        ),
                        ground_truth=count,
                        llm_prompt=(
                            f"Step 1: Find all entities where '{rel}' = "
                            f"'{tail}'.\n"
                            f"Step 2: For each of those entities, look up "
                            f"its '{col}' value in the '{type_label}' "
                            f"section.\n"
                            f"Step 3: Count how many have '{col}' {op_word} "
                            f"{thresh}.\n"
                            f"Report ONLY the count.\n"
                            f"{ANSWER_FORMATS['int']}"
                        ),
                        answer_type="int",
                        metadata={
                            "pattern": "conditional_filter_count",
                            "relation": rel, "tail": tail,
                            "enm_type": enm_type, "column": col,
                            "encoder": encoder_name,
                            "threshold": thresh, "op": op,
                        },
                    )

                    if self._validate(graph, q):
                        questions.append(q)
                        idx += 1

        # ================================================================
        # Pattern 5: Argmax/min → cross-category threshold check
        #
        # "Does the entity with the highest AUC in model validation
        #  meet the 4/5ths rule for AIR in fair lending?"
        #
        # Chains: argmax → cross-category lookup → threshold check.
        # Answer: a boolean.
        # ================================================================
        for type1, entries1 in enm_by_type.items():
            cols1 = _group_by_column(entries1)

            for col1, col1_entries in cols1.items():
                sorted_col = sorted(col1_entries, key=lambda x: x[1])

                for which, target_base in [
                    ("lowest", sorted_col[0][0]),
                    ("highest", sorted_col[-1][0]),
                ]:
                    # Check for ties — skip if ambiguous
                    if which == "lowest":
                        ties = [b for b, v, _ in col1_entries
                                if v == sorted_col[0][1]]
                    else:
                        ties = [b for b, v, _ in col1_entries
                                if v == sorted_col[-1][1]]
                    if len(ties) > 1:
                        continue

                    # Look for this entity in a different category
                    for type2, entries2 in enm_by_type.items():
                        if type1 == type2:
                            continue
                        cols2 = _group_by_column(entries2)

                        for col2, col2_entries in cols2.items():
                            match = [
                                (base, val, eid)
                                for base, val, eid in col2_entries
                                if base == target_base
                            ]
                            if not match:
                                continue

                            cross_val = match[0][1]
                            cross_eid = match[0][2]

                            # Try to find a threshold for col2
                            enc_name = _match_encoder(
                                type2, cross_eid, cross_val
                            )
                            if (enc_name is None
                                    or enc_name not in graph.phase_encoders):
                                continue

                            thresholds = DEFAULT_THRESHOLDS.get(enc_name, [])
                            if not thresholds:
                                continue
                            thresh, op, description = thresholds[0]

                            satisfied, _ = graph.check_threshold(
                                enc_name, cross_val, thresh, op
                            )
                            if satisfied is None:
                                continue

                            t1_label = type1.replace("_", " ")
                            t2_label = type2.replace("_", " ")

                            def make_thresh_fn(
                                tp1, c1, w, tp2, c2, enc, th, o
                            ):
                                def fn(g):
                                    # Step 1: argmax/min
                                    best_base = None
                                    best_val = None
                                    for k, entry in g.enm.items():
                                        if k.type != tp1:
                                            continue
                                        _, ec = _parse_column(k.id)
                                        if ec != c1:
                                            continue
                                        b = k.id.split("/")[0]
                                        if _is_aggregate(b):
                                            continue
                                        v = entry.value
                                        if best_val is None or \
                                           (w == "lowest" and v < best_val) or \
                                           (w == "highest" and v > best_val):
                                            best_base = b
                                            best_val = v
                                    if best_base is None:
                                        return None
                                    # Step 2: cross-lookup
                                    for k, entry in g.enm.items():
                                        if k.type != tp2:
                                            continue
                                        b2, ec2 = _parse_column(k.id)
                                        if b2 == best_base and ec2 == c2:
                                            # Step 3: threshold check
                                            sat, _ = g.check_threshold(
                                                enc, entry.value, th, o
                                            )
                                            return sat
                                    return None
                                return fn

                            q = GeneratedQuestion(
                                qid=f"multi_hop_{idx:03d}",
                                category=self.category,
                                natural_language=(
                                    f"Does the entity with the {which} "
                                    f"'{col1}' in '{t1_label}' meet "
                                    f"{description} for '{col2}' "
                                    f"in '{t2_label}'?"
                                ),
                                graph_answer_fn=make_thresh_fn(
                                    type1, col1, which,
                                    type2, col2, enc_name, thresh, op,
                                ),
                                ground_truth=satisfied,
                                llm_prompt=(
                                    f"Step 1: In the '{t1_label}' section, "
                                    f"find the entity with the {which} "
                                    f"'{col1}' value.\n"
                                    f"Step 2: Look up that entity's '{col2}' "
                                    f"in the '{t2_label}' section.\n"
                                    f"Step 3: Is that value {description}?\n"
                                    f"{ANSWER_FORMATS['bool']}"
                                ),
                                answer_type="bool",
                                metadata={
                                    "pattern": "argmax_threshold",
                                    "type1": type1, "col1": col1,
                                    "which": which,
                                    "type2": type2, "col2": col2,
                                    "encoder": enc_name,
                                    "threshold": thresh, "op": op,
                                    "bridge_entity": target_base,
                                },
                            )

                            if self._validate(graph, q):
                                questions.append(q)
                                idx += 1

        return questions
