#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:09:15 2026

@author: anonyleg
"""


"""
TeraSort (3-round) in PySpark --- EXTENDED VERSION FOR LAB INSPECTION

This script is functionally equivalent to the slide version, but it is
*instrumented* to make the MapReduce semantics explicit:
 
  (a) Tactical prints to inspect intermediate state (pivots, interval
      assignments, cardinalities, ...).

  (b) Per-round MapReduce performance metrics:
        - local space  M_L      = max list size passed to any reducer
        - aggregate space M_A   = sum of list sizes passed to all reducers
                                 = total amount of data read by the reduce phase
        - replication rate r    = (size of RDD AFTER map) / (size of RDD BEFORE map)

The script supports two input element types: numbers and arbitrary strings
(the algorithm only needs total ordering). Pick the parsing mode with --type.
"""

import argparse
import os
import random
from pyspark import SparkContext, SparkConf


# =========================================================================
# Helpers for diagnostics and metrics
# =========================================================================

def banner(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def measure_round(name, rdd_before_map, rdd_after_map, rdd_before_reduce):
    """
    Compute and print the three classic MR metrics for one round.

      replication_rate = |rdd_after_map| / |rdd_before_map|
      local_space      = max_k |L_k|        (max list size at any reducer)
      aggregate_space  = sum_k |L_k|        (= |rdd_after_map|, but recomputed
                                              from the grouped RDD as a sanity check)

    rdd_before_reduce must be a (key, iterable) RDD, i.e. the output of a
    groupByKey() before the user-defined reduce logic is applied.
    """
    banner(f"Metrics --- {name}")

    n_in  = rdd_before_map.count()
    n_out = rdd_after_map.count()
    rep_rate = n_out / n_in if n_in > 0 else float("nan")

    # For each key, compute the size of its value list. We need to materialize
    # the iterable, so we map it to its length.
    sizes_rdd = rdd_before_reduce.map(lambda kv: (kv[0], sum(1 for _ in kv[1])))
    sizes = sizes_rdd.collect()
    if not sizes:
        print("  (empty round)")
        return
    local_space     = max(sz for _, sz in sizes)
    aggregate_space = sum(sz for _, sz in sizes)
    n_reducers      = len(sizes)

    print(f"  Input size  (before Map) : {n_in}")
    print(f"  Output size (after  Map) : {n_out}")
    print(f"  Replication rate r       : {rep_rate:.3f}")
    print(f"  Number of reducer keys   : {n_reducers}")
    print(f"  Local space  M_L (max)   : {local_space}")
    print(f"  Aggregate space M_A (sum): {aggregate_space}")
    print(f"  Per-key list sizes       : {sorted(sizes)}")



# =========================================================================
# CLI parsing
# =========================================================================
if __name__ == "__main__":	
    

    # =========================================================================
    # CLI parsing
    # =========================================================================

    parser = argparse.ArgumentParser(
        description="Three-round TeraSort in PySpark (instrumented)")
    parser.add_argument("input_path",
                        help="Path to the input file (one element per line)")
    parser.add_argument("--N", type=int, required=True,
                        help="Total number of elements in the input")
    parser.add_argument("--p", type=int, required=True,
                        help="Number of buckets in Round 1 (p > 0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for reproducibility")
    parser.add_argument("--master", default="local[*]")
    args = parser.parse_args()
    
    if "://" not in args.input_path and not os.path.isfile(args.input_path):
        parser.error(f"input file not found: {args.input_path}")
    if args.p <= 0 or args.N <= 0:
        parser.error("N and p must be positive")
    
    # NOTE: random.seed only seeds the *driver* RNG. Each Spark task uses its
    # own Python interpreter, so reproducibility across distributed runs would
    # require seeding inside mapPartitions. For local[*] runs it works as long
    # as we use plain map (not mapPartitions).
    if args.seed is not None:
        random.seed(args.seed)
    
    conf = SparkConf().setAppName("TeraSort-3Rounds-Instrumented") \
                      .setMaster(args.master)
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")  # quiet down INFO chatter
    
    N, p = args.N, args.p
    
    
    # =========================================================================
    # Detect input type at startup
    # =========================================================================
    #
    # We inspect the first non-empty line of the input file: if it parses as a
    # float we treat the whole file as numbers, otherwise as strings. This way
    # the user does not have to specify --type explicitly.
    #
    # We do this with plain Python (not Spark) on the driver, on a single line.
    # Note: this assumes args.input_path is locally readable; for HDFS/S3 paths
    # the user should add an explicit --type-like flag, but for the lab this is
    # sufficient.
    
    def detect_input_type(path):
        if "://" in path:
            # Cannot peek a remote URI from plain Python; default to numbers and
            # let parsing fail loudly if needed.
            return "number"
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    float(line)
                    return "number"
                except ValueError:
                    return "string"
        raise ValueError(f"input file {path} appears to be empty")
    
    input_type = detect_input_type(args.input_path)
    print(f"[startup] detected input type: {input_type}")
    
    
    # =========================================================================
    # Input loading
    # =========================================================================
    
    def parse_value(line):
        line = line.strip()
        return float(line) if input_type == "number" else line
    
    raw      = sc.textFile(args.input_path)
    elements = raw.zipWithIndex().map(lambda x: (x[1], parse_value(x[0])))
    elements.cache()  # we will reuse it for measurements; keep it in memory
    
    banner("Input")
    print(f"  Total elements N = {elements.count()}")
    print(f"  First 5 (id, value): {elements.take(5)}")
    
    
    # =========================================================================
    # ROUND 1
    # =========================================================================
    
    banner("ROUND 1 --- Map: bucket assignment + pivot sampling")
    
    def round1_map(pair):
        i, s = pair
        out = [(i % p, ("OBJECT", s))]
        if random.random() <= p / N:
            for w in range(p):
                out.append((w, ("PIVOT", s)))
        return out
    
    mapped1 = elements.flatMap(round1_map)
    mapped1.cache()
    
    # Diagnostic: how many pivots were sampled?
    sampled_pivots = sorted({s for (_k, (tag, s)) in mapped1.collect()
                             if tag == "PIVOT"})
    print(f"  Sampled pivots ({len(sampled_pivots)}): {sampled_pivots}")
    print(f"  (expected ~ p = {p}, since each element is a pivot with prob p/N)")
    
    # Group by bucket id
    grouped1 = mapped1.groupByKey()
    grouped1.cache()
    
    # Metrics for Round 1
    measure_round("Round 1", elements, mapped1, grouped1)
    
    # Reduce body: split objects/pivots, sort pivots, partition objects by interval
    def round1_reduce(j_and_list):
        j, L = j_and_list
        L = list(L)
        objects = [s for (tag, s) in L if tag == "OBJECT"]
        pivots  = sorted(s for (tag, s) in L if tag == "PIVOT")
        boundaries = [None] + pivots + [None]  # None sentinels = +/- infty
        # Diagnostic: show what each bucket reducer sees
        print(f"  [bucket {j}] |objects|={len(objects)}  pivots={pivots}")
        out = []
        for s in objects:
            for i in range(len(boundaries) - 1):
                lo, hi = boundaries[i], boundaries[i + 1]
                if (lo is None or lo < s) and (hi is None or s <= hi):
                    out.append((i, s))
                    break
        return out
    
    intervals = grouped1.flatMap(round1_reduce)
    intervals.cache()
    print()
    print(f"  Round 1 output (first 10 (interval, value) pairs): "
          f"{intervals.take(10)}")
    
    
    # =========================================================================
    # ROUND 2
    # =========================================================================
    
    banner("ROUND 2 --- Reduce: count per interval + forward elements")
    
    # Map of Round 2 is the identity; for measurements we treat 'intervals' as
    # the input and the output of Map as the same RDD.
    mapped2  = intervals
    grouped2 = mapped2.groupByKey()
    grouped2.cache()
    
    measure_round("Round 2", intervals, mapped2, grouped2)
    
    def round2_reduce(i_and_list):
        i, L = i_and_list
        L = list(L)
        yield (i, ("CARD", i, len(L)))
        for s in L:
            yield (i, ("ELEM", s))
    
    with_card = grouped2.flatMap(round2_reduce)
    with_card.cache()
    
    # Diagnostic: show all cardinalities
    cardinalities = sorted(
        (v[1], v[2])
        for (_k, v) in with_card.collect()
        if v[0] == "CARD"
    )
    print()
    print(f"  Cardinalities (interval, N_i): {cardinalities}")
    print(f"  Sum of cardinalities         : {sum(N_i for _, N_i in cardinalities)}"
          f" (should equal N = {N})")
    
    
    # =========================================================================
    # ROUND 3
    # =========================================================================
    
    banner("ROUND 3 --- Map: replicate cardinalities; Reduce: global ranking")
    
    eta_plus_1 = (with_card
                  .filter(lambda kv: kv[1][0] == "CARD")
                  .count())
    print(f"  eta + 1 (number of intervals) = {eta_plus_1}")
    
    def round3_map(kv):
        i, val = kv
        if val[0] == "CARD":
            for w in range(eta_plus_1):
                yield (w, val)
        else:
            yield (i, val)
    
    mapped3 = with_card.flatMap(round3_map)
    mapped3.cache()
    grouped3 = mapped3.groupByKey()
    grouped3.cache()
    
    measure_round("Round 3", with_card, mapped3, grouped3)
    
    def round3_reduce(k_and_list):
        k, L = k_and_list
        L = list(L)
        elems = sorted(v[1] for v in L if v[0] == "ELEM")
        cards = {v[1]: v[2] for v in L if v[0] == "CARD"}
        R = 1 + sum(cards[ell] for ell in range(k))
        print(f"  [reducer {k}] |elems|={len(elems)}  R={R}  cards={cards}")
        return [(R + pi, s) for pi, s in enumerate(elems)]
    
    sorted_pairs = grouped3.flatMap(round3_reduce)
    
    
    # =========================================================================
    # Final output
    # =========================================================================
    
    banner("Final globally sorted output")
    final = sorted(sorted_pairs.collect())
    for rank, s in final:
        print(f"  {rank}: {s}")
    
    # Sanity check
    sorted_values = [s for _, s in final]
    expected = sorted([s for _, s in elements.collect()])
    print()
    print(f"  Sanity check: sorted output matches Python sorted(): "
          f"{sorted_values == expected}")
    
    sc.stop()