#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:46:18 2026

@author: anonyleg
"""

"""
Inspecting the content of an RDD (for pedagogical reasons).

Even though Spark abstracts away the cluster, sometimes we want to *see*
how data is actually distributed: which records sit in which partition,
how the shuffle redistributes them, what the reduce input looks like.

This script demonstrates three techniques on a small WordCount pipeline:

  1. mapPartitionsWithIndex  --- inspect the content of each partition
  2. glom                    --- materialize each partition as a list
  3. collect on a grouped RDD --- inspect the (key, list) pairs that the
                                 reducer would actually receive

Run:
    spark-submit inspect_rdd.py
"""

from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("InspectRDD").setMaster("local[4]")
# Note: local[4] forces 4 worker threads, so we get 4 partitions by default.
# This makes the partition view more interesting to look at.
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")


def banner(title):
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


# =========================================================================
# A small WordCount pipeline, broken into named steps so we can inspect
# each intermediate RDD.
# =========================================================================

lines = sc.parallelize([
    "the quick brown fox",
    "the lazy dog",
    "the quick fox jumps",
    "over the lazy dog",
], numSlices=4)

words = lines.flatMap(lambda line: line.split())
pairs = words.map(lambda w: (w, 1))
grouped = pairs.groupByKey()
counts  = grouped.mapValues(sum)


# =========================================================================
# Technique 1: mapPartitionsWithIndex
#
# This transformation lets us inspect (and tag) each partition with its
# index. We use it to print a labeled view of the partition contents.
# =========================================================================

banner("Technique 1: mapPartitionsWithIndex")
print("Goal: see exactly which records sit in which partition.")
print()

def label_with_partition(idx, iterator):
    # iterator is a Python iterator over the records in partition #idx
    for rec in iterator:
        yield (idx, rec)

print("Lines, partition by partition:")
for idx, line in lines.mapPartitionsWithIndex(label_with_partition).collect():
    print(f"  [partition {idx}]  {line!r}")

print()
print("(word, 1) pairs, partition by partition (after flatMap+map):")
for idx, p in pairs.mapPartitionsWithIndex(label_with_partition).collect():
    print(f"  [partition {idx}]  {p}")


# =========================================================================
# Technique 2: glom
#
# glom() returns one ARRAY per partition, instead of streaming the elements.
# This is the simplest way to see "what's in each partition" without
# writing a helper function.
# =========================================================================

banner("Technique 2: glom")
print("Goal: same as above, but more compact (one list per partition).")
print()

print("Lines (one row per partition):")
for idx, partition in enumerate(lines.glom().collect()):
    print(f"  [partition {idx}]  {partition}")

print()
print("(word, 1) pairs (one row per partition):")
for idx, partition in enumerate(pairs.glom().collect()):
    print(f"  [partition {idx}]  {partition}")


# =========================================================================
# Technique 3: inspecting the input of the reduce phase
#
# After groupByKey() each record is a (key, ResultIterable). Spark gives
# us this RDD, but we cannot print a ResultIterable directly --- we need
# to materialize it into a list first.
#
# This is exactly what the Reduce function would see for each key.
# =========================================================================

banner("Technique 3: reduce input (after groupByKey)")
print("Goal: see the key-list pairs that each reducer call processes.")
print()

reduce_input = grouped.mapValues(list).collect()
for k, L in sorted(reduce_input):
    print(f"  key = {k!r:14}   list = {L}")


# =========================================================================
# Bonus: inspecting how keys are distributed across reduce partitions
# =========================================================================

banner("Bonus: how keys are distributed across reducer partitions")
print("Goal: see which reducer would handle which keys after the shuffle.")
print()

def label_keys(idx, iterator):
    for k, L in iterator:
        yield (idx, k, list(L))

key_distribution = grouped.mapPartitionsWithIndex(label_keys).collect()
for idx, k, L in sorted(key_distribution):
    print(f"  [reducer partition {idx}]  key = {k!r:14}   list = {L}")


# =========================================================================
# Final result for reference
# =========================================================================

banner("Final WordCount result")
for w, c in sorted(counts.collect()):
    print(f"  {w}: {c}")

sc.stop()