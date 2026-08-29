# Created on Mon Feb 18 17:01:27 2019
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anonym

WordCount example in PySpark.

This version compares:

1. GOOD WordCount:
   flatMap -> map -> reduceByKey

2. BAD WordCount:
   flatMap -> map -> groupByKey -> mapValues(sum)

The two versions produce the same logical result,
but reduceByKey is usually much more efficient.
"""

from __future__ import print_function

from operator import add
from pyspark.sql import SparkSession

import argparse
import string
import time


def clean_word(word, table):
    """
    Normalize a word by removing punctuation and converting to lowercase.
    """
    return word.translate(table).lower()


def print_sample(title, rdd, n=10):
    """
    Print a small sample of the result.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    output = rdd.take(n)

    for word, frequency in output:
        print("frequency[" + str(word) + "]:", frequency)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--f",
        metavar="filepath",
        required=True,
        default="",
        help="Path to the file to wordcount"
    )

    args = parser.parse_args()

    filename = str(args.f)

    if filename == "":
        raise Exception("Empty file")

    print("Processing:", filename)

    spark = (
        SparkSession
        .builder
        .appName("PythonWordCountGoodVsBad")
        .getOrCreate()
    )

    sc = spark.sparkContext

    # Optional: reduce Spark log noise
    sc.setLogLevel("WARN")

    # Read file as DataFrame, then convert to RDD
    lines = spark.read.text(filename).rdd

    # Each row is a Spark Row object.
    # r[0] extracts the text column.
    lines = lines.map(lambda r: r[0].strip())

    # Remove punctuation and selected special characters
    table = str.maketrans(
        "",
        "",
        string.punctuation + "“‘—”’£æϰητος"
    )

    # ---------------------------------------------------------------------
    # Common preprocessing step
    # ---------------------------------------------------------------------
    # flatMap splits each line into many words.
    #
    # Example:
    # Input line: "hello world hello"
    # Output:     "hello", "world", "hello"
    #
    # Then map turns each word into a pair:
    # "hello" -> ("hello", 1)
    # ---------------------------------------------------------------------

    words = (
        lines
        .flatMap(lambda line: line.split(" "))
        .map(lambda word: clean_word(word, table))
        .filter(lambda word: word != "")
    )

    word_pairs = words.map(lambda word: (word, 1))

    # Persist because both GOOD and BAD versions reuse the same input RDD.
    # This avoids recomputing the parsing pipeline twice.
    word_pairs.persist()

    # Force Spark to materialize the persisted RDD.
    total_words = word_pairs.count()
    print("Total words:", total_words)

    # =====================================================================
    # GOOD WORDCOUNT
    # =====================================================================
    #
    # reduceByKey performs local aggregation before shuffling data.
    #
    # For example, inside one partition:
    #
    # ("spark", 1), ("spark", 1), ("spark", 1)
    #
    # can become:
    #
    # ("spark", 3)
    #
    # before being sent across the network.
    #
    # This reduces shuffle size and is usually much faster.
    # =====================================================================

    start_good = time.time()

    counts_good = word_pairs.reduceByKey(add)

    # Action: force execution
    output_good = counts_good.take(10)

    end_good = time.time()

    print("\nGOOD WordCount using reduceByKey")
    print("--------------------------------")
    for word, frequency in output_good:
        print("frequency[" + str(word) + "]:", frequency)

    print("GOOD elapsed time: %.4f seconds" % (end_good - start_good))

    # =====================================================================
    # BAD WORDCOUNT
    # =====================================================================
    #
    # groupByKey sends all values for each key across the network.
    #
    # For example:
    #
    # ("spark", 1), ("spark", 1), ("spark", 1)
    #
    # becomes:
    #
    # ("spark", [1, 1, 1])
    #
    # and only afterwards do we sum the list.
    #
    # This can be very expensive because Spark must shuffle and store
    # all individual values.
    # =====================================================================

    start_bad = time.time()

    counts_bad = (
        word_pairs
        .groupByKey()
        .mapValues(sum)
    )

    # Action: force execution
    output_bad = counts_bad.take(10)

    end_bad = time.time()

    print("\nBAD WordCount using groupByKey + sum")
    print("------------------------------------")
    for word, frequency in output_bad:
        print("frequency[" + str(word) + "]:", frequency)

    print("BAD elapsed time: %.4f seconds" % (end_bad - start_bad))

    # =====================================================================
    # Optional correctness check
    # =====================================================================
    #
    # Both versions should compute the same frequencies.
    # We compare a small sorted sample.
    # =====================================================================

    print("\nCorrectness check")
    print("-----------------")

    top_good = counts_good.takeOrdered(10, key=lambda x: -x[1])
    top_bad = counts_bad.takeOrdered(10, key=lambda x: -x[1])

    print("\nTop 10 words from GOOD version:")
    for word, frequency in top_good:
        print(word, frequency)

    print("\nTop 10 words from BAD version:")
    for word, frequency in top_bad:
        print(word, frequency)

    if top_good == top_bad:
        print("\nThe two versions match on the top 10 words.")
    else:
        print("\nThe two versions do not have the same top 10 ordering.")
        print("This may happen when words have equal frequencies.")

    word_pairs.unpersist()

    spark.stop()