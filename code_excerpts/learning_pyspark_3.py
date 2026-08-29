#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:21:43 2026

@author: anonyleg
"""
import argparse
import os
from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    # Robust CLI parsing
    parser = argparse.ArgumentParser(description="Two-round Matrix-Matrix Multiplication in PySpark")
    parser.add_argument("input_path",help="Path to the input file with entries 'tag,row,col,val'")
    #il flag --master permette di specificare il cluster manager, se diverso da local
    parser.add_argument("--master", default="local[*]",help="Spark master URL (default: local[*])")
    parser.add_argument("--output", default=None,help="Optional output path; if omitted, prints to stdout")
    args = parser.parse_args()
     
    # Sanity check on local input paths
    if "://" not in args.input_path and not os.path.isfile(args.input_path):
        parser.error(f"input file not found: {args.input_path}")
     
    conf = SparkConf().setAppName("MMM-2Rounds").setMaster(args.master)
    sc   = SparkContext(conf=conf)
     
    # Input RDD: one entry per line, format "tag,row,col,val"
    # Example: read from HDFS instead of the local filesystem
    # raw = sc.textFile("hdfs://namenode:9000/user/student/matrices.txt")
    raw = sc.textFile(args.input_path)
     
    def parse(line):
        tag, r, c, v = line.split(",")
        if tag not in ("A", "B"):
            raise ValueError(f"unexpected matrix tag: {tag!r}")
        return (tag, int(r), int(c), float(v))
     
    entries = raw.map(parse)   # RDD of (tag, row, col, val)
 
    # ----- ROUND 1 -----
     
    # Map: re-key each entry by its join index j
    #   if tag=A: emit (col, ('A', row, val))   --> j = col of A
    #   if tag=B: emit (row, ('B', col, val))   --> j = row of B
    def remap_by_j(e):
        tag, r, c, v = e
        if tag == "A":
            return (c, ("A", r, v))
        else:
            return (r, ("B", c, v))
     
    keyed_by_j = entries.map(remap_by_j)   # narrow
     
    # Shuffle + Reduce: group by join index, then emit partial products
    def cartesian_products(j_and_list):
        j, L = j_and_list
        LA = [(r, v) for (tag, r, v) in L if tag == "A"]
        LB = [(k, v) for (tag, k, v) in L if tag == "B"]
        for (i, a) in LA:
            for (k, b) in LB:
                yield ((i, k), a * b)
     
    partial = (keyed_by_j
               .groupByKey()                # wide: shuffle on j
               .flatMap(cartesian_products))
    
    # ----- ROUND 2 -----
     
    # Map of round 2 is the identity, no code needed:
    # 'partial' is already an RDD keyed by (i, k).
     
    # Shuffle + Reduce: sum all partial products for the same output cell
    result = partial.reduceByKey(lambda a, b: a + b)   # wide: shuffle on (i,k)
     
    # Action: trigger execution
    if args.output:
        result.map(lambda x: f"{x[0][0]},{x[0][1]},{x[1]}") \
              .saveAsTextFile(args.output)
    else:
        for (i, k), c in result.collect():
            print(f"C[{i},{k}] = {c}")
     
    sc.stop()
    
