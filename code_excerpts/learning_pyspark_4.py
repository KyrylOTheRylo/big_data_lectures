#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:26:41 2026

@author: anonyleg
"""

import argparse
import os
import random
from pyspark import SparkContext, SparkConf
if __name__ == "__main__":	
	# Robust CLI parsing

    parser = argparse.ArgumentParser(description="Three-round TeraSort in PySpark")
    parser.add_argument("input_path",help="Path to the input file with one number per line")
    parser.add_argument("--N", type=int, required=True,help="Total number of elements in the input")
    parser.add_argument("--p", type=int, required=True,help="Number of buckets in Round 1 (p > 0)")
    #il flag --master permette di specificare il cluster manager, se diverso da local
    parser.add_argument("--master", default="local[*]")
    args = parser.parse_args()
		

    if "://" not in args.input_path and not os.path.isfile(args.input_path):
        parser.error(f"input file not found: {args.input_path}")
    if args.p <= 0 or args.N <= 0:
        parser.error("N and p must be positive")
 
    conf = SparkConf().setAppName("TeraSort-3Rounds").setMaster(args.master)
    sc   = SparkContext(conf=conf)
 
    N, p = args.N, args.p
     
    # Input RDD: one number per line --> RDD of (id, value) pairs
    raw = sc.textFile(args.input_path)
    elements = raw.zipWithIndex().map(lambda x: (x[1], float(x[0])))
    # elements is an RDD of (id, s)
    # ----- ROUND 1 -----
     
    # Map: send element to bucket (i mod p); with prob. p/N, also replicate
    # the element as a pivot to all p buckets.
    def round1_map(pair):
        i, s = pair
        out = [(i % p, ("OBJECT", s))]      # always emit the object
        if random.random() <= p / N:        # sampled as pivot
            for w in range(p):
                out.append((w, ("PIVOT", s)))
        return out
 
    mapped1 = elements.flatMap(round1_map)  # narrow
     
    # Reduce: split objects/pivots, sort pivots, partition objects by interval
    def round1_reduce(j_and_list):
        j, L = j_and_list
        objects = [s for (tag, s) in L if tag == "OBJECT"]
        pivots  = sorted(s for (tag, s) in L if tag == "PIVOT")
        # Add sentinels for the open intervals at both ends
        boundaries = [float("-inf")] + pivots + [float("inf")]
        for s in objects:
            # find interval i such that boundaries[i] < s <= boundaries[i+1]
            for i in range(len(boundaries) - 1):
                if boundaries[i] < s <= boundaries[i + 1]:
                    yield (i, s)
                    break
     
    intervals = mapped1.groupByKey().flatMap(round1_reduce)  # wide: shuffle on bucket j
    # ----- ROUND 2 -----
    # Map of Round 2 is the identity: 'intervals' is already keyed by i.
     
    # Reduce: for each interval, emit the cardinality + all the elements
    def round2_reduce(i_and_list):
        i, L = i_and_list
        L = list(L)
        yield (i, ("CARD", i, len(L)))     # cardinality entry
        for s in L:
            yield (i, ("ELEM", s))         # tagged element
     
    with_card = intervals.groupByKey().flatMap(round2_reduce)
    
    # ----- ROUND 3 -----
 
    # eta + 1 = number of intervals = number of pivots + 1.
    # We need it to know how many copies of each cardinality to produce.
    # We compute it from the data (cheap: small action on the cardinalities).
    eta_plus_1 = (with_card.filter(lambda kv: kv[1][0] == "CARD").count())
 
    # Map: forward elements unchanged; replicate each cardinality eta+1 times
    def round3_map(kv):
        i, val = kv
        if val[0] == "CARD":
            for w in range(eta_plus_1):
                yield (w, val)             # ("CARD", i, N_i) replicated
        else:                              # val[0] == "ELEM"
            yield (i, val)                 # forward element
 
    mapped3 = with_card.flatMap(round3_map)  # narrow
 
    # Reduce: sort local elements, compute starting rank from cardinalities,
    # emit (global_rank, s) pairs.
    def round3_reduce(k_and_list):
        k, L = k_and_list
        elems = sorted(v[1] for v in L if v[0] == "ELEM")
        cards = {v[1]: v[2] for v in L if v[0] == "CARD"}
        # Starting rank of this reducer
        R = 1 + sum(cards[ell] for ell in range(k))
        for pi, s in enumerate(elems):
            yield (R + pi, s)
 
    sorted_pairs = mapped3.groupByKey().flatMap(round3_reduce)
    # wide: shuffle on interval index
     
    # Action: trigger and print the globally sorted sequence
    for rank, s in sorted(sorted_pairs.collect()):
        print(f"{rank}: {s}")
 
    sc.stop()
