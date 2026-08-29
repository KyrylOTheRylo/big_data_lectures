#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:09:15 2026

@author: anonyleg
"""
"""
Relational operators on RDDs --- pure MapReduce style.

In this version the algorithms work on a SINGLE RDD that contains the tuples
of both relations, exactly as they would appear once stored together on a
DFS. The Map and Reduce functions process them uniformly, without any
knowledge of how the input was originally split between two tables.

For 'difference' and 'natural join' the origin must be carried as part of
the record itself, exactly as in the lecture pseudo-code. For 'union' and
'intersection' the origin is irrelevant.

Run:
    spark-submit relational_ops_mr_style.py
"""

from pyspark import SparkContext, SparkConf






# =========================================================================
# CLI parsing
# =========================================================================
if __name__ == "__main__":	


    
    conf = SparkConf().setAppName("RelationalOps-MRStyle").setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    
    # =========================================================================
    # UNION  R u S
    #
    # Input on DFS: a single RDD containing the tuples of R and S, mixed.
    # Map:    for each tuple t, emit (t, t).
    # Reduce: for each key t, emit it once  (global deduplication).
    # =========================================================================
    
    # In a real run this single RDD would come from sc.textFile(...) of a path
    # where R and S have been stored together on the DFS.
    input_union = sc.parallelize([
        (1, "Alice"), (2, "Bob"), (3, "Carol"),     # came from R
        (2, "Bob"),   (3, "Carol"), (4, "Dave"),    # came from S
    ])
    
    def union_map(t):
        return [(t, t)]
    
    def union_reduce(key_and_list):
        t, _L = key_and_list
        return [t]                              # emit once per distinct key
    
    union_rs = (input_union
                .flatMap(union_map)             # MAP
                .groupByKey()                   # SHUFFLE / GROUP-BY-KEY
                .flatMap(union_reduce))         # REDUCE
    
    print("UNION   R u S :", sorted(union_rs.collect()))
    
    
    # =========================================================================
    # INTERSECTION  R n S
    #
    # Same input as union (single mixed RDD, no origin tags).
    # Map:    for each tuple t, emit (t, t).
    # Reduce: for key t, emit t iff |L_t| == 2  (i.e. it appeared twice).
    #
    # Caveat: this assumes R and S contain DISTINCT tuples each, which is the
    # usual pure-relational hypothesis (relations as sets, no duplicates).
    # Then a list of size 2 means "appeared in both R and S".
    # =========================================================================
    
    input_inter = input_union   # same mixed RDD as before
    
    def inter_map(t):
        return [(t, t)]
    
    def inter_reduce(key_and_list):
        t, L = key_and_list
        return [t] if sum(1 for _ in L) == 2 else []
    
    inter_rs = (input_inter
                .flatMap(inter_map)
                .groupByKey()
                .flatMap(inter_reduce))
    
    print("INTER   R n S :", sorted(inter_rs.collect()))
    
    
    # =========================================================================
    # SET DIFFERENCE  R \ S
    #
    # Difference is NOT symmetric, so each record on the DFS must remember its
    # origin. This is the "(t, F) with F in {R, S}" notation of the slides.
    #
    # Map:    for each record (t, F), emit (t, F).
    # Reduce: for key t, emit t iff the list of flags is exactly [R].
    # =========================================================================
    
    input_diff = sc.parallelize([
        ((1, "Alice"), "R"), ((2, "Bob"),   "R"), ((3, "Carol"), "R"),
        ((2, "Bob"),   "S"), ((3, "Carol"), "S"), ((4, "Dave"),  "S"),
    ])
    
    def diff_map(record):
        t, flag = record
        return [(t, flag)]
    
    def diff_reduce(key_and_list):
        t, L = key_and_list
        flags = set(L)
        return [t] if flags == {"R"} else []
    
    diff_rs = (input_diff
               .flatMap(diff_map)
               .groupByKey()
               .flatMap(diff_reduce))
    
    print("DIFF    R \\ S :", sorted(diff_rs.collect()))
    
    
    # =========================================================================
    # NATURAL JOIN  Employee(EmpId, Name, DeptName) |><| Dept(DeptName, Manager)
    #
    # Single mixed RDD again: each record carries the origin flag, plus the
    # tuple itself. The Map function picks the join attribute (DeptName in
    # both relations) as the MapReduce key.
    #
    # Map:    for record (t, F):
    #           if F = R (Employee): key = DeptName,
    #                                value = ('R', (EmpId, Name))
    #           if F = S (Dept):     key = DeptName,
    #                                value = ('S', (Manager,))
    # Reduce: for join key b, do the cross product of R-payloads and S-payloads.
    # =========================================================================
    
    input_join = sc.parallelize([
        ((3415, "Harry",   "Finance"),    "R"),
        ((2241, "Sally",   "Sales"),      "R"),
        ((3401, "George",  "Finance"),    "R"),
        ((1257, "Mary",    "HR"),         "R"),
        (("Finance",    "George"),        "S"),
        (("Sales",      "Harriet"),       "S"),
        (("Production", "Charles"),       "S"),
    ])
    
    def join_map(record):
        t, flag = record
        if flag == "R":
            emp_id, name, dept_name = t
            return [(dept_name, ("R", (emp_id, name)))]
        else:
            dept_name, manager = t
            return [(dept_name, ("S", (manager,)))]
    
    def join_reduce(key_and_list):
        b, L = key_and_list
        R_payloads = [payload for (label, payload) in L if label == "R"]
        S_payloads = [payload for (label, payload) in L if label == "S"]
        out = []
        for (emp_id, name) in R_payloads:
            for (manager,) in S_payloads:
                out.append((emp_id, name, b, manager))
        return out
    
    joined = (input_join
              .flatMap(join_map)
              .groupByKey()
              .flatMap(join_reduce))
    
    print("JOIN    R |><| S :", sorted(joined.collect()))
    
    sc.stop()