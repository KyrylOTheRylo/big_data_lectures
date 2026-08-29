
"""
Relational operators on RDDs --- pure MapReduce style.

In this version the algorithms work on a SINGLE RDD that contains the tuples
of all involved relations, each TAGGED with its origin. This is exactly
what an MR algorithm sees on a DFS: a flat stream of records, each carrying
the information needed to be processed correctly --- including which
relation it came from.

Run:
    spark-submit relational_ops_mr_style.py
"""

from pyspark import SparkContext, SparkConf

if __name__ == "__main__":	
    conf = SparkConf().setAppName("RelationalOps-MRStyle").setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    
    # =========================================================================
    # Single tagged RDD shared by union, intersection and difference.
    # Each record is (tuple, origin_flag) with origin_flag in {'R', 'S'}.
    # =========================================================================
    
    mixed_RS = sc.parallelize([
        ((1, "Alice"), "R"), ((2, "Bob"),   "R"), ((3, "Carol"), "R"),
        ((2, "Bob"),   "S"), ((3, "Carol"), "S"), ((4, "Dave"),  "S"),
    ])
    
    
    # =========================================================================
    # UNION  R u S
    #
    # Map:    for each record (t, F), emit (t, F) keyed by t.
    #         The flag F is irrelevant for union but we keep it for uniformity.
    # Reduce: for each key t, emit it once  (global deduplication).
    # =========================================================================
    
    def union_map(record):
        t, flag = record
        return [(t, flag)]
    
    def union_reduce(key_and_list):
        t, _L = key_and_list
        return [t]                              # emit once per distinct key
    
    union_rs = (mixed_RS
                .flatMap(union_map)             # MAP
                .groupByKey()                   # SHUFFLE / GROUP-BY-KEY
                .flatMap(union_reduce))         # REDUCE
    
    print("UNION   R u S :", sorted(union_rs.collect()))
    
    
    # =========================================================================
    # INTERSECTION  R n S
    #
    # Map:    for each record (t, F), emit (t, F).
    # Reduce: for key t, emit t iff both 'R' and 'S' appear among the flags.
    # =========================================================================
    
    def inter_map(record):
        t, flag = record
        return [(t, flag)]
    
    def inter_reduce(key_and_list):
        t, L = key_and_list
        flags = set(L)
        return [t] if "R" in flags and "S" in flags else []
    
    inter_rs = (mixed_RS
                .flatMap(inter_map)
                .groupByKey()
                .flatMap(inter_reduce))
    
    print("INTER   R n S :", sorted(inter_rs.collect()))
    
    
    # =========================================================================
    # SET DIFFERENCE  R \ S
    #
    # Map:    for each record (t, F), emit (t, F).
    # Reduce: for key t, emit t iff the list of flags contains 'R' but not 'S'.
    # =========================================================================
    
    def diff_map(record):
        t, flag = record
        return [(t, flag)]
    
    def diff_reduce(key_and_list):
        t, L = key_and_list
        flags = set(L)
        return [t] if "R" in flags and "S" not in flags else []
    
    diff_rs = (mixed_RS
               .flatMap(diff_map)
               .groupByKey()
               .flatMap(diff_reduce))
    
    print("DIFF    R \\ S :", sorted(diff_rs.collect()))
    
    
    # =========================================================================
    # NATURAL JOIN  Employee(EmpId, Name, DeptName) |><| Dept(DeptName, Manager)
    #
    # Single tagged RDD again, this time on the join example.
    #
    # Map:    for record (t, F):
    #           if F = R (Employee): key = DeptName,
    #                                value = ('R', (EmpId, Name))
    #           if F = S (Dept):     key = DeptName,
    #                                value = ('S', (Manager,))
    # Reduce: for join key b, do the cross product of R-payloads and S-payloads.
    # =========================================================================
    
    mixed_join = sc.parallelize([
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
    
    joined = (mixed_join
              .flatMap(join_map)
              .groupByKey()
              .flatMap(join_reduce))
    
    print("JOIN    R |><| S :", sorted(joined.collect()))
    
    sc.stop()




