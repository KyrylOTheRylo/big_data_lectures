#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:09:15 2026

@author: anonyleg
"""


from pyspark import SparkContext, SparkConf


if __name__ == "__main__":	
    
    conf = SparkConf().setAppName("RelationalOps").setMaster("local[*]")
    sc   = SparkContext(conf=conf)
     
    R = sc.parallelize([(1, "Alice"), (2, "Bob"),   (3, "Carol")])
    S = sc.parallelize([(2, "Bob"),   (3, "Carol"), (4, "Dave")])

    union_rs = R.union(S).distinct()
    print(union_rs.collect())
    # [(1, 'Alice'), (2, 'Bob'), (3, 'Carol'), (4, 'Dave')]

    inter_rs = R.intersection(S)
    print(inter_rs.collect())
    # [(2, 'Bob'), (3, 'Carol')]

    diff_rs = R.subtract(S)
    print(diff_rs.collect())
    # [(1, 'Alice')]
    # Employees(EmpId, Name, DeptName)
    emp = sc.parallelize([
        (3415, "Harry",   "Finance"),
        (2241, "Sally",   "Sales"),
        (3401, "George",  "Finance"),
        (1257, "Mary",    "HR"),
    ])
     
    # Depts(DeptName, Manager)
    dept = sc.parallelize([
        ("Finance",    "George"),
        ("Sales",      "Harriet"),
        ("Production", "Charles"),
    ])
    # Re-key each side by the join attribute (DeptName)
    emp_by_dept  = emp.map(lambda t: (t[2], (t[0], t[1])))   # (Dept, (Id, Name))
    dept_by_name = dept.map(lambda t: (t[0], t[1]))          # (Dept, Manager)
     
    joined = emp_by_dept.join(dept_by_name)
    # joined: (DeptName, ((EmpId, Name), Manager))
     
    # Flatten to a relational tuple (EmpId, Name, DeptName, Manager)
    result = joined.map(lambda kv: (kv[1][0][0], kv[1][0][1], kv[0], kv[1][1]))
    print(result.collect())
    # [(3415, 'Harry',  'Finance', 'George'),
    #  (3401, 'George', 'Finance', 'George'),
    #  (2241, 'Sally',  'Sales',   'Harriet')]

    sc.stop()