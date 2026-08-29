import argparse
import os
import random

from pyspark import SparkConf, SparkContext


if __name__ == "__main__":
    # Robust CLI parsing
    parser = argparse.ArgumentParser(description="Two-round Matrix-Matrix Multiplication in PySpark")
    parser.add_argument("input_path", help="Path to the input file with entries 'tag,row,col,val'")
    # il flag --master permette di specificare il cluster manager, se diverso da local
    parser.add_argument("--master", default="local[*]", help="Spark master URL (default: local[*])")
    parser.add_argument("--output", default=None, help="Optional output path; if omitted, prints to stdout")
    args = parser.parse_args()

    # Sanity check on local input paths
    if "://" not in args.input_path and not os.path.isfile(args.input_path):
        parser.error(f"input file not found: {args.input_path}")

    conf = SparkConf().setAppName("MMM-2Rounds").setMaster(args.master)
    sc = SparkContext(conf=conf)

    # Input RDD: one entry per line, format "tag,row,col,val"
    # Example: read from HDFS instead of the local filesystem
    # raw = sc.textFile("hdfs://namenode:9000/user/student/matrices.txt")
    raw = sc.textFile(args.input_path)


    def parse(line):
        tag, r, c, v = line.split(",")
        if tag not in ("A", "B"):
            raise ValueError(f"unexpected matrix tag: {tag!r}")
        return tag, int(r), int(c), float(v)


    entries = raw.map(parse)  # RDD of (tag, row, col, val)

    # Compute matrix dimensions
    n = entries.filter(lambda e: e[0] == "A").map(lambda e: e[1]).max() + 1
    m = entries.filter(lambda e: e[0] == "B").map(lambda e: e[2]).max() + 1


    entries = raw.map(parse)  # RDD of (tag, row, col, val)


    # ----- ROUND 1 -----

    # Map: re-key each entry by its join index j
    #   if tag=A: emit (col, ('A', row, val))   --> j = col of A
    #   if tag=B: emit (row, ('B', col, val))   --> j = row of B
    def remap_by_j(e):
        tag, r, c, v = e
        results = []
        if tag == "A":
            for k in range(m):
                results.append(((r, k), ("A", c, v)))
        else:
            for i in range(n):
                results.append(((i, c), ("B", r, v)))
        return results

    mapped_entries = entries.flatMap(remap_by_j)

    def cartesian_products(j_and_list):
        j, L = j_and_list
        La = []
        Lb = []
        for tag, ind, val in L:
            if tag == "A":
                La.append((tag, ind, val))
            else:
                Lb.append((tag, ind, val))
        #     sort La and Lb by ind
        La.sort(key=lambda x: x[1])
        Lb.sort(key=lambda x: x[1])

        x = 0
        for ind in range(len(La)):
            x += La[ind][2]*Lb[ind][2]

        return j , x


    partial = (mapped_entries.groupByKey()# wide: shuffle on j
               .map(cartesian_products))  # wide: shuffle on (i,k)

    print(partial)


