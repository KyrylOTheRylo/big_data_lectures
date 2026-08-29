import argparse
import os
import random

from pyspark import SparkConf, SparkContext

if __name__ == "__main__":
    # Robust CLI parsing

    parser = argparse.ArgumentParser(description="Three-round TeraSort in PySpark")
    parser.add_argument("input_path", help="Path to the input file with one number per line")
    parser.add_argument("--N", type=int, required=True, help="Total number of elements in the input")
    parser.add_argument("--p", type=int, required=True, help="Number of buckets in Round 1 (p > 0)")
    # il flag --master permette di specificare il cluster manager, se diverso da local
    parser.add_argument("--master", default="local[*]")
    args = parser.parse_args()

    if "://" not in args.input_path and not os.path.isfile(args.input_path):
        parser.error(f"input file not found: {args.input_path}")
    if args.p <= 0 or args.N <= 0:
        parser.error("N and p must be positive")

    conf = SparkConf().setAppName("TeraSort-3Rounds").setMaster(args.master)
    sc = SparkContext(conf=conf)

    N, p = args.N, args.p
    text = sc.textFile(args.input_path)
    print("Input file read successfully")
    sorted_text_pairs = text.zipWithIndex().map(lambda x: (x[1], float(x[0])))
    print("Input file read successfully")


    def map_round1(pair):
        i, s = pair
        alpha = random.random()
        out = [(i % p, ("OBJECT", s))]
        if alpha < p/N:
            for i in range(p):
                out.append(("PIVOT", s))


        return  out

    mapped1 = sorted_text_pairs.map(map_round1)

    print("Input file read successfully")
