import argparse
import os
from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    # Robust CLI parsing: argparse gives us help, type checks, error messages
    parser = argparse.ArgumentParser(description="WordCount in PySpark")
    parser.add_argument("input_path",help="Path to the input text file (local or hdfs://)")
    #il flag --master permette di specificare il cluster manager, se diverso da local

    parser.add_argument("--master", default="local[*]",help="Spark master URL (default: local[*])")
    args = parser.parse_args()
    
    # Sanity check: for local paths, verify the file exists.
    # For HDFS / S3 / ... we leave the check to Spark itself.
    if "://" not in args.input_path and not os.path.isfile(args.input_path):
        parser.error(f"input file not found: {args.input_path}")
    
    # 1) Create the Spark context (entry point of any Spark application)
    conf = SparkConf().setAppName("WordCount").setMaster(args.master)
    sc   = SparkContext(conf=conf)
    
    # 2) Build the input RDD: one element per line of the input file
    lines = sc.textFile(args.input_path)
    # Example: read from HDFS instead of the local filesystem
    # lines = sc.textFile("hdfs://namenode:9000/user/student/input.txt")
    
    # 3) Map phase: tokenize and emit (word, 1) for each word
    pairs = (lines
    .flatMap(lambda line: line.split())   # narrow: tokenize
    .map(lambda w: (w, 1)))               # narrow: attach count 1
    
    # 4) Shuffle + Reduce phase: sum counts grouped by word
    counts = pairs.reduceByKey(lambda a, b: a + b) # wide: shuffle barrier
    
    # 5) Action: trigger computation and collect the result
    for word, count in counts.collect():
        print(f"{word}: {count}")
    
    sc.stop()