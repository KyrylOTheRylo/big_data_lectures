"""
TERASORT ALGORITHM WITH MAPREDUCE IN PYSPARK
=============================================

OVERVIEW:
TeraSort is a benchmark algorithm that sorts 1 terabyte of data and is used to measure 
the performance of sorting implementations. It's particularly useful for testing distributed 
computing systems like Hadoop and Spark.

KEY CONCEPTS:
1. MapReduce: Consists of Map, Shuffle, and Reduce phases
   - Map:     Transforms input data into key-value pairs
   - Shuffle: Groups and sorts by key (automatic in Spark)
   - Reduce:  Aggregates values for each key

2. TeraSort: A distributed sorting algorithm with 3 phases:
   - Sampling:   Determines partition boundaries from data sample
   - Partition:  Uses boundaries to distribute data across partitions
   - Sort:       Each partition is sorted (Spark handles this automatically)

3. Why it's useful:
   - Tests full capabilities of a distributed system
   - I/O intensive (exercises disk and network)
   - CPU-bound sorting operations
   - Good benchmark for MapReduce/Spark frameworks
"""

from pyspark.sql import SparkSession
from pyspark.rdd import RDD
import random
import sys
from typing import List, Tuple


class TeraSort:
    """
    TeraSort implementation in PySpark.
    
    This class implements the TeraSort algorithm which sorts terabytes of data
    across a distributed cluster using MapReduce principles.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize TeraSort with a Spark session.
        
        Args:
            spark: SparkSession object for distributed computing
        """
        self.spark = spark
        self.sc = spark.sparkContext
    
    def generate_terabyte_data(self, num_records: int) -> RDD:
        """
        Generate sample data for TeraSort. In production, this would be huge files.
        
        Each record is 100 bytes:
        - 10-byte key (random hex string)
        - 90-byte value (padding)
        
        Args:
            num_records: Number of records to generate (100M records = ~10GB)
        
        Returns:
            RDD of key-value pairs (key, value)
        
        EXPLANATION:
        - We generate records distributed across all Spark partitions
        - Each record has a 10-byte key and 90-byte value (100 bytes total = TeraSort spec)
        - Random generation simulates unsorted input data
        """
        
        def generate_partition(partition_id, num_per_partition):
            """Generate records for this partition."""
            random.seed(partition_id)  # Ensure reproducibility per partition
            for i in range(num_per_partition):
                # Generate 10-byte key (20 hex characters = 10 bytes)
                key = ''.join(random.choices('0123456789ABCDEF', k=20))
                # Generate 90-byte value (180 hex characters = 90 bytes)
                value = ''.join(random.choices('0123456789ABCDEF', k=180))
                yield (key, value)
        
        # Distribute data generation across partitions
        num_partitions = self.sc.defaultParallelism
        num_per_partition = num_records // num_partitions
        
        rdd = self.sc.parallelize(
            range(num_partitions), 
            num_partitions
        ).flatMap(lambda pid: generate_partition(pid, num_per_partition))
        
        return rdd
    
    def sample_for_partition_boundaries(self, rdd: RDD, sample_size: int = 1000) -> List[str]:
        """
        PHASE 1: SAMPLING
        ===============
        Extract a sample of keys to determine partition boundaries.
        
        This is the first key phase of TeraSort. We sample records to find
        representative keys that will be used to partition the full dataset evenly.
        
        Args:
            rdd: Input RDD of (key, value) pairs
            sample_size: Number of records to sample (should be small for efficiency)
        
        Returns:
            List of partition boundaries (keys)
        
        ALGORITHM:
        1. Sample random records from the full dataset
        2. Extract and sort the sampled keys
        3. Select evenly-spaced keys as partition boundaries
        4. This ensures each partition gets roughly equal data volume
        
        WHY THIS WORKS:
        - With uniform random keys, a sorted sample gives us uniform partition boundaries
        - Sampling is much faster than sorting entire dataset first
        - Reduces data movement and shuffle time
        """
        
        # Get total count for sampling calculation
        total_count = rdd.count()

        # Ensure sample_size doesn't exceed total count
        sample_size = min(sample_size, total_count)

        # Calculate sampling fraction
        sample_fraction = min(1.0, (sample_size * 1.5) / total_count)  # 1.5x to ensure we get enough

        # Take a random sample from the RDD
        sampled_keys = (rdd
                       .map(lambda x: x[0])  # Extract only keys
                       .sample(False, sample_fraction, seed=42)
                       .collect())
        
        # If we still don't have enough samples, try increasing fraction
        if len(sampled_keys) < sample_size:
            sampled_keys = (rdd
                           .map(lambda x: x[0])
                           .sample(False, min(1.0, (sample_size * 2) / total_count), seed=42)
                           .collect())

        # Sort sampled keys
        sampled_keys.sort()
        
        # Remove duplicates to ensure unique boundaries
        sampled_keys = list(dict.fromkeys(sampled_keys))  # Preserves order, removes duplicates

        # Select evenly-spaced keys as partition boundaries
        # For N partitions, we need N-1 boundaries
        num_partitions = self.sc.defaultParallelism

        # If we have fewer unique keys than partitions, just use all of them
        if len(sampled_keys) <= num_partitions:
            boundaries = sampled_keys[:-1] if len(sampled_keys) > 1 else []
        else:
            # Select N-1 evenly spaced keys
            step = max(1, len(sampled_keys) // num_partitions)
            boundaries = [sampled_keys[i * step] for i in range(1, num_partitions) if i * step < len(sampled_keys)]

        return boundaries
    
    @staticmethod
    def partition_key(key: str, boundaries: List[str]) -> int:
        """
        Determine which partition a key belongs to using binary search.
        
        Given a set of sorted boundaries, find which partition this key
        should be assigned to using binary search (efficient).
        
        Args:
            key: The key to partition
            boundaries: Sorted list of partition boundaries
        
        Returns:
            Partition ID (integer between 0 and len(boundaries))
        
        EXPLANATION:
        - Binary search finds the correct partition in O(log N) time
        - If key < first boundary, partition = 0
        - If key >= last boundary, partition = N
        - Otherwise, partition = index of first boundary >= key
        """
        
        # Binary search to find partition
        left, right = 0, len(boundaries)
        while left < right:
            mid = (left + right) // 2
            if boundaries[mid] <= key:
                left = mid + 1
            else:
                right = mid
        return left
    
    def terasort(self, rdd: RDD, sample_size: int = 1000) -> RDD:
        """
        FULL TERASORT IMPLEMENTATION
        ===========================
        
        Main TeraSort algorithm combining all phases.
        
        Args:
            rdd: Input RDD of (key, value) pairs
            sample_size: Number of samples for determining boundaries
        
        Returns:
            Sorted RDD (by key)
        
        PHASES:
        1. SAMPLING:   Determine partition boundaries from samples
        2. PARTITIONING: Assign records to partitions based on boundaries
        3. SORTING:    Sort data within each partition (Spark handles)
        
        MAPREDUCE PATTERN:
        
        MAP PHASE:
        ----------
        Input:  (key, value) pairs
        Processing: (partition_id, (key, value))
        Output: Records tagged with partition ID
        
        SHUFFLE & SORT PHASE (Automatic in Spark):
        -------------------------------------------
        - Groups records by partition ID
        - Sorts records within each partition by key
        - Distributes (i.e., "shuffles") data to appropriate executors
        
        REDUCE PHASE:
        -------
        Input:  Sorted records grouped by partition
        Processing: Output sorted (key, value) pairs
        Output: Fully sorted dataset
        """
        
        print("=" * 70)
        print("TERASORT ALGORITHM - EXECUTION")
        print("=" * 70)
        
        # PHASE 1: SAMPLING
        print("\n[PHASE 1/3] Sampling for partition boundaries...")
        boundaries = self.sample_for_partition_boundaries(rdd, sample_size)
        print(f"  ✓ Found {len(boundaries)} partition boundaries")
        print(f"  Sample boundaries (first 5): {boundaries[:5]}")
        
        # PHASE 2: PARTITIONING (MAP PHASE in MapReduce)
        print("\n[PHASE 2/3] Partitioning data using boundaries...")
        
        # Broadcast boundaries to all executors for efficient access
        broadcasted_boundaries = self.sc.broadcast(boundaries)
        
        # Create a static function for mapping that doesn't capture self
        def map_to_partition(record: Tuple[str, str]) -> Tuple[int, Tuple[str, str]]:
            """
            MAP FUNCTION: Determine partition for each record.
            
            Input:  (key, value) record
            Output: (partition_id, (key, value))
            
            This maps the input record to a partition ID based on the key.
            The partition ID determines which executor will handle this record.
            """
            key, value = record
            # Use TeraSort.partition_key static method instead of self.partition_key
            partition_id = TeraSort.partition_key(key, broadcasted_boundaries.value)
            return (partition_id, (key, value))
        
        # Apply mapping: tag each record with its partition ID
        partitioned_rdd = rdd.map(map_to_partition)
        
        # PHASE 3: SORTING (SHUFFLE + REDUCE PHASE in MapReduce)
        print("[PHASE 3/3] Shuffling and sorting...")
        
        # Use sortByKey to:
        # 1. SHUFFLE: Distribute records to executors based on partition_id
        # 2. SORT: Sort records within each partition by (partition_id, key)
        # This is the "shuffle and sort" phase of MapReduce
        
        def reduce_to_sorted(partition_rdd):
            """
            REDUCE FUNCTION: Sort records within this partition.
            
            Input:  RDD of this partition's records
            Output: Sorted (key, value) pairs
            
            We extract the actual key-value pairs and sort by key.
            """
            # Extract (key, value) from (partition_id, (key, value))
            records = list(partition_rdd)
            # Sort by key
            records.sort(key=lambda x: x[1][0])
            # Return sorted records without partition ID
            return [(k, v) for _, (k, v) in records]
        
        # sortByKey sorts by the first element (partition_id, then key)
        sorted_rdd = (partitioned_rdd
                     .sortByKey()
                     .mapPartitions(reduce_to_sorted))
        
        print("  ✓ Data shuffled and sorted")
        print("=" * 70)
        
        return sorted_rdd
    
    def verify_sort(self, rdd: RDD) -> bool:
        """
        Verify that the RDD is correctly sorted (for testing).
        
        Args:
            rdd: RDD to verify
        
        Returns:
            True if sorted, False otherwise
        
        WARNING: This collects data to the driver - only use on small datasets!
        """
        
        collected = rdd.take(1000)  # Check first 1000 records
        keys = [k for k, v in collected]
        
        is_sorted = all(keys[i] <= keys[i+1] for i in range(len(keys)-1))
        return is_sorted


class MapReduceExplanation:
    """
    Educational class explaining MapReduce concepts and how TeraSort uses them.
    """
    
    @staticmethod
    def explain_mapreduce():
        """Print detailed explanation of MapReduce."""
        explanation = """
        MAPREDUCE PROGRAMMING MODEL
        ============================
        
        MapReduce is a distributed computing framework that processes large datasets
        by breaking them into smaller pieces and processing them in parallel.
        
        Three Main Phases:
        -------------------
        
        1. MAP PHASE
           -----------
           - Function: Input → (Key, Value) pairs
           - Purpose: Transform raw input into structured key-value data
           - Parallelism: Runs simultaneously on different data chunks
           - Example (in TeraSort):
             Input:  Raw hex strings
             Output: (partition_id, (hex_key, hex_value))
           
           Process:
           ┌─────────────────────────────────┐
           │  Raw Input Data                 │
           └─────────────────────────────────┘
                      ↓ (split into chunks)
           ┌─────────┬─────────┬─────────┐
           │ Chunk 1 │ Chunk 2 │ Chunk 3 │
           └────┬────┴────┬────┴────┬────┘
                ↓         ↓         ↓
           ┌─────────┬─────────┬─────────┐
           │ Map 1   │ Map 2   │ Map 3   │ (runs in parallel)
           └─────────┴─────────┴─────────┘
                ↓         ↓         ↓
           Key-Value Pairs Produced
        
        2. SHUFFLE & SORT PHASE (Implicit in frameworks like Spark/Hadoop)
           ───────────────────────────────────────────────────────────────
           - Purpose: Group all values with the same key together
           - Sort: Keys are sorted; values for each key are usually in order
           - Network Intensive: Data moves across the network to group by key
           - Framework Automatic: Developers don't implement this; the framework does
           
           Process:
           (Key1, V1), (Key3, V3), (Key1, V2), (Key2, V4), (Key3, V5)
                        ↓ (shuffle & sort)
           (Key1, [V1, V2]), (Key2, [V4]), (Key3, [V3, V5])
        
        3. REDUCE PHASE
           ──────────────
           - Function: (Key, [Values]) → Output
           - Purpose: Aggregate values with the same key
           - Parallelism: Independent reduce operations can run in parallel
           - Example (in TeraSort):
             Input:  (Key, [(Value1, Value2, ...)])
             Output: Sorted key-value pairs
        
        HOW TERASORT USES MAPREDUCE:
        ===========================
        
        Problem: Sort a terabyte of data across 1000 machines
        Solution: Divide into 1000 regions, sort in parallel
        
        Step 1 - MAP: Tag each record with target partition
        ────────────────────────────────────────────────
        Input:  (key="ABC123", value="VERYPRECIOUSDATA")
        Map:    (partition=5, (key="ABC123", value="VERYPRECIOUSDATA"))
                [Partition determined by sampling-based boundaries]
        
        Step 2 - SHUFFLE & SORT: Group and shuffle to target partition
        ──────────────────────────────────────────────────────────────
        - All records with partition=5 are sent to the same executor
        - Records arrive at their target partition
        - Spark automatically sorts within each partition
        
        Step 3 - REDUCE: Finalize output
        ────────────────────────────────
        Output: Records now sorted within partitions, and partition IDs
                ensure global sorted order
        
        Benefits of MapReduce for TeraSort:
        ──────────────────────────────────
        1. Scalability: Can sort petabytes with 10000+ machines
        2. Fault Tolerance: If a machine fails, its work is redone
        3. Locality: Data is processed where it's stored
        4. Load Balancing: Equal partitions mean equal work per machine
        """
        print(explanation)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create Spark Session
    spark = SparkSession.builder \
        .appName("TeraSort") \
        .master("local[4]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*70)
    print("TERASORT WITH MAPREDUCE - PYSPARK IMPLEMENTATION")
    print("="*70)
    
    # Print MapReduce explanation
    print("\n")
    MapReduceExplanation.explain_mapreduce()
    
    # Initialize TeraSort
    terasort = TeraSort(spark)
    
    # Generate sample data (10MB for demo; in production this would be TB)
    print("\n[SETUP] Generating sample data (100,000 records ≈ 10MB)...")
    num_records = 1000
    input_rdd = terasort.generate_terabyte_data(num_records)
    print(f"✓ Generated {num_records} records")
    
    # Run TeraSort
    print("\n")
    sorted_rdd = terasort.terasort(input_rdd, sample_size=100)
    
    # Verify sorting (on small sample)
    print("\n[VERIFICATION] Checking sort correctness...")
    is_sorted = terasort.verify_sort(sorted_rdd)
    print(f"✓ Data is sorted: {is_sorted}")
    
    # Show some results
    print("\n[RESULTS] First 10 sorted records:")
    print("-" * 70)
    for i, (key, value) in enumerate(sorted_rdd.take(10)):
        print(f"{i+1:2d}. Key: {key} | Value: {value[:20]}...")
    
    print("\n[RESULTS] Last 10 sorted records:")
    print("-" * 70)
    # Get the last 10 records by taking the top 10 in reverse order
    for i, (key, value) in enumerate(sorted_rdd.sortByKey(ascending=False).take(10), 1):
        print(f"{i:2d}. Key: {key} | Value: {value[:20]}...")
    
    print("\n" + "="*70)
    print("TERASORT EXECUTION COMPLETE")
    print("="*70 + "\n")
    
    spark.stop()

