# Nano Vector DB (Educational)

This is a personal study project created to understand how Vector Databases work "under the hood". When developing applications based on LLMs and RAG (Retrieval-Augmented Generation), we often take vector search infrastructure (like Pinecone or Milvus) for granted.

I wrote this minimal Vector DB in C++ from scratch to deconstruct and concretely explore these concepts.

## What's Inside

The codebase is heavily commented for educational purposes and covers three core engineering concepts:

1. **Cosine Similarity & SIMD**: The mathematical calculation of semantic proximity between two vectors is implemented using Cosine Similarity. To make it extremely fast, the function leverages native SIMD hardware instructions for Apple Silicon (`<arm_neon.h>`), allowing the CPU to Multiply-Accumulate blocks of 4 floats in parallel within a single clock cycle.
2. **Linear Search (Brute Force)**: The baseline `O(N)` approach. It compares the query against *all* records in memory. It guarantees a 100% exact match but scales poorly on large datasets.
3. **Approximate Nearest Neighbors (ANN) via Graph**: The true core of modern Vector DBs. During insertion, vectors are connected to their most similar "neighbors", creating a "Navigable Small World". The search algorithm (`search_graph`) traverses the graph hopping from node to node (Greedy Walk), getting closer to the query while ignoring 99% of irrelevant nodes, bringing the search time complexity down to `O(log N)`.

## Benchmarks

I stress-tested the database by generating 10,000 vectors with 128 dimensions (a typical embedding structure). The logs speak for themselves:

```text
--- TEST 1: LINEAR BRUTE-FORCE SEARCH (ALL NODES) ---
Time elapsed: ~260 microseconds.

--- TEST 2: APPROXIMATE GRAPH SEARCH (GREEDY WALK) ---
Time elapsed: ~1-5 microseconds.

=== RESULT ===
Graph Search is on average 50x to 250x faster than full linear scanning.
```

*(Because graph search is "approximate", results may rarely differ slightly from exact search. This is the standard trade-off accepted in AI engineering to achieve extreme performance).*

## Compilation and Execution

A standard C++ compiler is required. It is highly recommended to use the `-O3` flag to allow the compiler to maximize mathematical instruction optimizations.

```bash
g++ -O3 -std=c++17 main.cpp nano_vdb.cpp -o nano_vdb_test
./nano_vdb_test
```
