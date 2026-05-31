# Nano Vector DB (Educational)

This is a personal study project created to understand how Vector Databases work "under the hood". When developing applications based on LLMs and RAG (Retrieval-Augmented Generation), we often take vector search infrastructure (like Pinecone or Milvus) for granted.

I wrote this minimal Vector DB in C++ from scratch to deconstruct and concretely explore these concepts, and then bridged it to Python to run a real Artificial Intelligence RAG pipeline.

## What's Inside

The codebase is heavily commented for educational purposes and covers the entire lifecycle of an AI search engine:

1. **Cosine Similarity & SIMD**: The mathematical calculation of semantic proximity between two vectors is implemented using Cosine Similarity. To make it extremely fast, the function leverages native SIMD hardware instructions for Apple Silicon (`<arm_neon.h>`).
2. **Linear Search (Brute Force)**: The baseline `O(N)` approach. It guarantees a 100% exact match but scales poorly on large datasets.
3. **Hierarchical Navigable Small World (HNSW Concepts)**: The true core of modern Vector DBs. During insertion, the algorithm uses the graph itself to find the best neighbors in `O(log N)` time. The search algorithm (`search_graph`) traverses the graph hopping from node to node (Greedy Walk), ignoring 99% of irrelevant nodes and completing queries in microseconds.
4. **Python Bindings (`pybind11`)**: The C++ engine is compiled into a shared library (`.so`) that can be natively imported into Python. This replicates the exact architecture of production ML libraries like PyTorch or TensorFlow (Python frontend, C++ backend).
5. **Real Embeddings (Semantic Search)**: A Python script uses a local AI model (`sentence-transformers/all-MiniLM-L6-v2`) to convert Italian sentences into 384-dimensional vectors, store them in the C++ DB, and answer natural language queries based on mathematical *meaning* rather than exact keywords.

## Benchmarks (C++ Core)

I stress-tested the database by generating 10,000 vectors with 128 dimensions. The graph construction is fully optimized (`O(N log N)`).

```text
--- TEST 1: LINEAR BRUTE-FORCE SEARCH (ALL NODES) ---
Time elapsed: ~400 microseconds.

--- TEST 2: APPROXIMATE GRAPH SEARCH (GREEDY WALK) ---
Time elapsed: ~1-7 microseconds.

=== RESULT ===
Graph Search is on average 50x to 400x faster than full linear scanning.
```

## How to Run

### 1. C++ Engine Benchmark
Compile with optimizations enabled:
```bash
g++ -O3 -std=c++17 main.cpp nano_vdb.cpp -o nano_vdb_test
./nano_vdb_test
```

### 2. Python AI & Semantic Search
Set up a Virtual Environment, install the dependencies, compile the C++ binding, and run the real embedding RAG test:
```bash
# 1. Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyBind11 and the AI Model
pip install pybind11 sentence-transformers

# 3. Compile the C++ Shared Library for Python
c++ -O3 -Wall -shared -std=c++17 -fPIC -undefined dynamic_lookup $(python -m pybind11 --includes) nano_vdb.cpp binding.cpp -o nano_vdb$(python-config --extension-suffix)

# 4. Run the Semantic Search script
python real_embeddings.py
```
