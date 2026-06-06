# Nano Vector DB (Educational & Interactive)

This is a personal study project created to understand how Vector Databases work "under the hood". When developing applications based on LLMs and RAG (Retrieval-Augmented Generation), we often take vector search infrastructure (like Pinecone or Milvus) for granted.

This repository implements a minimal Vector DB in C++ from scratch featuring **Scalar Quantization (SQ8)** and **hardware SIMD acceleration**, bridges it to Python, and wraps it in a **Go + Python microservices-based interactive Web interface**.

---

## Repository Structure

```text
nano-vdb/
├── core/                       # The C++ core database engine
│   ├── nano_vdb.cpp            # Database implementation & similarity logic
│   ├── nano_vdb.hpp            # Database headers & structures
│   ├── binding.cpp             # PyBind11 bindings mapping C++ to Python
│   └── nano_vdb.pyi            # Python type hints for autocomplete
├── playground/                 # CLI experiments, tests and benchmarks
│   ├── main.cpp                # Pure C++ command-line benchmark
│   ├── test_binding.py         # Testing the pybind11 native library
│   ├── real_embeddings.py      # Local HuggingFace sentence embedding test
│   └── rag_ollama.py           # Command-line RAG prototype using Ollama
├── static/                     # Premium Dark-Mode Web Frontend
│   ├── index.html              
│   ├── style.css               
│   └── app.js                  
├── server.py                   # Python FastAPI microservice (ML & VectorDB backend)
├── main.go                     # Go API Gateway & Web Server (Frontend host)
├── go.mod                      # Go module config
└── README.md                   
```

---

## Features

1. **Per-Vector Scalar Quantization (SQ8)**: Compressed original 32-bit floats to 8-bit integers (`uint8_t`), reducing the vector memory footprint by ~74% while keeping precision loss under 1.5%.
2. **Apple Silicon Neon (SIMD) Acceleration**: Cosine similarity is computed directly on CPU registers using hardware instructions (`<arm_neon.h>`), processing 4 elements in a single clock cycle.
3. **Optimized Cosine Similarity Math**: Leverages the algebraic equivalence:
   $$\sum Q_i D_i \approx D_{min} \sum Q_i + scale \sum Q_i q_i$$
   to precompute query statistics and perform only additions and multiplications on bytes inside the main loop.
4. **Hierarchical Navigable Small World (HNSW Concepts)**: Traverses nodes on a Small World graph (Greedy Walk) in $O(\log N)$ average complexity instead of $O(N)$ linear scans.
5. **Go & Python Microservices Architecture**: 
   - A **Go** web server acts as the main entry point, serving a premium Glassmorphism frontend and proxying API calls.
   - A **Python FastAPI** service runs in background handling embedding generation via HuggingFace models, querying the native C++ DB, and communicating with local Ollama instances.

---

## How to Run

### 1. Compile Python Bindings (C++ shared library)
Make sure you compile from the root directory pointing to the `core/` folder:
```bash
# 1. Setup and activate Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install pybind11 sentence-transformers fastapi uvicorn

# 3. Compile the C++ shared library for Python
c++ -O3 -Wall -shared -std=c++17 -fPIC -undefined dynamic_lookup $(python -m pybind11 --includes) core/nano_vdb.cpp core/binding.cpp -o nano_vdb$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
```

### 2. Start the Microservices

#### Backend: Python FastAPI ML Service
Starts on port `8000`:
```bash
venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

#### Gateway & Frontend: Go Web Server
Starts on port `8080`:
```bash
go run main.go
```

Now open your browser and navigate to: 👉 **[http://localhost:8080](http://localhost:8080)**

---

## CLI Playground Sandbox (Optional)

If you wish to run the original C++ benchmark:
```bash
g++ -O3 -std=c++17 playground/main.cpp core/nano_vdb.cpp -o playground/nano_vdb_test
./playground/nano_vdb_test
```
