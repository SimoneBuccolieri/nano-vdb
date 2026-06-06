import nano_vdb # Stiamo importando la nostra libreria C++ nativa!
import random
import time

DIMS = 128
# Creiamo un'istanza del DB (esattamente come facevamo in C++)
db = nano_vdb.NanoVectorDB(DIMS)

print("Inizio inserimento di 10.000 vettori da Python...")
start = time.time()

# Generiamo vettori casuali da Python e li spariamo nel motore C++
for i in range(10000):
    vec = [random.uniform(-1.0, 1.0) for _ in range(DIMS)]
    # pybind11 converte magicamente la lista Python in std::vector<float>!
    db.insert(f"py_node_{i}", vec)

print(f"Inserimento completato in {time.time() - start:.2f} secondi.")

# Creiamo una query di test
query = [random.uniform(-1.0, 1.0) for _ in range(DIMS)]

print("\n--- TEST: GRAPH SEARCH DA PYTHON ---")
start = time.perf_counter()
results = db.search_graph(query, 3)
end = time.perf_counter()

for r in results:
    print(r) # Questo userà il __repr__ che abbiamo definito in binding.cpp

print(f"Tempo impiegato dalla ricerca: {(end - start) * 1e6:.0f} microsecondi")
