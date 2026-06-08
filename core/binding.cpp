#include "nano_vdb.hpp"
#include "llama_inference.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Essenziale: traduce automaticamente std::vector e std::string tra C++ e Python!

namespace py = pybind11;

// Questa macro crea il modulo Python "nano_vdb"
PYBIND11_MODULE(nano_vdb, m) {
  m.doc() = "NanoVectorDB Python Bindings"; // Docstring del modulo

  // Esportiamo la struct SearchResult in modo che Python possa leggerla
  py::class_<SearchResult>(m, "SearchResult")
      .def_readonly("id", &SearchResult::id)
      .def_readonly("score", &SearchResult::score)
      .def_readonly("index", &SearchResult::index)
      // Aggiungiamo il metodo __repr__ per stampare i risultati comodamente in
      // Python
      .def("__repr__", [](const SearchResult &r) {
        return "<SearchResult id='" + r.id +
               "' score=" + std::to_string(r.score) + ">";
      });

  // Esportiamo la classe principale NanoVectorDB
  py::class_<NanoVectorDB>(m, "NanoVectorDB")
      // Esportiamo il costruttore (che richiede le dimensioni)
      .def(py::init<size_t>(), py::arg("dims"))

      // Esportiamo le funzioni di inserimento e ricerca
      .def("insert", &NanoVectorDB::insert, py::arg("id"), py::arg("vec"))
      .def("search_graph", &NanoVectorDB::search_graph, py::arg("query_vec"),
           py::arg("top_k"))
      .def("search_linear", &NanoVectorDB::search_linear, py::arg("query_vec"),
           py::arg("top_k"))

      // Esportiamo persistenza e utility
      .def("save_to_file", &NanoVectorDB::save_to_file, py::arg("filename"))
      .def("load_from_file", &NanoVectorDB::load_from_file, py::arg("filename"))
      .def("size", &NanoVectorDB::size);

  // Esportiamo la classe LlamaEngine per l'inferenza nativa dei Transformer LLM
  py::class_<LlamaEngine>(m, "LlamaEngine")
      .def(py::init<const std::string&, const std::string&>(), py::arg("model_path"), py::arg("tokenizer_path"))
      .def("generate", &LlamaEngine::generate, py::arg("prompt"), py::arg("max_tokens"), py::arg("temperature") = 0.9f)
      .def("is_ready", &LlamaEngine::is_ready);
}
