#ifndef LLAMA_INFERENCE_HPP
#define LLAMA_INFERENCE_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

// Configurazione dei parametri architetturali del modello
struct Config {
    int dim;        // dimensione di embedding
    int hidden_dim; // dimensione nascosta nello strato Feed-Forward
    int n_layers;   // numero di strati (Transformer blocks)
    int n_heads;    // numero di heads nell'attenzione
    int n_kv_heads; // numero di heads per Key/Value (solitamente uguale a n_heads)
    int vocab_size; // dimensione del vocabolario (es. 32000)
    int seq_len;    // lunghezza massima della sequenza di token
};

// PUNTATORI AI PESI DEL MODELLO
// Mappiamo direttamente i pesi dal file binario senza copiarli, per performance estreme.
struct TransformerWeights {
    float* token_embedding_table;    // (vocab_size, dim)
    
    // Pesi per ogni layer del Transformer
    float* rms_att_weight;           // (n_layers, dim)
    float* rms_ffn_weight;           // (n_layers, dim)
    
    // Matrici dei pesi dell'attenzione Q, K, V e Output Projection
    float* wq;                       // (n_layers, dim, dim)
    float* wk;                       // (n_layers, dim, dim)
    float* wv;                       // (n_layers, dim, dim)
    float* wo;                       // (n_layers, dim, dim)
    
    // Matrici dei pesi del Feed-Forward Network (SwiGLU)
    float* w1;                       // (n_layers, hidden_dim, dim)
    float* w2;                       // (n_layers, dim, hidden_dim)
    float* w3;                       // (n_layers, hidden_dim, dim)
    
    // Strato di normalizzazione e classificazione finale
    float* rms_final_weight;         // (dim,)
    float* wcls;                     // (vocab_size, dim) - Spesso punta alla token_embedding_table (weight sharing)
};

// BUFFER DI MEMORIA TEMPORANEI (RUN STATE)
// Contiene la memoria di lavoro allocata una sola volta per i calcoli del forward pass
struct RunState {
    std::vector<float> x;      // stato nascosto corrente (dim)
    std::vector<float> xb;     // buffer ausiliario per la moltiplicazione di matrici (dim)
    std::vector<float> xb2;    // secondo buffer ausiliario (dim)
    std::vector<float> hb;     // buffer per lo stato nascosto FFN (hidden_dim)
    
    // Query, Key, Value
    std::vector<float> q;      // query (dim)
    std::vector<float> k;      // key (dim)
    std::vector<float> v;      // value (dim)
    std::vector<float> att;    // pesi di attenzione (n_heads, seq_len)
    std::vector<float> logits; // output logit (vocab_size)
    
    // Key/Value Cache (necessaria per non ricalcolare le chiavi/valori dei token passati)
    std::vector<float> key_cache;   // (n_layers, seq_len, dim)
    std::vector<float> value_cache; // (n_layers, seq_len, dim)
};

// TOKENIZER DI LLAMA2
// Legge il file tokenizer.bin e mappa i token ID in parole e viceversa.
class Tokenizer {
private:
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    std::unordered_map<std::string, int> sorted_vocab; // Mappa per ricerca O(1) velocissima
    int vocab_size;
    unsigned int max_token_length;

    int str_lookup(const std::string& str) const;

public:
    Tokenizer();
    bool load(const std::string& path);
    std::string decode(int prev_token, int token) const;
    std::vector<int> encode(const std::string& text) const;
    int get_vocab_size() const { return vocab_size; }
    
    // Token speciali
    int bos_token_id = 1; // Beginning of Sentence
    int eos_token_id = 2; // End of Sentence
};

// MOTORE DI INFERENZA PRINCIPALE
class LlamaEngine {
private:
    Config config;
    TransformerWeights weights;
    RunState state;
    Tokenizer tokenizer;
    
    // Puntatori per la gestione della memoria del file mappato (mmap)
    void* data;
    size_t file_size;
    bool model_loaded;

    // Funzione interna per eseguire il forward pass del modello sul token corrente
    float* forward(int token, int pos);
    
    // Selezione del prossimo token in base alla probabilità (logits)
    int sample(float* logits, float temperature);

public:
    LlamaEngine(const std::string& model_path, const std::string& tokenizer_path);
    ~LlamaEngine();
    
    // Genera testo a partire da un prompt
    std::string generate(const std::string& prompt, int max_tokens, float temperature = 0.9f);
    
    bool is_ready() const { return model_loaded; }
};

#endif // LLAMA_INFERENCE_HPP
