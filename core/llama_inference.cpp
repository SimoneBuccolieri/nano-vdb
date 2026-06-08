#include "llama_inference.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// --- OPERAZIONI MATEMATICHE DI BASE ---

// Normalizzazione RMSNorm: normalizza il vettore 'x' usando la radice media dei quadrati
// e lo scala usando il peso 'weight'. Garantisce stabilità numerica nei Transformer.
static void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; ++i) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss += 1e-5f; // epsilon per evitare divisione per zero
    ss = 1.0f / std::sqrt(ss);
    
    for (int i = 0; i < size; ++i) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

// Softmax: trasforma i logits grezzi in una distribuzione di probabilità che somma a 1.
static void softmax(float* x, int size) {
    // Troviamo il valore massimo per stabilità numerica (evita overflow esponenziale)
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

// Moltiplicazione Matrice-Vettore (MatMul): calcola o = W * x
// o ha dimensione 'xout', W ha dimensione (xout, xin), x ha dimensione 'xin'.
static void matmul(float* o, float* x, float* w, int xin, int xout) {
    // Nota: in un motore reale questo verrebbe parallelizzato con OpenMP o SIMD,
    // ma la versione lineare è ottima per comprendere il funzionamento sequenziale.
    for (int i = 0; i < xout; ++i) {
        float val = 0.0f;
        for (int j = 0; j < xin; ++j) {
            val += w[i * xin + j] * x[j];
        }
        o[i] = val;
    }
}

// --- TOKENIZER IMPLEMENTAZIONE ---

Tokenizer::Tokenizer() : vocab_size(0), max_token_length(0) {}

// Carica il vocabolario binario generato da llama2.c
bool Tokenizer::load(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::cerr << "Errore: Impossibile aprire " << path << std::endl;
        return false;
    }
    
    // Legge la lunghezza massima di un token
    if (std::fread(&max_token_length, sizeof(unsigned int), 1, file) != 1) {
        std::fclose(file);
        return false;
    }
    
    // Leggiamo i token uno alla volta
    float score;
    int len;
    vocab.clear();
    vocab_scores.clear();
    sorted_vocab.clear();
    
    while (std::fread(&score, sizeof(float), 1, file) == 1) {
        if (std::fread(&len, sizeof(int), 1, file) != 1) break;
        
        std::string token(len, '\0');
        if (std::fread(&token[0], sizeof(char), len, file) != (size_t)len) break;
        
        vocab.push_back(token);
        vocab_scores.push_back(score);
        // Popoliamo la mappa per la ricerca veloce O(1)
        sorted_vocab[token] = vocab.size() - 1;
    }
    
    vocab_size = vocab.size();
    std::fclose(file);
    return vocab_size > 0;
}

// Cerca una stringa nel vocabolario e restituisce il suo ID (o -1)
int Tokenizer::str_lookup(const std::string& str) const {
    auto it = sorted_vocab.find(str);
    if (it != sorted_vocab.end()) {
        return it->second;
    }
    return -1;
}

// Converte un ID di token in stringa leggibile (gestendo spazi speciali di SentencePiece)
std::string Tokenizer::decode(int prev_token, int token) const {
    if (token < 0 || token >= vocab_size) return "";
    
    std::string s = vocab[token];
    
    // Gestione del carattere speciale byte fallback o spazi di SentencePiece
    if (prev_token == bos_token_id && s[0] == ' ') {
        s = s.substr(1);
    }
    
    // Sostituiamo i caratteri di escape esadecimali <0xXX> se presenti
    if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
        char val = (char)std::strtol(s.substr(3, 2).c_str(), nullptr, 16);
        return std::string(1, val);
    }
    
    // Sostituiamo il carattere speciale di spazio con uno spazio reale
    std::string result = "";
    for (size_t i = 0; i < s.length(); ++i) {
        if ((unsigned char)s[i] == 0xe2 && i + 2 < s.length() && 
            (unsigned char)s[i+1] == 0x96 && (unsigned char)s[i+2] == 0x81) {
            result += " ";
            i += 2;
        } else {
            result += s[i];
        }
    }
    
    return result;
}

// Codifica del testo in token ID usando l'algoritmo Byte Pair Encoding (BPE)
std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    
    // 1. Aggiungiamo BOS (inizio sequenza)
    tokens.push_back(bos_token_id);
    
    if (text.empty()) return tokens;
    
    // 2. Aggiungiamo lo spazio dummy iniziale se previsto ( SentencePiece default )
    int dummy_prefix = str_lookup(" ");
    if (dummy_prefix != -1) {
        tokens.push_back(dummy_prefix);
    }
    
    // 3. Elaboriamo i caratteri UTF-8 / byte grezzi
    std::string str_buffer;
    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        
        // Se non è un byte di continuazione UTF-8, iniziamo un nuovo codepoint
        if ((c & 0xC0) != 0x80) {
            str_buffer.clear();
        }
        
        str_buffer += c;
        
        // Se il prossimo byte è di continuazione UTF-8, accumuliamo nel buffer
        if (i + 1 < text.length() && (text[i+1] & 0xC0) == 0x80 && str_buffer.length() < 4) {
            continue;
        }
        
        // Cerchiamo il codepoint nel vocabolario
        int id = str_lookup(str_buffer);
        if (id != -1) {
            tokens.push_back(id);
        } else {
            // Byte fallback: codifichiamo ogni byte individualmente come `byte + 3`
            for (char bc : str_buffer) {
                unsigned char b = static_cast<unsigned char>(bc);
                tokens.push_back(b + 3);
            }
        }
        str_buffer.clear();
    }
    
    // 4. Ciclo di merging BPE iterativo
    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;
        
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            // Uniamo i due token consecutivi in una stringa per verificare se esiste un merge
            std::string concat_str = vocab[tokens[i]] + vocab[tokens[i+1]];
            int id = str_lookup(concat_str);
            if (id != -1 && vocab_scores[id] > best_score) {
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        
        if (best_idx == -1) {
            break; // Nessun altro merge possibile in base al vocabolario
        }
        
        // Eseguiamo il merge unendo la coppia (best_idx, best_idx + 1) in best_id
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1); // Rimuoviamo il secondo elemento unito
    }
    
    return tokens;
}

// --- ENGINE DI INFERENZA IMPLEMENTAZIONE ---

LlamaEngine::LlamaEngine(const std::string& model_path, const std::string& tokenizer_path)
    : data(MAP_FAILED), file_size(0), model_loaded(false) {
    
    // 1. Carica il Tokenizer
    if (!tokenizer.load(tokenizer_path)) {
        std::cerr << "Errore: Caricamento Tokenizer fallito!" << std::endl;
        return;
    }
    
    // 2. Carica i pesi del Modello via mmap (mappatura memoria su file)
    int fd = open(model_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "Errore: Impossibile aprire il file del modello " << model_path << std::endl;
        return;
    }
    
    struct stat statbuf;
    if (fstat(fd, &statbuf) < 0) {
        close(fd);
        return;
    }
    file_size = statbuf.st_size;
    
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (data == MAP_FAILED) {
        std::cerr << "Errore: mmap fallito!" << std::endl;
        return;
    }
    
    // 3. Leggiamo l'header dei parametri di configurazione
    std::memcpy(&config, data, sizeof(Config));
    
    // Se vocab_size è positivo, il classificatore finale (wcls) condivide i pesi con
    // la token_embedding_table per ridurre lo spazio (tied weights).
    // Se è negativo, i pesi del classificatore finale sono salvati separatamente.
    int shared_classifier = config.vocab_size > 0 ? 1 : 0;
    if (config.vocab_size < 0) {
        config.vocab_size = -config.vocab_size;
    }
    
    // Verifichiamo la bontà del file leggendo la dimensione
    std::cout << "[LlamaEngine] Modello caricato con successo!" << std::endl;
    std::cout << " - Layers: " << config.n_layers << std::endl;
    std::cout << " - Dimensioni: " << config.dim << std::endl;
    std::cout << " - Dimensioni Hidden FFN: " << config.hidden_dim << std::endl;
    std::cout << " - Vocab Size: " << config.vocab_size << std::endl;
    std::cout << " - Seq Len: " << config.seq_len << std::endl;
    std::cout << " - Shared Classifier: " << (shared_classifier ? "SI" : "NO") << std::endl;
    
    // 4. Mappiamo i puntatori dei pesi saltando l'header iniziale
    float* weights_ptr = (float*)((char*)data + sizeof(Config));
    
    weights.token_embedding_table = weights_ptr;
    weights_ptr += config.vocab_size * config.dim;
    
    weights.rms_att_weight = weights_ptr;
    weights_ptr += config.n_layers * config.dim;
    
    weights.wq = weights_ptr;
    weights_ptr += config.n_layers * config.dim * config.dim;
    
    weights.wk = weights_ptr;
    weights_ptr += config.n_layers * config.dim * config.dim;
    
    weights.wv = weights_ptr;
    weights_ptr += config.n_layers * config.dim * config.dim;
    
    weights.wo = weights_ptr;
    weights_ptr += config.n_layers * config.dim * config.dim;
    
    weights.rms_ffn_weight = weights_ptr;
    weights_ptr += config.n_layers * config.dim;
    
    weights.w1 = weights_ptr;
    weights_ptr += config.n_layers * config.hidden_dim * config.dim;
    
    weights.w2 = weights_ptr;
    weights_ptr += config.n_layers * config.dim * config.hidden_dim;
    
    weights.w3 = weights_ptr;
    weights_ptr += config.n_layers * config.hidden_dim * config.dim;
    
    weights.rms_final_weight = weights_ptr;
    weights_ptr += config.dim;
    
    // Mappatura condizionale di wcls
    if (shared_classifier) {
        weights.wcls = weights.token_embedding_table;
    } else {
        weights.wcls = weights_ptr;
        weights_ptr += config.vocab_size * config.dim;
    }
    
    // 5. Allocazione dei buffer temporanei del RunState
    state.x.resize(config.dim);
    state.xb.resize(config.dim);
    state.xb2.resize(config.dim);
    state.hb.resize(config.hidden_dim);
    
    state.q.resize(config.dim);
    state.k.resize(config.dim);
    state.v.resize(config.dim);
    state.att.resize(config.n_heads * config.seq_len);
    state.logits.resize(config.vocab_size);
    
    state.key_cache.resize(config.n_layers * config.seq_len * config.dim);
    state.value_cache.resize(config.n_layers * config.seq_len * config.dim);
    
    model_loaded = true;
}

LlamaEngine::~LlamaEngine() {
    if (data != MAP_FAILED) {
        munmap(data, file_size);
    }
}

// FORWARD PASS DEL TRANSFORMER
// Calcola le probabilità del token successivo basandosi sull'attenzione e la posizione corrente
float* LlamaEngine::forward(int token, int pos) {
    if (pos < 0 || pos >= config.seq_len) {
        return state.logits.data();
    }
    float* x = state.x.data();
    int dim = config.dim;
    int kv_dim = config.dim; // Llama2.c ha n_kv_heads == n_heads, quindi kv_dim == dim
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;
    
    // 1. Embedding lookup per il token corrente
    float* content_row = weights.token_embedding_table + token * dim;
    std::memcpy(x, content_row, dim * sizeof(float));
    
    // 2. Iteriamo attraverso tutti i blocchi del Transformer (layers)
    for (int l = 0; l < config.n_layers; ++l) {
        
        // 2a. Applichiamo RMSNorm prima dell'attenzione (Pre-Attention normalization)
        rmsnorm(state.xb.data(), x, weights.rms_att_weight + l * dim, dim);
        
        // 2b. Moltiplicazione delle matrici per ottenere Q, K, V (proiezioni dell'attenzione)
        matmul(state.q.data(), state.xb.data(), weights.wq + l * dim * dim, dim, dim);
        matmul(state.k.data(), state.xb.data(), weights.wk + l * dim * dim, dim, dim);
        matmul(state.v.data(), state.xb.data(), weights.wv + l * dim * dim, dim, dim);
        
        // 2c. Applicazione dei Rotary Positional Embeddings (RoPE) su Q e K
        // Questo inietta l'informazione di posizione dei token nel grafo dell'attenzione
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / std::pow(10000.0f, (float)head_dim / head_size);
            float val = pos * freq;
            float fcr = std::cos(val);
            float fsi = std::sin(val);
            
            // Q RoPE
            float q0 = state.q[i];
            float q1 = state.q[i+1];
            state.q[i]   = q0 * fcr - q1 * fsi;
            state.q[i+1] = q0 * fsi + q1 * fcr;
            
            // K RoPE
            float k0 = state.k[i];
            float k1 = state.k[i+1];
            state.k[i]   = k0 * fcr - k1 * fsi;
            state.k[i+1] = k0 * fsi + k1 * fcr;
        }
        
        // 2d. Salvataggio di Key e Value correnti nella Cache per il layer corrente
        int loff = l * config.seq_len * dim; // offset del layer corrente
        float* key_cache_row = state.key_cache.data() + loff + pos * dim;
        float* value_cache_row = state.value_cache.data() + loff + pos * dim;
        std::memcpy(key_cache_row, state.k.data(), dim * sizeof(float));
        std::memcpy(value_cache_row, state.v.data(), dim * sizeof(float));
        
        // 2e. Calcolo del Multi-Head Attention
        // Per ogni head dell'attenzione:
        for (int h = 0; h < config.n_heads; ++h) {
            float* q_head = state.q.data() + h * head_size;
            float* att_head = state.att.data() + h * config.seq_len;
            
            // Calcoliamo i pesi di attenzione facendo il prodotto scalare Q * K per tutte le posizioni passate
            for (int t = 0; t <= pos; ++t) {
                float* k_cache_t = state.key_cache.data() + loff + t * dim + h * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) {
                    score += q_head[i] * k_cache_t[i];
                }
                score /= std::sqrt(head_size); // Scaliamo il punteggio
                att_head[t] = score;
            }
            
            // Applichiamo la Softmax per ottenere pesi normalizzati in [0, 1] per la sequenza [0, pos]
            softmax(att_head, pos + 1);
            
            // Calcoliamo il vettore di output pesato combinando con i Value passati
            float* xb_head = state.xb.data() + h * head_size;
            std::memset(xb_head, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; ++t) {
                float* v_cache_t = state.value_cache.data() + loff + t * dim + h * head_size;
                float a = att_head[t];
                for (int i = 0; i < head_size; ++i) {
                    xb_head[i] += a * v_cache_t[i];
                }
            }
        }
        
        // Proiettiamo l'output dell'attenzione indietro: xb2 = wo * xb
        matmul(state.xb2.data(), state.xb.data(), weights.wo + l * dim * dim, dim, dim);
        
        // Residual connection: aggiungiamo l'output dell'attenzione allo stato nascosto originario
        for (int i = 0; i < dim; ++i) {
            x[i] += state.xb2[i];
        }
        
        // 2f. Feed-Forward Network (FFN) usando SwiGLU activation
        // RMSNorm pre-FFN
        rmsnorm(state.xb.data(), x, weights.rms_ffn_weight + l * dim, dim);
        
        // Proiettiamo nei due rami del gate SwiGLU: w1 (gating) e w3 (value)
        matmul(state.hb.data(), state.xb.data(), weights.w1 + l * hidden_dim * dim, dim, hidden_dim);
        matmul(state.xb2.data(), state.xb.data(), weights.w3 + l * hidden_dim * dim, dim, hidden_dim);
        
        // Applichiamo l'attivazione SiLU (o Swish): x * sigmoid(x)
        for (int i = 0; i < hidden_dim; ++i) {
            float val = state.hb[i];
            // silu(x) = x / (1 + exp(-x))
            state.hb[i] = val * (1.0f / (1.0f + std::exp(-val)));
        }
        
        // Element-wise multiply tra i due rami
        for (int i = 0; i < hidden_dim; ++i) {
            state.hb[i] *= state.xb2[i];
        }
        
        // Proiezione finale FFN e ritorno a dimensione 'dim'
        matmul(state.xb.data(), state.hb.data(), weights.w2 + l * dim * hidden_dim, hidden_dim, dim);
        
        // Residual connection
        for (int i = 0; i < dim; ++i) {
            x[i] += state.xb[i];
        }
    }
    
    // 3. Normalizzazione finale (Final RMSNorm)
    rmsnorm(x, x, weights.rms_final_weight, dim);
    
    // 4. Generiamo i logits proiettando sul vocabolario: logits = wcls * x
    matmul(state.logits.data(), x, weights.wcls, dim, config.vocab_size);
    
    return state.logits.data();
}

// CAMPIONAMENTO (SAMPLER)
// Sceglie il token successivo basandosi sulle probabilità generate
int LlamaEngine::sample(float* logits, float temperature) {
    if (temperature <= 0.0f) {
        // Campionamento deterministico (Argmax): seleziona il token con punteggio massimo
        int max_i = 0;
        float max_val = logits[0];
        for (int i = 1; i < config.vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_i = i;
            }
        }
        return max_i;
    } else {
        // Campionamento probabilistico
        // Dividiamo i logits per la temperatura
        for (int i = 0; i < config.vocab_size; ++i) {
            logits[i] /= temperature;
        }
        
        // Convertiamo in probabilità tramite softmax
        softmax(logits, config.vocab_size);
        
        // Estraiamo un valore casuale tra 0.0 e 1.0
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float r = dis(gen);
        
        // Ricerca cumulativa
        float cdf = 0.0f;
        for (int i = 0; i < config.vocab_size; ++i) {
            cdf += logits[i];
            if (r < cdf) {
                return i;
            }
        }
        return config.vocab_size - 1; // Fallback
    }
}

// GENERAZIONE DI TESTO A PARTIRE DA UN PROMPT
std::string LlamaEngine::generate(const std::string& prompt, int max_tokens, float temperature) {
    if (!model_loaded) return "[Errore: Modello non caricato]";
    
    // 1. Codifichiamo il prompt testuale in token ID
    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    int num_prompt_tokens = prompt_tokens.size();
    
    if (num_prompt_tokens <= 0) return "";
    
    // Preveniamo crash tagliando il prompt se supera la lunghezza massima della sequenza (seq_len = 256)
    // Lasciamo uno spazio minimo per la generazione di almeno 32 token (o meno se config.seq_len è piccolo)
    int min_generation_space = std::min(32, config.seq_len / 4);
    if (min_generation_space < 1) min_generation_space = 1;
    
    int max_allowed_prompt = config.seq_len - min_generation_space - 1;
    if (num_prompt_tokens > max_allowed_prompt) {
        std::vector<int> truncated;
        truncated.push_back(prompt_tokens[0]); // Mantieni BOS
        int start = num_prompt_tokens - max_allowed_prompt + 1;
        for (int i = start; i < num_prompt_tokens; ++i) {
            truncated.push_back(prompt_tokens[i]);
        }
        prompt_tokens = truncated;
        num_prompt_tokens = prompt_tokens.size();
    }
    
    std::string result_text = "";
    int token = prompt_tokens[0]; // Inizia con BOS (Beginning of Sentence)
    int next_token;
    
    // Preveniamo overflow se max_tokens supera la seq_len del modello
    int tokens_to_gen = std::min(max_tokens, config.seq_len - num_prompt_tokens - 1);
    if (tokens_to_gen <= 0) return "";
    
    // 2. Elaborazione dei token del prompt (Prefill phase)
    int pos = 0;
    for (pos = 0; pos < num_prompt_tokens - 1; ++pos) {
        forward(prompt_tokens[pos], pos);
    }
    
    // L'ultimo token del prompt definisce la base per il primo token generato
    token = prompt_tokens[num_prompt_tokens - 1];
    
    // 3. Loop di generazione (Autoregressive phase)
    for (int g = 0; g < tokens_to_gen; ++g) {
        // Calcola il forward pass per il token corrente alla posizione corrente
        float* logits = forward(token, pos);
        
        // Campiona il prossimo token
        next_token = sample(logits, temperature);
        
        // Se incontriamo la fine del testo (EOS), interrompiamo la generazione
        if (next_token == tokenizer.eos_token_id) {
            break;
        }
        
        // Decodifichiamo il token in testo e lo aggiungiamo al risultato
        std::string piece = tokenizer.decode(token, next_token);
        result_text += piece;
        
        // Prepariamo per il ciclo successivo
        token = next_token;
        pos++;
    }
    
    return result_text;
}
