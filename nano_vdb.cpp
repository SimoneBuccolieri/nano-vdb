#include "nano_vdb.hpp"
#include <algorithm>
#include <arm_neon.h> // Hardware instructions for Apple Silicon (extreme speed)
#include <cmath>
#include <fstream>
#include <stdexcept>

// PRIVATE CORE FUNCTION: Calculates Cosine Similarity between two vectors.
// The closer the score is to 1.0, the more semantically similar the texts are.
float NanoVectorDB::calculate_similarity(const std::vector<float> &a,
                                         const std::vector<float> &b) const {
  // --- NEON (SIMD) EXPLANATION ---
  // SIMD stands for "Single Instruction, Multiple Data". 
  // Instead of performing one multiplication at a time, the Apple Silicon chip
  // can process 4 simultaneously. 'float32x4_t' acts as a register holding 4 floats.
  
  size_t simd_loops = dimensions / 4; 

  // vdupq_n_f32(0.0f): Fill a register with four zeros: [0.0, 0.0, 0.0, 0.0]
  float32x4_t v_dot_product = vdupq_n_f32(0.0f);
  float32x4_t v_norm_a = vdupq_n_f32(0.0f);
  float32x4_t v_norm_b = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (size_t j = 0; j < simd_loops; ++j) {
    // vld1q_f32: Load 4 floats from memory into registers 'qa' and 'qb'
    float32x4_t qa = vld1q_f32(&a[i]);
    float32x4_t qb = vld1q_f32(&b[i]);

    // vmlaq_f32: "Vector Multiply Accumulate". 
    // Executes in a single CPU clock cycle:
    // 1. Multiplies 'qa' and 'qb' (4 pairs of numbers at once)
    // 2. Accumulates (adds) the result into 'v_dot_product'
    v_dot_product = vmlaq_f32(v_dot_product, qa, qb);
    
    // Similarly, calculate the vector magnitude (norm) for A and B
    v_norm_a = vmlaq_f32(v_norm_a, qa, qa);
    v_norm_b = vmlaq_f32(v_norm_b, qb, qb);

    i += 4; 
  }

  // vaddvq_f32: "Vector Add Across". Takes the register with the 4 partial results
  // and sums them all together returning a single standard 'float'.
  float dot_product = vaddvq_f32(v_dot_product);
  float norm_a = vaddvq_f32(v_norm_a);
  float norm_b = vaddvq_f32(v_norm_b);

  // Handle any remaining elements if 'dimensions' is not a perfect multiple of 4
  for (; i < dimensions; ++i) {
    dot_product += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  // Final Cosine Similarity formula
  // Prevents division by zero
  if (norm_a > 0.0f && norm_b > 0.0f) {
    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
  }
  return 0.0f;
}

// Utility function to help std::partial_sort order {index, score} pairs.
// Returns 'true' if element 'a' has a higher score than 'b'.
bool compare_candidates(const std::pair<size_t, float> &a, const std::pair<size_t, float> &b) {
  return a.second > b.second;
}

// GRAPH CONSTRUCTION & INSERTION
bool NanoVectorDB::insert(const std::string &id,
                          const std::vector<float> &vec) {
  if (vec.size() != dimensions)
    return false;

  // Explicit initialization for clarity
  VectorRecord new_record;
  new_record.id = id;
  new_record.data = vec;
  // new_record.neighbors is automatically initialized as empty
  
  registry.push_back(new_record);
  size_t new_index = registry.size() - 1;

  // If there are existing nodes, connect the new node to its most similar neighbors
  if (registry.size() > 1) {
    std::vector<std::pair<size_t, float>> candidates;
    candidates.reserve(new_index);

    // EXPENSIVE PHASE: Scan all previous nodes
    for (size_t i = 0; i < new_index; ++i) {
      float score = calculate_similarity(vec, registry[i].data);
      
      std::pair<size_t, float> pair_score;
      pair_score.first = i;
      pair_score.second = score;
      candidates.push_back(pair_score);
    }

    size_t actual_m = std::min(M, candidates.size());
    
    // --- std::partial_sort EXPLANATION ---
    // Instead of sorting the ENTIRE list (e.g., 10,000 elements) which is slow,
    // partial_sort only extracts the top 'actual_m' (e.g., 4) highest elements.
    // It places the top 4 at the very beginning of the list in perfect order,
    // and leaves the rest of the list unsorted, saving significant CPU cycles.
    std::partial_sort(
        candidates.begin(), 
        candidates.begin() + actual_m, // We only care about the top M
        candidates.end(),
        compare_candidates 
    );

    // Create bidirectional connections (edges) in the graph
    for (size_t i = 0; i < actual_m; ++i) {
      size_t neighbor_idx = candidates[i].first;
      
      // The new node adds the old node as a neighbor
      registry[new_index].neighbors.push_back(neighbor_idx);

      // The old node adds the new node to ITS neighbors (if it has space < M)
      if (registry[neighbor_idx].neighbors.size() < M) {
        registry[neighbor_idx].neighbors.push_back(new_index);
      }
    }
  }
  return true;
}

// Utility function to help std::partial_sort order SearchResults
bool compare_results(const SearchResult &a, const SearchResult &b) {
  return a.score > b.score; 
}

// LINEAR SEARCH (Brute Force with SIMD optimizations)
std::vector<SearchResult>
NanoVectorDB::search_linear(const std::vector<float> &query_vec,
                            size_t top_k) const {
  if (query_vec.size() != dimensions || registry.empty()) {
    std::vector<SearchResult> empty_res;
    return empty_res;
  }

  size_t actual_k = std::min(top_k, registry.size());
  std::vector<SearchResult> all_results;
  all_results.reserve(registry.size());

  // Loop through every single record in the DB
  // Explicit type is used instead of 'auto' for clarity
  for (const VectorRecord &record : registry) {
    float score = calculate_similarity(query_vec, record.data);
    
    SearchResult result;
    result.id = record.id;
    result.score = score;
    all_results.push_back(result);
  }

  // Sort to bring the top 'K' to the front, ignoring the rest
  std::partial_sort(
      all_results.begin(), 
      all_results.begin() + actual_k,
      all_results.end(),
      compare_results
  );

  // Return only the top K elements
  std::vector<SearchResult> top_results;
  for (size_t i = 0; i < actual_k; ++i) {
    top_results.push_back(all_results[i]);
  }
  return top_results;
}

// GRAPH SEARCH (Greedy Walk) - Complexity O(log N)
std::vector<SearchResult>
NanoVectorDB::search_graph(const std::vector<float> &query_vec,
                           size_t top_k) const {
  if (query_vec.size() != dimensions || registry.empty()) {
    std::vector<SearchResult> empty_res;
    return empty_res;
  }

  // Keep track of visited nodes to avoid infinite loops
  std::vector<bool> visited(registry.size(), false);
  
  // Track all nodes "touched" during the journey. The winners will be among these.
  std::vector<SearchResult> path_candidates;

  // Start from a fixed entry point (Node 0)
  size_t current_idx = 0;
  float current_score = calculate_similarity(query_vec, registry[current_idx].data);
  visited[current_idx] = true;
  
  SearchResult first_step;
  first_step.id = registry[current_idx].id;
  first_step.score = current_score;
  path_candidates.push_back(first_step);

  bool keep_walking = true;
  while (keep_walking) {
    keep_walking = false;
    size_t best_neighbor_idx = current_idx;
    float best_neighbor_score = current_score;

    // Explore ONLY the neighbors directly connected to the current node
    for (size_t neighbor_idx : registry[current_idx].neighbors) {
      if (!visited[neighbor_idx]) {
        visited[neighbor_idx] = true;
        
        float score = calculate_similarity(query_vec, registry[neighbor_idx].data);
        
        SearchResult neighbor_candidate;
        neighbor_candidate.id = registry[neighbor_idx].id;
        neighbor_candidate.score = score;
        path_candidates.push_back(neighbor_candidate);

        // If this neighbor is more similar to the query than where we are now, 
        // it becomes our next step!
        if (score > best_neighbor_score) {
          best_neighbor_score = score;
          best_neighbor_idx = neighbor_idx;
          keep_walking = true; // Better path found, keep walking!
        }
      }
    }

    // Materially "jump" to the new node and repeat
    if (keep_walking) {
      current_idx = best_neighbor_idx;
      current_score = best_neighbor_score;
    }
  }

  size_t actual_k = std::min(top_k, path_candidates.size());
  
  // Rank the nodes touched during the walk to extract the absolute best
  std::partial_sort(
      path_candidates.begin(), 
      path_candidates.begin() + actual_k,
      path_candidates.end(),
      compare_results
  );

  std::vector<SearchResult> top_results;
  for (size_t i = 0; i < actual_k; ++i) {
    top_results.push_back(path_candidates[i]);
  }
  return top_results;
}

// PERSISTENCE (Saves both vectors and the graph index to disk)
bool NanoVectorDB::save_to_file(const std::string &filename) const {
  std::ofstream out(filename, std::ios::binary);
  if (!out)
    return false;

  out.write(reinterpret_cast<const char *>(&dimensions), sizeof(dimensions));
  size_t num_records = registry.size();
  out.write(reinterpret_cast<const char *>(&num_records), sizeof(num_records));

  for (const VectorRecord &record : registry) {
    size_t id_size = record.id.size();
    out.write(reinterpret_cast<const char *>(&id_size), sizeof(id_size));
    out.write(record.id.data(), id_size);
    
    out.write(reinterpret_cast<const char *>(record.data.data()),
              dimensions * sizeof(float));

    size_t num_neighbors = record.neighbors.size();
    out.write(reinterpret_cast<const char *>(&num_neighbors),
              sizeof(num_neighbors));
    if (num_neighbors > 0) {
      out.write(reinterpret_cast<const char *>(record.neighbors.data()),
                num_neighbors * sizeof(size_t));
    }
  }
  return true;
}

// Loads the database and the graph edges from disk
bool NanoVectorDB::load_from_file(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    return false;

  size_t file_dims = 0;
  in.read(reinterpret_cast<char *>(&file_dims), sizeof(file_dims));
  if (file_dims != dimensions)
    return false;

  size_t num_records = 0;
  in.read(reinterpret_cast<char *>(&num_records), sizeof(num_records));

  registry.clear();
  registry.reserve(num_records);

  for (size_t i = 0; i < num_records; ++i) {
    size_t id_size = 0;
    in.read(reinterpret_cast<char *>(&id_size), sizeof(id_size));

    std::string id(id_size, '\0');
    in.read(&id[0], id_size);

    std::vector<float> data(dimensions);
    in.read(reinterpret_cast<char *>(data.data()), dimensions * sizeof(float));

    size_t num_neighbors = 0;
    in.read(reinterpret_cast<char *>(&num_neighbors), sizeof(num_neighbors));
    std::vector<size_t> neighbors(num_neighbors);
    if (num_neighbors > 0) {
      in.read(reinterpret_cast<char *>(neighbors.data()),
              num_neighbors * sizeof(size_t));
    }

    // Explicit constructor instantiation for readability
    VectorRecord loaded_record;
    loaded_record.id = id;
    loaded_record.data = data;
    loaded_record.neighbors = neighbors;

    registry.push_back(loaded_record);
  }
  return true;
}