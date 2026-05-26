#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../concepts/distance_metric.hpp"
#include "../concepts/vector_element.hpp"
#include "../distances/inner_product.hpp"
#include "../distances/l2.hpp"
#include "hnsw_node.hpp"
#include "hnsw_threading.hpp"
#include "quantization_snapshot.hpp"

// Prefetch hints for vector data during graph traversal
// Define HNSW_DISABLE_PREFETCH to disable prefetching (for benchmarking)
// Uses __builtin_prefetch on GCC/Clang, _mm_prefetch on MSVC
#ifndef HNSW_DISABLE_PREFETCH
#if defined(__GNUC__) || defined(__clang__)
#define HNSW_PREFETCH_READ(addr) __builtin_prefetch(addr, 0, 3)  // read, high temporal locality
#define HNSW_PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3) // write, high temporal locality
#elif defined(_MSC_VER)
#include <intrin.h>
#define HNSW_PREFETCH_READ(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#define HNSW_PREFETCH_WRITE(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define HNSW_PREFETCH_READ(addr) ((void)0)
#define HNSW_PREFETCH_WRITE(addr) ((void)0)
#endif
#else
// Prefetching disabled for benchmarking
#define HNSW_PREFETCH_READ(addr) ((void)0)
#define HNSW_PREFETCH_WRITE(addr) ((void)0)
#endif

namespace sqlite_vec_cpp::index {

// Forward declaration for friend access
template <concepts::VectorElement, typename> class HNSWQuantizedSearch;

/// Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search
/// Provides 100-1000x speedup over brute-force for large corpora (>100K vectors)
///
/// @tparam StorageT Storage type for vectors (float or float16_t)
/// @tparam MetricT Distance metric type (must work with float spans)
template <concepts::VectorElement StorageT, typename MetricT> class HNSWIndex {
    friend class HNSWQuantizedSearch<StorageT, MetricT>;

public:
#ifdef SQLITE_VEC_CPP_TESTING
    using TestingAfterInsertPublishHook = std::function<void()>;

    static void testing_set_after_insert_publish_hook(TestingAfterInsertPublishHook hook) {
        testing_after_insert_publish_hook_ = std::move(hook);
    }

    static void testing_clear_after_insert_publish_hook() {
        testing_after_insert_publish_hook_ = nullptr;
    }
#endif

    /// Configuration parameters for HNSW index.
    ///
    /// Default M=16 follows the hnswlib reference implementation and is safe for
    /// general-purpose use. For modern high-dimensional embeddings (384-1536d),
    /// consider M=24 which provides significantly better recall (98.4% vs 92.9%
    /// at 768d/10K) at ~2x build cost. See benchmarks/hnsw_m_sweep_benchmark.
    ///
    /// Research basis: Elliott & Clark (2024), "The Impacts of Data, Ordering,
    /// and Intrinsic Dimensionality on Recall in HNSW" found that real embedding
    /// vectors benefit from higher connectivity than SIFT1M-calibrated defaults.
    struct Config {
        size_t M = 16;                ///< Connections per node per layer (16-48 typical; 24 recommended for embeddings)
        size_t M_max = 32;            ///< Max connections for layers > 0 (usually 2*M)
        size_t M_max_0 = 64;          ///< Max connections at layer 0 (critical for recall, 2-4x M)
        size_t ef_construction = 200; ///< Exploration factor during construction (100-500; 200 minimum recommended)
        float ml_factor = 1.0f / std::log(2.0f); ///< Layer selection multiplier (1/ln(2))
        MetricT metric{};                        ///< Distance metric (operates on float spans)
        bool clamp_negative_distances =
            true; ///< Clamp negative distances to 0 (safe for L2/cosine)
        bool normalize_vectors =
            false; ///< Pre-normalize vectors for faster cosine distance during construction.
                   ///< When true, vectors are L2-normalized at insert time and distance
                   ///< computation uses inner product (1 - dot(a,b)) instead of full cosine.
                   ///< This gives ~3x speedup on NEON (4 FMA/iter vs 12 for full cosine).
                   ///< Only meaningful when metric is cosine; ignored for L2/IP metrics.
        uint32_t random_seed = 42; ///< Deterministic layer-selection seed.

        /// Create config optimized for high recall on large corpora
        /// @param corpus_size Expected number of vectors
        /// @param dim Vector dimensionality (higher dims need more connectivity)
        /// @return Config with parameters tuned for the corpus size
        static Config for_corpus(size_t corpus_size, size_t dim = 128) {
            Config cfg;

            // Higher M for high-dimensional data (embeddings typically 384-1536), but keep
            // 512+d on the same profile as 256+d to avoid pathological build times for
            // medium-sized corpora where reranking/refinement already recovers quality.
            if (dim >= 256) {
                cfg.M = 16;
                cfg.M_max = 32;
                cfg.M_max_0 = 64;
            } else if (dim >= 128) {
                cfg.M = 16;
                cfg.M_max = 32;
                cfg.M_max_0 = 64;
            } else {
                cfg.M = 12;
                cfg.M_max = 24;
                cfg.M_max_0 = 48;
            }

            // Higher ef_construction for larger corpora.
            // Minimum of 200 provides baseline recall quality; 400 for large corpora.
            if (corpus_size >= 100000) {
                cfg.ef_construction = 400;
            } else {
                cfg.ef_construction = 200;
            }

            return cfg;
        }
    };

    /// Candidate-set PHSS rerank over ANN results.
    ///
    /// This is intentionally post-search rather than inside beam traversal:
    /// we first fetch a small ANN candidate set, then build a local similarity
    /// graph and diffuse query mass over that graph. The default criterion uses
    /// a cheap LargestGap approximation over the pairwise similarity
    /// distribution, which keeps the hot path practical for SQLite workloads.
    struct PhssRerankConfig {
        enum class Mode : std::uint8_t {
            LargestGapApprox,
        };

        bool enabled = false;
        size_t candidates = 64;
        size_t min_candidates = 16;
        float blend_alpha = 0.8f;
        float attention_scale = 8.0f;
        size_t steps = 2;
        float self_weight = 0.5f;
        Mode mode = Mode::LargestGapApprox;
    };

    /// Alias for node type
    using NodeType = HNSWNode<StorageT>;

    /// Construct empty HNSW index
    explicit HNSWIndex(Config config = {}) : config_(config), rng_generator_(config.random_seed) {}

    /// Move constructor (required because std::shared_mutex is move-only)
    HNSWIndex(HNSWIndex&& other) noexcept
        : config_(other.config_), nodes_mutex_(), nodes_(std::move(other.nodes_)),
          deleted_ids_(std::move(other.deleted_ids_)),
          next_dense_id_(other.next_dense_id_.load(std::memory_order_relaxed)),
          entry_point_id_(other.entry_point_id_.load(std::memory_order_relaxed)),
          entry_point_layer_(other.entry_point_layer_.load(std::memory_order_relaxed)),
          rng_generator_(config_.random_seed) {}

    /// Move assignment
    HNSWIndex& operator=(HNSWIndex&& other) noexcept {
        if (this != &other) {
            config_ = other.config_;
            rng_generator_.reseed(config_.random_seed);
            nodes_ = std::move(other.nodes_);
            deleted_ids_ = std::move(other.deleted_ids_);
            next_dense_id_.store(other.next_dense_id_.load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
            entry_point_id_.store(other.entry_point_id_.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
            entry_point_layer_.store(other.entry_point_layer_.load(std::memory_order_relaxed),
                                     std::memory_order_relaxed);
        }
        return *this;
    }

    /// Delete copy operations (std::shared_mutex is not copyable)
    HNSWIndex(const HNSWIndex&) = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;

    /// Factory method for deserialization
    /// Reconstructs HNSW index from serialized state without rebuilding graph
    static HNSWIndex from_serialized(const Config& config, size_t entry_point_id,
                                     size_t entry_point_layer,
                                     std::unordered_map<size_t, NodeType>&& nodes) {
        HNSWIndex index(config);
        index.entry_point_id_.store(entry_point_id, std::memory_order_relaxed);
        index.entry_point_layer_.store(entry_point_layer, std::memory_order_relaxed);
        index.nodes_ = std::move(nodes);
        index.rebuild_dense_ids_unlocked();
        return index;
    }

    /// Factory method for deserialization with deleted nodes
    static HNSWIndex from_serialized(const Config& config, size_t entry_point_id,
                                     size_t entry_point_layer,
                                     std::unordered_map<size_t, NodeType>&& nodes,
                                     std::unordered_set<size_t>&& deleted_ids) {
        HNSWIndex index(config);
        index.entry_point_id_.store(entry_point_id, std::memory_order_relaxed);
        index.entry_point_layer_.store(entry_point_layer, std::memory_order_relaxed);
        index.nodes_ = std::move(nodes);
        index.deleted_ids_ = std::move(deleted_ids);
        index.rebuild_dense_ids_unlocked();
        return index;
    }

    /// Insert vector into index (storage type StorageT, converts to float32 for graph operations)
    void insert(size_t id, std::span<const StorageT> vector) {
        // For float storage, use input directly; for fp16, convert to float32
        // Use stack buffer for typical embedding dimensions (up to 1536)
        // to avoid heap allocation in the common case
        alignas(32) float stack_buffer[1536];
        std::span<const float> vector_f32;
        std::vector<float> heap_buffer; // Only used if vector.size() > 1536

        if constexpr (std::same_as<StorageT, float>) {
            vector_f32 = vector;
        } else {
            if (vector.size() <= 1536) {
                for (size_t i = 0; i < vector.size(); ++i) {
                    stack_buffer[i] = static_cast<float>(vector[i]);
                }
                vector_f32 = std::span<const float>(stack_buffer, vector.size());
            } else {
                heap_buffer = to_float_vector(vector);
                vector_f32 = heap_buffer;
            }
        }

        // Pre-normalize for cosine acceleration
        std::vector<float> norm_buffer;
        std::vector<StorageT> normalized_store;
        std::span<const StorageT> store_vec = vector;
        if (config_.normalize_vectors) {
            if constexpr (std::same_as<StorageT, float>) {
                norm_buffer.assign(vector.begin(), vector.end());
                normalize_vector_inplace(norm_buffer);
                store_vec = std::span<const StorageT>(norm_buffer);
                vector_f32 = std::span<const float>(norm_buffer);
            } else {
                norm_buffer.assign(vector_f32.begin(), vector_f32.end());
                normalize_vector_inplace(norm_buffer);
                if constexpr (std::same_as<StorageT, utils::float16_t>) {
                    normalized_store = utils::to_float16(std::span<const float>(norm_buffer));
                } else {
                    normalized_store.assign(norm_buffer.begin(), norm_buffer.end());
                }
                store_vec = std::span<const StorageT>(normalized_store);
                vector_f32 = std::span<const float>(norm_buffer);
            }
        }

        // Assign random layer using thread-local RNG
        size_t layer = random_layer();

        // Create and insert node with write lock (brief)
        bool is_first_node = false;
        bool is_new_entry_point = false;
        size_t old_entry = 0;
        size_t old_layer = 0;
        {
            std::unique_lock lock(nodes_mutex_);
            // Pass M_max hint for edge pre-allocation
            auto [it, inserted] = nodes_.emplace(id, NodeType(id, store_vec, layer, config_.M_max));
            if (!inserted) {
                return;
            }
            it->second.dense_id = next_dense_id_++;
            // Bump generation while holding write lock so snapshot readers
            // see a consistent (node-visible, generation-bumped) state.
            mutation_generation_.fetch_add(1, std::memory_order_release);

            // First node becomes entry point
            if (nodes_.size() == 1) {
                entry_point_id_.store(id, std::memory_order_relaxed);
                entry_point_layer_.store(layer, std::memory_order_relaxed);
                is_first_node = true;
            } else if (layer > entry_point_layer_.load(std::memory_order_relaxed)) {
                // Update entry point if new node has higher layer
                old_entry = entry_point_id_.load(std::memory_order_relaxed);
                old_layer = entry_point_layer_.load(std::memory_order_relaxed);
                entry_point_id_.store(id, std::memory_order_relaxed);
                entry_point_layer_.store(layer, std::memory_order_relaxed);
                is_new_entry_point = true;
            }
        } // Release write lock immediately

#ifdef SQLITE_VEC_CPP_TESTING
        if (testing_after_insert_publish_hook_) {
            testing_after_insert_publish_hook_();
        }
#endif

        if (is_first_node) {
            return;
        }

        if (is_new_entry_point) {
            // Handle new entry point case - connect from old entry point down
            size_t current = old_entry;
            for (size_t lc = old_layer;; --lc) {
                // Search with read lock
                std::vector<std::pair<size_t, float>> candidates;
                {
                    std::shared_lock lock(nodes_mutex_);
                    std::shared_lock deleted_lock(deleted_mutex_);
                    candidates = beam_search_layer_locked(vector_f32, current,
                                                          config_.ef_construction, lc, nullptr);
                }

                // Connect with write lock
                {
                    std::unique_lock lock(nodes_mutex_);
                    size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
                    connect_neighbors(id, candidates, M, lc);
                    if (lc == 0) {
                        // Final insert generation bump: a snapshot captured after node publish but
                        // before the graph connections completed must not look fresh after
                        // insert().
                        mutation_generation_.fetch_add(1, std::memory_order_release);
                    }
                }

                if (lc == 0)
                    break;
                if (!candidates.empty()) {
                    current = candidates[0].first;
                }
            }
            return;
        }

        // Normal case: navigate from entry point to find insertion point at each layer
        size_t current = entry_point_id_.load(std::memory_order_acquire);

        // Greedy search through upper layers (uses internal read lock)
        for (size_t lc = entry_point_layer_.load(std::memory_order_acquire); lc > layer; --lc) {
            std::shared_lock lock(nodes_mutex_);
            std::shared_lock deleted_lock(deleted_mutex_);
            current = greedy_search_layer_locked(vector_f32, current, lc, nullptr);
        }

        // Beam search and connect at each layer from node's layer down to 0
        for (size_t lc = layer;; --lc) {
            // Search with read lock (expensive operation)
            std::vector<std::pair<size_t, float>> candidates;
            {
                std::shared_lock lock(nodes_mutex_);
                std::shared_lock deleted_lock(deleted_mutex_);
                candidates = beam_search_layer_locked(vector_f32, current, config_.ef_construction,
                                                      lc, nullptr);
            }

            // Connect with write lock (cheap operation)
            {
                std::unique_lock lock(nodes_mutex_);
                size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
                connect_neighbors(id, candidates, M, lc);
                if (lc == 0) {
                    // Final insert generation bump: a snapshot captured after node publish but
                    // before the graph connections completed must not look fresh after insert().
                    mutation_generation_.fetch_add(1, std::memory_order_release);
                }
            }

            if (lc == 0)
                break;
            if (!candidates.empty()) {
                current = candidates[0].first;
            }
        }
    }

    /// Single-threaded insert — no locking, no edge copies, fastest path for bulk builds
    /// WARNING: NOT thread-safe. Use only when no concurrent readers or writers exist.
    /// For concurrent workloads, use insert() instead.
    void insert_single_threaded(size_t id, std::span<const StorageT> vector) {
        // Convert to float32 if needed
        alignas(32) float stack_buffer[1536];
        std::span<const float> vector_f32;
        std::vector<float> heap_buffer;

        if constexpr (std::same_as<StorageT, float>) {
            vector_f32 = vector;
        } else {
            if (vector.size() <= 1536) {
                for (size_t i = 0; i < vector.size(); ++i) {
                    stack_buffer[i] = static_cast<float>(vector[i]);
                }
                vector_f32 = std::span<const float>(stack_buffer, vector.size());
            } else {
                heap_buffer = to_float_vector(vector);
                vector_f32 = heap_buffer;
            }
        }

        // Pre-normalize for cosine: normalize the vector before storing.
        // After this, distance_nodes/distance_query_node use inner product (1 - dot).
        std::vector<float> norm_buffer;
        std::vector<StorageT> normalized_store;
        std::span<const StorageT> store_vec = vector; // what gets stored in the node
        if (config_.normalize_vectors) {
            if constexpr (std::same_as<StorageT, float>) {
                // For float storage, normalize in-place via buffer
                norm_buffer.assign(vector.begin(), vector.end());
                normalize_vector_inplace(norm_buffer);
                store_vec = std::span<const StorageT>(norm_buffer);
                vector_f32 = std::span<const float>(norm_buffer);
            } else {
                // For fp16: normalize the float32 version, then store as fp16
                norm_buffer.assign(vector_f32.begin(), vector_f32.end());
                normalize_vector_inplace(norm_buffer);
                if constexpr (std::same_as<StorageT, utils::float16_t>) {
                    normalized_store = utils::to_float16(std::span<const float>(norm_buffer));
                } else {
                    normalized_store.assign(norm_buffer.begin(), norm_buffer.end());
                }
                store_vec = std::span<const StorageT>(normalized_store);
                vector_f32 = std::span<const float>(norm_buffer);
            }
        }

        size_t layer = random_layer();

        // Create node — store normalized vector when normalize_vectors is enabled
        auto [it, inserted] = nodes_.emplace(id, NodeType(id, store_vec, layer, config_.M_max));
        if (!inserted)
            return;
        it->second.dense_id = next_dense_id_.fetch_add(1, std::memory_order_relaxed);
        register_flat_lookup(id, &it->second);

        // First node
        if (nodes_.size() == 1) {
            entry_point_id_.store(id, std::memory_order_relaxed);
            entry_point_layer_.store(layer, std::memory_order_relaxed);
            return;
        }

        size_t ep_layer = entry_point_layer_.load(std::memory_order_relaxed);

        if (layer > ep_layer) {
            size_t current = entry_point_id_.load(std::memory_order_relaxed);
            for (size_t lc = ep_layer;; --lc) {
                auto candidates =
                    beam_search_layer_unlocked(vector_f32, current, config_.ef_construction, lc);
                size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
                connect_neighbors_unlocked(id, candidates, M, lc);
                if (lc == 0)
                    break;
                if (!candidates.empty())
                    current = candidates[0].first;
            }
            entry_point_id_.store(id, std::memory_order_relaxed);
            entry_point_layer_.store(layer, std::memory_order_relaxed);
            return;
        }

        // Normal case: greedy descent through upper layers, beam search + connect at lower layers
        size_t current = entry_point_id_.load(std::memory_order_relaxed);

        for (size_t lc = ep_layer; lc > layer; --lc) {
            current = greedy_search_layer_unlocked(vector_f32, current, lc);
        }

        for (size_t lc = layer;; --lc) {
            auto candidates =
                beam_search_layer_unlocked(vector_f32, current, config_.ef_construction, lc);
            size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
            connect_neighbors_unlocked(id, candidates, M, lc);
            if (lc == 0)
                break;
            if (!candidates.empty())
                current = candidates[0].first;
        }

        mutation_generation_.fetch_add(1, std::memory_order_release);
    }

    /// Search for k nearest neighbors (query is always float32)
    /// @param query Query vector (float32)
    /// @param k Number of neighbors to return
    /// @param ef_search Exploration factor (higher = better recall, slower)
    /// @return Vector of (id, distance) pairs, sorted by distance
    using FilterFn = std::function<bool(size_t node_id)>;

    std::vector<std::pair<size_t, float>> search(std::span<const float> query, size_t k,
                                                 size_t ef_search = 50) const {
        return search_with_filter(query, k, ef_search, nullptr);
    }

    /// Search with pre-filtering
    /// @param query Query vector (float32)
    /// @param k Number of neighbors to return
    /// @param ef_search Exploration factor (higher = better recall, slower)
    /// @param filter Optional filter function: returns true if node should be included
    /// @return Vector of (id, distance) pairs, sorted by distance
    std::vector<std::pair<size_t, float>> search_with_filter(std::span<const float> query, size_t k,
                                                             size_t ef_search,
                                                             const FilterFn& filter) const {
        return search_with_filter_impl<false>(query, k, ef_search, filter);
    }

    /// Search optimized for read-mostly workloads.
    /// Uses direct neighbor references under the index read lock to avoid per-hop
    /// neighbor vector copies. Do not use concurrently with unlocked bulk-build or
    /// single-threaded insert APIs that mutate graph edges without taking nodes_mutex_.
    std::vector<std::pair<size_t, float>> search_read_mostly(std::span<const float> query, size_t k,
                                                             size_t ef_search = 50) const {
        return search_read_mostly_with_filter(query, k, ef_search, nullptr);
    }

    std::vector<std::pair<size_t, float>> search_phss_rerank(std::span<const float> query, size_t k,
                                                             size_t ef_search,
                                                             const PhssRerankConfig& cfg) const {
        return search_phss_rerank_with_filter(query, k, ef_search, cfg, nullptr);
    }

    /// Read-mostly search with pre-filtering.
    std::vector<std::pair<size_t, float>>
    search_read_mostly_with_filter(std::span<const float> query, size_t k, size_t ef_search,
                                   const FilterFn& filter) const {
        return search_with_filter_impl<true>(query, k, ef_search, filter);
    }

    std::vector<std::pair<size_t, float>>
    search_phss_rerank_with_filter(std::span<const float> query, size_t k, size_t ef_search,
                                   const PhssRerankConfig& cfg, const FilterFn& filter) const {
        if (!cfg.enabled) {
            return search_read_mostly_with_filter(query, k, ef_search, filter);
        }

        const size_t candidate_k = std::max(k, cfg.candidates);
        auto approx_candidates =
            search_read_mostly_with_filter(query, candidate_k, ef_search, filter);
        if (approx_candidates.size() <= k) {
            return approx_candidates;
        }
        if (approx_candidates.size() < cfg.min_candidates) {
            approx_candidates.resize(k);
            return approx_candidates;
        }

        const size_t n = approx_candidates.size();
        auto tri_index = [n](size_t i, size_t j) {
            // Dense upper-triangular packing for i < j.
            return i * (2 * n - i - 1) / 2 + (j - i - 1);
        };

        std::vector<float> query_signal(n, 0.0f);
        float max_query_logit = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n; ++i) {
            query_signal[i] = 1.0f / (1.0f + std::max(0.0f, approx_candidates[i].second));
            max_query_logit = std::max(max_query_logit, cfg.attention_scale * query_signal[i]);
        }

        std::vector<float> mass(n, 0.0f);
        float mass_sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            mass[i] = std::exp(cfg.attention_scale * query_signal[i] - max_query_logit);
            mass_sum += mass[i];
        }
        if (mass_sum <= 0.0f) {
            approx_candidates.resize(k);
            return approx_candidates;
        }
        for (float& x : mass)
            x /= mass_sum;

        std::vector<const NodeType*> candidate_nodes;
        candidate_nodes.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            candidate_nodes.push_back(try_get_node(approx_candidates[i].first));
        }

        const size_t tri_count = n * (n - 1) / 2;
        std::vector<float> sims_tri(tri_count, 0.0f);
        std::vector<float> sims_sorted;
        sims_sorted.reserve(tri_count);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                float sim = 0.0f;
                const auto* ni = candidate_nodes[i];
                const auto* nj = candidate_nodes[j];
                if (ni && nj) {
                    const float d = distance_nodes(*ni, *nj);
                    sim = 1.0f / (1.0f + std::max(0.0f, d));
                }
                sims_tri[tri_index(i, j)] = sim;
                sims_sorted.push_back(sim);
            }
        }

        float scale = 0.0f;
        if (!sims_sorted.empty()) {
            std::sort(sims_sorted.begin(), sims_sorted.end());
            if (sims_sorted.size() == 1) {
                scale = sims_sorted[0];
            } else {
                float max_gap = -1.0f;
                size_t best_idx = 0;
                for (size_t i = 0; i + 1 < sims_sorted.size(); ++i) {
                    const float gap = sims_sorted[i + 1] - sims_sorted[i];
                    if (gap > max_gap) {
                        max_gap = gap;
                        best_idx = i;
                    }
                }
                scale = (sims_sorted[best_idx] + sims_sorted[best_idx + 1]) * 0.5f;
            }
        }
        if (scale <= 0.0f) {
            approx_candidates.resize(k);
            return approx_candidates;
        }

        std::vector<std::vector<std::pair<size_t, float>>> adj(n);
        for (size_t i = 0; i < n; ++i) {
            float row_max = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < n; ++j) {
                if (i == j)
                    continue;
                const float sim = (i < j) ? sims_tri[tri_index(i, j)] : sims_tri[tri_index(j, i)];
                if (sim >= scale) {
                    row_max = std::max(row_max, cfg.attention_scale * sim);
                    adj[i].push_back({j, sim});
                }
            }
            if (adj[i].empty())
                continue;
            float row_sum = 0.0f;
            for (auto& [j, sim] : adj[i]) {
                sim = std::exp(cfg.attention_scale * sim - row_max);
                row_sum += sim;
            }
            if (row_sum > 0.0f) {
                for (auto& [j, sim] : adj[i])
                    sim /= row_sum;
            }
        }

        std::vector<float> next_mass(n, 0.0f);
        for (size_t step = 0; step < cfg.steps; ++step) {
            std::fill(next_mass.begin(), next_mass.end(), 0.0f);
            for (size_t i = 0; i < n; ++i) {
                next_mass[i] += cfg.self_weight * mass[i];
                if (adj[i].empty())
                    continue;
                const float carry = (1.0f - cfg.self_weight) * mass[i];
                for (const auto& [j, w] : adj[i])
                    next_mass[j] += carry * w;
            }
            mass.swap(next_mass);
        }

        auto zscore = [](std::vector<float>& v) {
            if (v.empty())
                return;
            double mean = 0.0;
            for (float x : v)
                mean += x;
            mean /= static_cast<double>(v.size());
            double var = 0.0;
            for (float x : v)
                var += (x - mean) * (x - mean);
            const float sd =
                static_cast<float>(std::sqrt(var / static_cast<double>(v.size())) + 1e-12);
            for (float& x : v)
                x = static_cast<float>((x - mean) / sd);
        };

        std::vector<float> query_z(n, 0.0f);
        for (size_t i = 0; i < n; ++i)
            query_z[i] = -approx_candidates[i].second;
        zscore(query_z);
        zscore(mass);

        std::vector<std::pair<size_t, float>> reranked;
        reranked.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const float score = cfg.blend_alpha * query_z[i] + (1.0f - cfg.blend_alpha) * mass[i];
            reranked.emplace_back(approx_candidates[i].first, -score);
        }
        std::sort(reranked.begin(), reranked.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        reranked.resize(k);
        return reranked;
    }

    /// Batch search for multiple queries in parallel
    /// Processes queries across multiple threads for higher throughput
    /// @param queries Vector of query vectors (each is float32)
    /// @param k Number of neighbors to return per query
    /// @param ef_search Exploration factor (higher = better recall, slower)
    /// @param num_threads Number of threads (0 = auto-detect based on hardware)
    /// @return Vector of results, one per query (each is vector of (id, distance) pairs)
    std::vector<std::vector<std::pair<size_t, float>>>
    search_batch(std::span<const std::span<const float>> queries, size_t k, size_t ef_search = 50,
                 size_t num_threads = 0) const {
        return search_batch_with_filter(queries, k, ef_search, nullptr, num_threads);
    }

    /// Batch search with pre-filtering
    /// @param queries Vector of query vectors (each is float32)
    /// @param k Number of neighbors to return per query
    /// @param ef_search Exploration factor (higher = better recall, slower)
    /// @param filter Optional filter function: returns true if node should be included
    /// @param num_threads Number of threads (0 = auto-detect based on hardware)
    /// @return Vector of results, one per query (each is vector of (id, distance) pairs)
    std::vector<std::vector<std::pair<size_t, float>>>
    search_batch_with_filter(std::span<const std::span<const float>> queries, size_t k,
                             size_t ef_search, FilterFn filter, size_t num_threads = 0) const {
        if (queries.empty() || nodes_.empty()) {
            return std::vector<std::vector<std::pair<size_t, float>>>(queries.size());
        }

        size_t actual_threads = num_threads ? num_threads : std::thread::hardware_concurrency();
        if (actual_threads == 0) {
            actual_threads = 1;
        }

        // For small batches, sequential is faster (avoids thread overhead)
        if (queries.size() < 4 || actual_threads == 1) {
            std::vector<std::vector<std::pair<size_t, float>>> results;
            results.reserve(queries.size());
            for (const auto& query : queries) {
                results.push_back(search_with_filter(query, k, ef_search, filter));
            }
            return results;
        }

        // Pre-allocate results
        std::vector<std::vector<std::pair<size_t, float>>> results(queries.size());

        // Use thread pool for parallel search
        ThreadPool pool(actual_threads);
        pool.parallel_for(queries.size(), [&](size_t /*thread_id*/, size_t query_idx) {
            results[query_idx] = search_with_filter(queries[query_idx], k, ef_search, filter);
        });

        return results;
    }

    /// Batch build from vectors (sequential, uses optimized unlocked insert)
    void build(std::span<const size_t> ids, std::span<const std::span<const StorageT>> vectors) {
        if (ids.size() != vectors.size()) {
            throw std::invalid_argument("ids and vectors must have same size");
        }

        reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            insert_single_threaded(ids[i], vectors[i]);
        }
    }

    /// Reserve capacity for expected number of nodes
    /// Call before bulk insert_single_threaded() for best performance
    void reserve(size_t expected_size) {
        nodes_.reserve(expected_size);
        flat_lookup_.reserve(expected_size);
    }

    /// Parallel batch build from vectors
    /// Uses concurrent insert() calls from multiple threads (hnswlib pattern)
    /// Note: For bulk building, sequential may be faster due to lock contention.
    /// This approach shines for concurrent multi-client workloads.
    /// @param ids Vector of IDs to insert
    /// @param vectors Vector of vectors to insert (must match ids.size())
    /// @param num_threads Number of threads (0 = auto-detect)
    void build_parallel(std::span<const size_t> ids,
                        std::span<const std::span<const StorageT>> vectors,
                        size_t num_threads = 0) {
        build_parallel(ids, vectors, num_threads, 256);
    }

    /// Parallel batch build with configurable batch size
    /// Uses the same thread-safe insert() path, but schedules work in batches to
    /// reduce per-item scheduling overhead.
    /// @param ids Vector of IDs to insert
    /// @param vectors Vector of vectors to insert (must match ids.size())
    /// @param num_threads Number of threads (0 = auto-detect)
    /// @param batch_size Size of batches for parallel phases (default 256)
    void build_parallel(std::span<const size_t> ids,
                        std::span<const std::span<const StorageT>> vectors, size_t num_threads,
                        size_t batch_size) {
        if (ids.size() != vectors.size()) {
            throw std::invalid_argument("ids and vectors must have same size");
        }

        if (ids.empty())
            return;

        if (ids.size() <= 1) {
            for (size_t i = 0; i < ids.size(); ++i) {
                insert(ids[i], vectors[i]);
            }
            return;
        }

        size_t start_index = 0;
        if (nodes_.empty()) {
            insert(ids[0], vectors[0]);
            start_index = 1;
        }

        if (start_index >= ids.size())
            return;

        auto ids_span = ids.subspan(start_index);
        auto vectors_span = vectors.subspan(start_index);

        size_t actual_threads = num_threads ? num_threads : std::thread::hardware_concurrency();
        if (actual_threads == 0) {
            actual_threads = 1;
        }

        ThreadPool pool(actual_threads);

        constexpr size_t DEFAULT_BATCH_SIZE = 256;
        size_t actual_batch_size = batch_size ? batch_size : DEFAULT_BATCH_SIZE;

        if (actual_batch_size >= ids_span.size()) {
            pool.parallel_for(ids_span.size(), [&](size_t /*thread*/, size_t i) {
                insert(ids_span[i], vectors_span[i]);
            });
            return;
        }

        size_t batch_count = (ids_span.size() + actual_batch_size - 1) / actual_batch_size;

        std::atomic<size_t> max_layer{entry_point_layer_.load(std::memory_order_relaxed)};
        std::vector<size_t> node_layers(ids_span.size());

        for (size_t batch = 0; batch < batch_count; ++batch) {
            size_t batch_start = batch * actual_batch_size;
            size_t batch_end = std::min(batch_start + actual_batch_size, ids_span.size());
            size_t batch_items = batch_end - batch_start;

            // Phase 1: Create nodes sequentially (unordered_map mutation is inherently serial)
            // No thread pool overhead — the global lock serialized this anyway.
            for (size_t idx = 0; idx < batch_items; ++idx) {
                size_t i = batch_start + idx;
                size_t layer = random_layer();
                node_layers[i] = layer;

                // Pre-normalize vector for cosine acceleration
                std::vector<float> norm_buf;
                std::vector<StorageT> normalized_store;
                std::span<const StorageT> store_vec = vectors_span[i];
                if (config_.normalize_vectors) {
                    if constexpr (std::same_as<StorageT, float>) {
                        norm_buf.assign(vectors_span[i].begin(), vectors_span[i].end());
                        normalize_vector_inplace(norm_buf);
                        store_vec = std::span<const StorageT>(norm_buf);
                    } else {
                        norm_buf.reserve(vectors_span[i].size());
                        for (const auto& value : vectors_span[i]) {
                            norm_buf.push_back(static_cast<float>(value));
                        }
                        normalize_vector_inplace(norm_buf);
                        if constexpr (std::same_as<StorageT, utils::float16_t>) {
                            normalized_store = utils::to_float16(std::span<const float>(norm_buf));
                        } else {
                            normalized_store.assign(norm_buf.begin(), norm_buf.end());
                        }
                        store_vec = std::span<const StorageT>(normalized_store);
                    }
                }

                auto [it, inserted] = nodes_.emplace(
                    ids_span[i], NodeType(ids_span[i], store_vec, layer, config_.M_max));
                if (!inserted) {
                    continue;
                }
                it->second.dense_id = next_dense_id_.fetch_add(1, std::memory_order_relaxed);

                size_t current_max = max_layer.load(std::memory_order_relaxed);
                while (layer > current_max) {
                    if (max_layer.compare_exchange_weak(current_max, layer,
                                                        std::memory_order_relaxed,
                                                        std::memory_order_relaxed)) {
                        break;
                    }
                }
            }

            // Phase 2: Graph construction in parallel (beam search + connect)

            pool.parallel_for(batch_items, [&](size_t /*thread*/, size_t idx) {
                size_t i = batch_start + idx;
                size_t layer = node_layers[i];

                alignas(32) float stack_buffer[1536];
                std::span<const float> vector_f32;
                std::vector<float> heap_buffer;

                if constexpr (std::same_as<StorageT, float>) {
                    vector_f32 = vectors_span[i];
                } else {
                    if (vectors_span[i].size() <= 1536) {
                        for (size_t d = 0; d < vectors_span[i].size(); ++d) {
                            stack_buffer[d] = static_cast<float>(vectors_span[i][d]);
                        }
                        vector_f32 = std::span<const float>(stack_buffer, vectors_span[i].size());
                    } else {
                        heap_buffer = to_float_vector(vectors_span[i]);
                        vector_f32 = heap_buffer;
                    }
                }

                // Normalize query vector for beam search (nodes already store normalized vectors)
                std::vector<float> norm_query;
                if (config_.normalize_vectors) {
                    norm_query = normalize_vector(vector_f32);
                    vector_f32 = std::span<const float>(norm_query);
                }

                size_t current = entry_point_id_.load(std::memory_order_acquire);
                for (size_t lc = entry_point_layer_.load(std::memory_order_acquire); lc > layer;
                     --lc) {
                    current = greedy_search_layer_batch(vector_f32, current, lc);
                }

                for (size_t lc = layer;; --lc) {
                    auto candidates =
                        beam_search_layer_batch(vector_f32, current, config_.ef_construction, lc);

                    size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
                    size_t num_connections = std::min(M, candidates.size());

                    for (size_t c = 0; c < num_connections; ++c) {
                        size_t neighbor_id = candidates[c].first;

                        auto* node_ptr = try_get_node(ids_span[i]);
                        auto* neighbor_ptr = try_get_node(neighbor_id);
                        if (!node_ptr || !neighbor_ptr)
                            continue;

                        node_ptr->add_edge(neighbor_id, lc);
                        neighbor_ptr->add_edge(ids_span[i], lc);

                        size_t M_max = (lc == 0) ? config_.M_max_0 : config_.M_max;
                        if (neighbor_ptr->neighbors(lc).size() > M_max) {
                            prune_connections_batch(neighbor_id, lc);
                        }
                    }

                    if (lc == 0)
                        break;
                    if (!candidates.empty()) {
                        current = candidates[0].first;
                    }
                }
            });
        }

        size_t max_l = max_layer.load(std::memory_order_relaxed);
        entry_point_layer_.store(max_l, std::memory_order_relaxed);

        for (size_t i = 0; i < ids_span.size(); ++i) {
            if (node_layers[i] == max_l) {
                entry_point_id_.store(ids_span[i], std::memory_order_relaxed);
                break;
            }
        }
    }

    /// Get number of vectors in index
    size_t size() const { return nodes_.size(); }

    /// Get maximum layer in index
    size_t max_layer() const { return entry_point_layer_; }

    /// Get entry point node ID
    size_t entry_point() const { return entry_point_id_; }

    /// Check if index is empty
    bool empty() const { return nodes_.empty(); }

    /// Get configuration
    const Config& config() const { return config_; }

    // ========== Adaptive Search API ==========

    /// Calculate recommended ef_search based on corpus size for target recall
    /// Uses empirical scaling: ef_search ≈ k * sqrt(N) / 10 for 95%+ recall
    /// @param k Number of neighbors to retrieve
    /// @param target_recall Target recall (0.9 = 90%, 0.95 = 95%, 0.99 = 99%)
    /// @return Recommended ef_search value
    [[nodiscard]] size_t recommended_ef_search(size_t k, float target_recall = 0.95f) const {
        size_t n = nodes_.size();
        if (n == 0)
            return k;

        // Base ef_search scales with sqrt(N) for consistent recall
        // Higher target_recall requires exponentially more exploration
        double base = static_cast<double>(k) * std::sqrt(static_cast<double>(n)) / 10.0;

        // Recall multiplier: 1.0 at 90%, 1.5 at 95%, 3.0 at 99%
        const double recall_factor = target_recall >= 0.99f   ? 3.0
                                     : target_recall >= 0.95f ? 1.5
                                     : target_recall >= 0.90f ? 1.0
                                                              : 0.7;

        size_t ef = static_cast<size_t>(base * recall_factor);

        // Clamp to reasonable range [k, 2000]
        return std::clamp(ef, k, size_t{2000});
    }

    /// Search with adaptive ef_search based on corpus size
    /// Automatically selects ef_search for target recall
    /// @param query Query vector (float32)
    /// @param k Number of neighbors to return
    /// @param target_recall Target recall (default 0.95 = 95%)
    /// @return Vector of (id, distance) pairs, sorted by distance
    std::vector<std::pair<size_t, float>> search_adaptive(std::span<const float> query, size_t k,
                                                          float target_recall = 0.95f) const {
        size_t ef = recommended_ef_search(k, target_recall);
        return search(query, k, ef);
    }

    // ========== Graph Quality Metrics ==========

    /// Statistics about the HNSW graph structure
    struct GraphStats {
        size_t num_nodes = 0;            ///< Total nodes in graph
        size_t num_layers = 0;           ///< Number of layers (max_layer + 1)
        size_t total_edges = 0;          ///< Total edges across all layers
        double avg_degree_layer0 = 0.0;  ///< Average degree at layer 0
        double avg_degree_upper = 0.0;   ///< Average degree at layers > 0
        size_t min_degree_layer0 = 0;    ///< Minimum degree at layer 0
        size_t max_degree_layer0 = 0;    ///< Maximum degree at layer 0
        size_t orphan_count = 0;         ///< Nodes with 0 edges at layer 0 (bad for recall)
        size_t underconnected = 0;       ///< Nodes with degree < M/2 at layer 0
        double connectivity_score = 0.0; ///< 0-1 score (1 = fully connected)

        /// Check if graph quality is acceptable for high recall
        [[nodiscard]] bool is_healthy() const {
            return orphan_count == 0 && connectivity_score >= 0.8 && min_degree_layer0 >= 2;
        }
    };

    /// Compute graph quality metrics
    /// @return GraphStats with connectivity analysis
    [[nodiscard]] GraphStats compute_graph_stats() const {
        GraphStats stats;
        stats.num_nodes = nodes_.size();

        if (nodes_.empty()) {
            return stats;
        }

        stats.num_layers = entry_point_layer_.load(std::memory_order_relaxed) + 1;

        size_t total_degree_layer0 = 0;
        size_t total_degree_upper = 0;
        size_t nodes_in_upper_layers = 0;
        stats.min_degree_layer0 = std::numeric_limits<size_t>::max();
        stats.max_degree_layer0 = 0;

        for (const auto& [id, node] : nodes_) {
            // Layer 0 stats (all nodes exist at layer 0)
            size_t degree0 = node.neighbors(0).size();
            total_degree_layer0 += degree0;
            stats.total_edges += degree0;

            if (degree0 < stats.min_degree_layer0) {
                stats.min_degree_layer0 = degree0;
            }
            if (degree0 > stats.max_degree_layer0) {
                stats.max_degree_layer0 = degree0;
            }
            if (degree0 == 0) {
                stats.orphan_count++;
            }
            if (degree0 < config_.M / 2) {
                stats.underconnected++;
            }

            // Upper layer stats
            size_t node_max_layer = node.num_layers() > 0 ? node.num_layers() - 1 : 0;
            for (size_t layer = 1; layer <= node_max_layer; ++layer) {
                size_t degree = node.neighbors(layer).size();
                total_degree_upper += degree;
                stats.total_edges += degree;
                nodes_in_upper_layers++;
            }
        }

        // Edges are counted twice (bidirectional), so divide by 2
        stats.total_edges /= 2;

        stats.avg_degree_layer0 =
            static_cast<double>(total_degree_layer0) / static_cast<double>(stats.num_nodes);

        if (nodes_in_upper_layers > 0) {
            stats.avg_degree_upper = static_cast<double>(total_degree_upper) /
                                     static_cast<double>(nodes_in_upper_layers);
        }

        // Connectivity score: based on avg degree vs target M and orphan rate
        double degree_ratio = stats.avg_degree_layer0 / static_cast<double>(config_.M_max_0);
        double orphan_ratio =
            1.0 - static_cast<double>(stats.orphan_count) / static_cast<double>(stats.num_nodes);
        double underconnected_ratio =
            1.0 - static_cast<double>(stats.underconnected) / static_cast<double>(stats.num_nodes);

        stats.connectivity_score = std::min({degree_ratio, orphan_ratio, underconnected_ratio});
        stats.connectivity_score = std::clamp(stats.connectivity_score, 0.0, 1.0);

        return stats;
    }

    /// Iterator support for serialization
    auto begin() const { return nodes_.begin(); }
    auto end() const { return nodes_.end(); }

    /// Get node by ID (for serialization)
    const NodeType* get_node(size_t id) const { return try_get_node(id); }

    // ========== Soft Deletion API ==========

    /// Mark node as deleted (soft delete)
    /// Graph edges remain but node is excluded from search results
    /// @param id Node ID to mark as deleted
    void remove(size_t id) {
        std::shared_lock nodes_lock(nodes_mutex_);
        if (nodes_.count(id) == 0)
            return;
        std::unique_lock deleted_lock(deleted_mutex_);
        deleted_ids_.insert(id);
        mutation_generation_.fetch_add(1, std::memory_order_release);
    }

    /// Check if node is soft-deleted
    /// @param id Node ID to check
    /// @return true if node is marked as deleted
    [[nodiscard]] bool is_deleted(size_t id) const {
        std::shared_lock deleted_lock(deleted_mutex_);
        return deleted_ids_.contains(id);
    }

    /// Count of soft-deleted nodes
    [[nodiscard]] size_t deleted_count() const {
        std::shared_lock deleted_lock(deleted_mutex_);
        return deleted_ids_.size();
    }

    /// Count of active (non-deleted) nodes
    [[nodiscard]] size_t active_size() const {
        std::shared_lock nodes_lock(nodes_mutex_);
        std::shared_lock deleted_lock(deleted_mutex_);
        return nodes_.size() - deleted_ids_.size();
    }

    /// Get the set of deleted IDs (for serialization)
    [[nodiscard]] const std::unordered_set<size_t>& deleted_ids() const { return deleted_ids_; }

    /// Monotonically increasing counter, bumped on every insert/delete.
    /// Used by HNSWQuantizedSearch to detect stale quantization snapshots.
    [[nodiscard]] uint64_t mutation_generation() const {
        return mutation_generation_.load(std::memory_order_acquire);
    }

    /// Capture a point-in-time snapshot of all vectors for quantization.
    /// Holds the read lock for the duration of the copy, so the returned
    /// snapshot is guaranteed consistent with the captured generation.
    [[nodiscard]] QuantizationSnapshot snapshot_for_quantization() const {
        std::shared_lock lock(nodes_mutex_);
        QuantizationSnapshot snap;
        snap.generation = mutation_generation_.load(std::memory_order_acquire);
        snap.entries.reserve(nodes_.size());
        for (const auto& [id, node] : nodes_) {
            if (snap.dim == 0)
                snap.dim = node.vector.size();
            QuantizationSnapshot::Entry entry;
            entry.dense_id = node.dense_id;
            if constexpr (std::same_as<StorageT, float>) {
                entry.vector.assign(node.vector.begin(), node.vector.end());
            } else {
                entry.vector = to_float_vector(node.as_span());
            }
            snap.entries.push_back(std::move(entry));
        }
        return snap;
    }

    /// Check if compaction is recommended
    /// @param threshold Fraction of deleted nodes (default 0.2 = 20%)
    /// @return true if deleted_count > threshold * size
    [[nodiscard]] bool needs_compaction(float threshold = 0.2f) const {
        std::shared_lock nodes_lock(nodes_mutex_);
        if (nodes_.empty())
            return false;
        std::shared_lock deleted_lock(deleted_mutex_);
        return static_cast<float>(deleted_ids_.size()) / static_cast<float>(nodes_.size()) >
               threshold;
    }

    /// Rebuild index without deleted nodes
    /// Creates a new index with only active nodes and fresh graph connections
    /// @return New HNSWIndex without deleted nodes
    [[nodiscard]] HNSWIndex compact() const {
        HNSWIndex compacted(config_);

        // Collect active nodes (preserving order for determinism)
        std::vector<size_t> active_ids;
        std::vector<std::span<const StorageT>> active_vectors;
        active_ids.reserve(active_size());
        active_vectors.reserve(active_size());

        for (const auto& [id, node] : nodes_) {
            if (!is_deleted(id)) {
                active_ids.push_back(id);
                active_vectors.push_back(node.as_span());
            }
        }

        // Rebuild with fresh graph (this ensures optimal connectivity)
        for (size_t i = 0; i < active_ids.size(); ++i) {
            compacted.insert(active_ids[i], active_vectors[i]);
        }

        return compacted;
    }

    /// Isolate deleted nodes by removing incoming edges to them
    /// This improves search quality by not traversing to dead-ends
    /// Call after batch deletions but before search
    void isolate_deleted() {
        std::unique_lock nodes_lock(nodes_mutex_);
        std::shared_lock deleted_lock(deleted_mutex_);
        if (deleted_ids_.empty())
            return;

        for (auto& [id, node] : nodes_) {
            if (is_deleted_unlocked(id))
                continue;

            // Remove edges to deleted nodes at each layer
            for (size_t layer = 0; layer < node.edges.size(); ++layer) {
                auto& layer_edges = node.edges[layer];
                std::erase_if(layer_edges,
                              [this](size_t neighbor) { return is_deleted_unlocked(neighbor); });
            }
        }
        mutation_generation_.fetch_add(1, std::memory_order_release);
    }

    /// Restore a soft-deleted node (undo delete)
    /// @param id Node ID to restore
    /// @return true if node was restored, false if not found or not deleted
    bool restore(size_t id) {
        std::shared_lock nodes_lock(nodes_mutex_);
        if (nodes_.count(id) == 0)
            return false;
        std::unique_lock deleted_lock(deleted_mutex_);
        bool erased = deleted_ids_.erase(id) > 0;
        if (erased)
            mutation_generation_.fetch_add(1, std::memory_order_release);
        return erased;
    }

    /// Clear all deletion markers (does not rebuild graph)
    void clear_deletions() {
        std::unique_lock deleted_lock(deleted_mutex_);
        if (!deleted_ids_.empty()) {
            deleted_ids_.clear();
            mutation_generation_.fetch_add(1, std::memory_order_release);
        }
    }

    /// Convert storage vector to float32 (no-op for float, converts for fp16)
    /// @param src Input vector in storage format
    /// @return Vector converted to float32
    std::vector<float> to_float_vector(std::span<const StorageT> src) const {
        if constexpr (std::same_as<StorageT, float>) {
            return std::vector<float>(src.begin(), src.end());
        } else {
            // StorageT is a custom type like float16_t, use utility conversion
            return utils::to_float32(std::span<const utils::float16_t>(
                reinterpret_cast<const utils::float16_t*>(src.data()), src.size()));
        }
    }

private:
    static constexpr size_t kInvalidDenseId = std::numeric_limits<size_t>::max();

    Config config_;

    // Thread-safe graph storage
    mutable std::shared_mutex nodes_mutex_;
    mutable std::shared_mutex deleted_mutex_;
    std::unordered_map<size_t, NodeType> nodes_;
    std::unordered_set<size_t> deleted_ids_;
    std::atomic<uint64_t> mutation_generation_{
        0}; ///< Incremented on insert/delete for staleness detection
    std::atomic<size_t> next_dense_id_{0};

#ifdef SQLITE_VEC_CPP_TESTING
    static inline TestingAfterInsertPublishHook testing_after_insert_publish_hook_{};
#endif

    // Flat lookup table for O(1) node access during single-threaded construction.
    // Indexed by external ID. Falls back to hash lookup if ID is out of range.
    // Only populated by insert_single_threaded() and cleared on concurrent insert().
    std::vector<NodeType*> flat_lookup_;

    // Atomic entry point (thread-safe updates)
    std::atomic<size_t> entry_point_id_{0};
    std::atomic<size_t> entry_point_layer_{0};

    // Per-thread RNG for parallel operations
    ThreadLocalRNG rng_generator_;

    /// Select random layer using thread-local RNG
    size_t random_layer() { return rng_generator_.random_layer(config_.ml_factor); }

    void rebuild_dense_ids_unlocked() {
        size_t dense_id = 0;
        for (auto& [id, node] : nodes_) {
            (void)id;
            node.dense_id = dense_id++;
        }
        next_dense_id_.store(dense_id, std::memory_order_relaxed);
    }

    bool is_deleted_unlocked(size_t id) const { return deleted_ids_.contains(id); }

    /// Register node in flat lookup table (call from insert_single_threaded)
    void register_flat_lookup(size_t id, NodeType* ptr) {
        if (id >= flat_lookup_.size()) {
            flat_lookup_.resize(id + 1, nullptr);
        }
        flat_lookup_[id] = ptr;
    }

    // ========== Unlocked methods for single-threaded insert ==========
    // These access node edges directly (no mutex, no copy) for maximum throughput.
    // Caller MUST ensure no concurrent access.

    template <bool ReadMostly>
    decltype(auto) search_neighbors(const NodeType& node, size_t layer) const {
        if constexpr (ReadMostly) {
            return node.neighbors_unlocked(layer);
        } else {
            return node.neighbors(layer);
        }
    }

    template <bool ReadMostly>
    std::vector<std::pair<size_t, float>> search_with_filter_impl(std::span<const float> query,
                                                                  size_t k, size_t ef_search,
                                                                  const FilterFn& filter) const {
        if (nodes_.empty())
            return {};

        std::vector<float> norm_query;
        std::span<const float> effective_query = query;
        if (config_.normalize_vectors) {
            norm_query = normalize_vector(query);
            effective_query = std::span<const float>(norm_query);
        }

        ef_search = std::max(ef_search, k);

        std::shared_lock lock(nodes_mutex_);
        std::shared_lock deleted_lock(deleted_mutex_);

        const FilterFn* filter_ptr = filter ? &filter : nullptr;

        size_t current = entry_point_id_.load(std::memory_order_acquire);
        for (size_t lc = entry_point_layer_.load(std::memory_order_acquire); lc > 0; --lc) {
            current =
                greedy_search_layer_shared<ReadMostly>(effective_query, current, lc, filter_ptr);
        }

        auto candidates = beam_search_layer_shared<ReadMostly>(effective_query, current, ef_search,
                                                               0, filter_ptr);

        if (candidates.size() > k) {
            candidates.resize(k);
        }
        convert_result_distances(candidates);

        return candidates;
    }

    /// Greedy search without any locking (single-threaded insert path)
    size_t greedy_search_layer_unlocked(std::span<const float> query, size_t entry_point,
                                        size_t layer) const {
        const float kDistanceEpsilon =
            config_.clamp_negative_distances ? -1e-5f : std::numeric_limits<float>::lowest();

        size_t current = entry_point;
        const auto* current_node = try_get_node(current);
        if (!current_node)
            return current;
        float current_dist = comparable_distance_query_node(query, *current_node);

        bool changed = true;
        while (changed) {
            changed = false;
            const auto* node = try_get_node(current);
            if (!node)
                break;

            const auto& neighbors = node->neighbors_unlocked(layer);

            if (!neighbors.empty()) {
                prefetch_node(neighbors[0]);
            }

            for (size_t i = 0; i < neighbors.size(); ++i) {
                size_t neighbor = neighbors[i];
                if (i + 1 < neighbors.size()) {
                    prefetch_node(neighbors[i + 1]);
                }
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                float neighbor_dist = comparable_distance_query_node(query, *neighbor_node);
                if (neighbor_dist < kDistanceEpsilon || neighbor_dist >= current_dist)
                    continue;
                current = neighbor;
                current_dist = neighbor_dist;
                changed = true;
            }
        }
        return current;
    }

    /// Beam search without any locking (single-threaded insert path)
    std::vector<std::pair<size_t, float>> beam_search_layer_unlocked(std::span<const float> query,
                                                                     size_t entry_point, size_t ef,
                                                                     size_t layer) const {
        auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp)>
            top_candidates(cmp);

        auto cmp_min = [](const auto& a, const auto& b) { return a.first > b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp_min)>
            candidates(cmp_min);

        auto& visited = ThreadLocalVisitedPool::get(nodes_.size() + 1);

        const auto* entry_node = try_get_node(entry_point);
        if (!entry_node)
            return {};

        float entry_dist = comparable_distance_query_node(query, *entry_node);
        candidates.emplace(entry_dist, entry_point);
        size_t entry_dense = entry_node->dense_id;
        if (entry_dense == kInvalidDenseId)
            return {};
        visited.visit(entry_dense);

        const float kDistanceEpsilon =
            config_.clamp_negative_distances ? -1e-5f : std::numeric_limits<float>::lowest();
        if (entry_dist >= kDistanceEpsilon) {
            float entry_score =
                config_.clamp_negative_distances ? std::max(0.0f, entry_dist) : entry_dist;
            top_candidates.emplace(entry_score, entry_point);
        }

        while (!candidates.empty()) {
            auto [current_dist, current_id] = candidates.top();
            candidates.pop();

            const auto* current_node = try_get_node(current_id);
            if (!current_node)
                continue;

            if (!top_candidates.empty() && current_dist > top_candidates.top().first &&
                top_candidates.size() >= ef) {
                break;
            }

            const auto& neighbors = current_node->neighbors_unlocked(layer);

            // Prefetch first unvisited neighbor via flat lookup (O(1), no hash)
            for (size_t neighbor : neighbors) {
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                size_t neighbor_dense = neighbor_node->dense_id;
                if (neighbor_dense == kInvalidDenseId)
                    continue;
                if (!visited.is_visited(neighbor_dense)) {
                    prefetch_node(neighbor);
                    break;
                }
            }

            size_t prefetch_idx = 0;
            for (size_t i = 0; i < neighbors.size(); ++i) {
                size_t neighbor = neighbors[i];
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                size_t neighbor_dense = neighbor_node->dense_id;
                if (neighbor_dense == kInvalidDenseId)
                    continue;
                if (visited.is_visited(neighbor_dense))
                    continue;
                visited.visit(neighbor_dense);

                // Prefetch next unvisited neighbor
                for (size_t j = i + 1; j < neighbors.size() && prefetch_idx <= i; ++j) {
                    size_t next_neighbor = neighbors[j];
                    const auto* next_node = try_get_node(next_neighbor);
                    if (!next_node)
                        continue;
                    size_t next_dense = next_node->dense_id;
                    if (next_dense == kInvalidDenseId)
                        continue;
                    if (!visited.is_visited(next_dense)) {
                        prefetch_node(next_neighbor);
                        prefetch_idx = j;
                        break;
                    }
                }

                float neighbor_dist = comparable_distance_query_node(query, *neighbor_node);
                if (neighbor_dist < kDistanceEpsilon)
                    continue;

                bool should_explore = top_candidates.empty() || top_candidates.size() < ef ||
                                      neighbor_dist < top_candidates.top().first;

                if (should_explore) {
                    float scored = config_.clamp_negative_distances ? std::max(0.0f, neighbor_dist)
                                                                    : neighbor_dist;
                    candidates.emplace(scored, neighbor);
                }

                if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                    float scored = config_.clamp_negative_distances ? std::max(0.0f, neighbor_dist)
                                                                    : neighbor_dist;
                    top_candidates.emplace(scored, neighbor);
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        }

        std::vector<std::pair<size_t, float>> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            auto [dist, id] = top_candidates.top();
            top_candidates.pop();
            result.emplace_back(id, dist);
        }

        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        return result;
    }

    /// Connect neighbors without locking (single-threaded insert path)
    void connect_neighbors_unlocked(size_t node_id,
                                    const std::vector<std::pair<size_t, float>>& candidates,
                                    size_t M, size_t layer) {
        size_t num_connections = std::min(M, candidates.size());
        if (num_connections == 0)
            return;

        auto* node_ptr = try_get_node(node_id);
        if (!node_ptr)
            return;

        size_t M_max = (layer == 0) ? config_.M_max_0 : config_.M_max;

        for (size_t i = 0; i < num_connections; ++i) {
            size_t neighbor_id = candidates[i].first;
            auto* neighbor_ptr = try_get_node(neighbor_id);
            if (!neighbor_ptr)
                continue;

            node_ptr->add_edge_unlocked(neighbor_id, layer);
            neighbor_ptr->add_edge_unlocked(node_id, layer);

            // Check neighbor degree and prune if needed
            if (neighbor_ptr->neighbors_unlocked(layer).size() > M_max) {
                prune_connections_unlocked(neighbor_id, layer);
            }
        }
    }

    /// Prune connections without locking (single-threaded insert path)
    void prune_connections_unlocked(size_t node_id, size_t layer) {
        auto* node_ptr = try_get_node(node_id);
        if (!node_ptr)
            return;

        size_t M_max = (layer == 0) ? config_.M_max_0 : config_.M_max;
        auto& node = *node_ptr;
        const auto& neighbors = node.neighbors_unlocked(layer);

        if (neighbors.size() <= M_max)
            return;

        // Collect neighbor distances
        std::vector<std::pair<size_t, float>> neighbor_dists;
        neighbor_dists.reserve(neighbors.size());
        for (size_t neighbor : neighbors) {
            const auto* n = try_get_node(neighbor);
            if (!n)
                continue;
            neighbor_dists.emplace_back(neighbor, comparable_distance_nodes(node, *n));
        }

        std::sort(neighbor_dists.begin(), neighbor_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Heuristic pruning
        std::vector<size_t> selected;
        selected.reserve(M_max);

        for (const auto& [neighbor_id, dist_to_node] : neighbor_dists) {
            if (selected.size() >= M_max)
                break;

            bool is_diverse = true;
            for (size_t sel : selected) {
                const auto* sel_node = try_get_node(sel);
                const auto* nbr_node = try_get_node(neighbor_id);
                if (!sel_node || !nbr_node)
                    continue;
                float dist_to_selected = comparable_distance_nodes(*nbr_node, *sel_node);
                if (dist_to_selected < dist_to_node) {
                    is_diverse = false;
                    break;
                }
            }

            if (is_diverse) {
                selected.push_back(neighbor_id);
            }
        }

        // Fill remaining slots with closest
        if (selected.size() < M_max) {
            for (const auto& [neighbor_id, dist] : neighbor_dists) {
                if (selected.size() >= M_max)
                    break;
                if (std::find(selected.begin(), selected.end(), neighbor_id) == selected.end()) {
                    selected.push_back(neighbor_id);
                }
            }
        }

        // Remove edges not in selected
        std::unordered_set<size_t> selected_set(selected.begin(), selected.end());
        std::vector<size_t> to_remove;
        for (const auto& [neighbor_id, dist] : neighbor_dists) {
            if (!selected_set.contains(neighbor_id)) {
                to_remove.push_back(neighbor_id);
            }
        }

        for (size_t neighbor : to_remove) {
            node.remove_edge_unlocked(neighbor, layer);
            auto* nbr_ptr = try_get_node(neighbor);
            if (nbr_ptr) {
                nbr_ptr->remove_edge_unlocked(node_id, layer);
            }
        }
    }

    // ========== End unlocked methods ==========

    size_t greedy_search_layer_batch(std::span<const float> query, size_t entry_point,
                                     size_t layer) const {
        const float kDistanceEpsilon =
            config_.clamp_negative_distances ? -1e-5f : std::numeric_limits<float>::lowest();

        size_t current = entry_point;
        const auto* current_node = try_get_node(current);
        if (!current_node)
            return current;
        float current_dist = comparable_distance_query_node(query, *current_node);

        auto passes_filter = [&](size_t id) { return !is_deleted_unlocked(id); };
        size_t best_active = passes_filter(current) ? current : static_cast<size_t>(-1);
        float best_active_dist =
            passes_filter(current) ? current_dist : std::numeric_limits<float>::max();

        bool changed = true;
        while (changed) {
            changed = false;
            const auto* node = try_get_node(current);
            if (!node)
                break;

            auto neighbors = node->neighbors(layer);

            if (!neighbors.empty()) {
                prefetch_node(neighbors[0]);
            }

            for (size_t i = 0; i < neighbors.size(); ++i) {
                size_t neighbor = neighbors[i];
                if (i + 1 < neighbors.size()) {
                    prefetch_node(neighbors[i + 1]);
                }

                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;

                float neighbor_dist = comparable_distance_query_node(query, *neighbor_node);
                if (neighbor_dist < kDistanceEpsilon || neighbor_dist >= current_dist)
                    continue;
                current = neighbor;
                current_dist = neighbor_dist;
                changed = true;

                if (passes_filter(neighbor) && neighbor_dist < best_active_dist) {
                    best_active = neighbor;
                    best_active_dist = neighbor_dist;
                }
            }
        }

        return (best_active != static_cast<size_t>(-1)) ? best_active : current;
    }

    std::vector<std::pair<size_t, float>> beam_search_layer_batch(std::span<const float> query,
                                                                  size_t entry_point, size_t ef,
                                                                  size_t layer) const {
        auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp)>
            top_candidates(cmp);

        auto cmp_min = [](const auto& a, const auto& b) { return a.first > b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp_min)>
            candidates(cmp_min);

        auto& visited = ThreadLocalVisitedPool::get(nodes_.size() + 1);

        const auto* entry_node = try_get_node(entry_point);
        if (!entry_node)
            return {};

        float entry_dist = comparable_distance_query_node(query, *entry_node);
        candidates.emplace(entry_dist, entry_point);
        size_t entry_dense = entry_node->dense_id;
        if (entry_dense == kInvalidDenseId) {
            return {};
        }
        visited.visit(entry_dense);

        auto passes_filter = [&](size_t id) { return !is_deleted_unlocked(id); };

        const float kDistanceEpsilon =
            config_.clamp_negative_distances ? -1e-5f : std::numeric_limits<float>::lowest();
        if (passes_filter(entry_point) && entry_dist >= kDistanceEpsilon) {
            float entry_score =
                config_.clamp_negative_distances ? std::max(0.0f, entry_dist) : entry_dist;
            top_candidates.emplace(entry_score, entry_point);
        }

        while (!candidates.empty()) {
            auto [current_dist, current_id] = candidates.top();
            candidates.pop();

            const auto* current_node = try_get_node(current_id);
            if (!current_node)
                continue;

            if (!top_candidates.empty() && current_dist > top_candidates.top().first &&
                top_candidates.size() >= ef) {
                break;
            }

            auto neighbors = current_node->neighbors(layer);

            for (size_t neighbor : neighbors) {
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                size_t neighbor_dense = neighbor_node->dense_id;
                if (neighbor_dense == kInvalidDenseId)
                    continue;
                if (!visited.is_visited(neighbor_dense)) {
                    prefetch_node(neighbor);
                    break;
                }
            }

            size_t prefetch_idx = 0;
            for (size_t i = 0; i < neighbors.size(); ++i) {
                size_t neighbor = neighbors[i];
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                size_t neighbor_dense = neighbor_node->dense_id;
                if (neighbor_dense == kInvalidDenseId)
                    continue;
                if (visited.is_visited(neighbor_dense))
                    continue;
                visited.visit(neighbor_dense);

                for (size_t j = i + 1; j < neighbors.size() && prefetch_idx <= i; ++j) {
                    size_t next_neighbor = neighbors[j];
                    const auto* next_node = try_get_node(next_neighbor);
                    if (!next_node)
                        continue;
                    size_t next_dense = next_node->dense_id;
                    if (next_dense == kInvalidDenseId)
                        continue;
                    if (!visited.is_visited(next_dense)) {
                        prefetch_node(next_neighbor);
                        prefetch_idx = j;
                        break;
                    }
                }

                float neighbor_dist = comparable_distance_query_node(query, *neighbor_node);
                if (neighbor_dist < kDistanceEpsilon)
                    continue;

                bool should_explore = top_candidates.empty() || top_candidates.size() < ef ||
                                      neighbor_dist < top_candidates.top().first;

                if (should_explore) {
                    float scored = config_.clamp_negative_distances ? std::max(0.0f, neighbor_dist)
                                                                    : neighbor_dist;
                    candidates.emplace(scored, neighbor);
                }

                if (passes_filter(neighbor)) {
                    if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                        float scored = config_.clamp_negative_distances
                                           ? std::max(0.0f, neighbor_dist)
                                           : neighbor_dist;
                        top_candidates.emplace(scored, neighbor);
                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        std::vector<std::pair<size_t, float>> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            auto [dist, id] = top_candidates.top();
            top_candidates.pop();
            result.emplace_back(id, dist);
        }

        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        return result;
    }

    void prune_connections_batch(size_t node_id, size_t layer) {
        prune_connections(node_id, layer);
    }

    template <bool ReadMostly>
    size_t greedy_search_layer_shared(std::span<const float> query, size_t entry_point,
                                      size_t layer, const FilterFn* filter) const {
        // Small negative threshold to handle floating-point error in distance calculation
        const float kDistanceEpsilon =
            config_.clamp_negative_distances ? -1e-5f : std::numeric_limits<float>::lowest();

        size_t current = entry_point;
        const auto* initial_node = try_get_node(current);
        if (!initial_node)
            return current;
        float current_dist = comparable_distance_query_node(query, *initial_node);

        const bool check_deleted = !deleted_ids_.empty();
        auto passes_filter = [&](size_t id) {
            if (check_deleted && is_deleted_unlocked(id))
                return false;
            return !filter || (*filter)(id);
        };
        size_t best_active = passes_filter(current) ? current : static_cast<size_t>(-1);
        float best_active_dist =
            passes_filter(current) ? current_dist : std::numeric_limits<float>::max();

        bool changed = true;
        while (changed) {
            changed = false;
            const auto* current_node = try_get_node(current);
            if (!current_node)
                break;

            const auto& neighbors = search_neighbors<ReadMostly>(*current_node, layer);

            // Prefetch first neighbor's vector data
            if (!neighbors.empty()) {
                prefetch_node(neighbors[0]);
            }

            for (size_t i = 0; i < neighbors.size(); ++i) {
                size_t neighbor = neighbors[i];

                // Prefetch next neighbor while computing distance for current
                if (i + 1 < neighbors.size()) {
                    prefetch_node(neighbors[i + 1]);
                }

                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                float neighbor_dist = comparable_distance_query_node(query, *neighbor_node);
                if (neighbor_dist < kDistanceEpsilon || neighbor_dist >= current_dist)
                    continue;
                current = neighbor;
                current_dist = neighbor_dist;
                changed = true;

                if (passes_filter(neighbor) && neighbor_dist < best_active_dist) {
                    best_active = neighbor;
                    best_active_dist = neighbor_dist;
                }
            }
        }

        return (best_active != static_cast<size_t>(-1)) ? best_active : current;
    }

    /// Greedy search (read-only, called under shared lock)
    size_t greedy_search_layer_locked(std::span<const float> query, size_t entry_point,
                                      size_t layer, const FilterFn* filter) const {
        return greedy_search_layer_shared<false>(query, entry_point, layer, filter);
    }

    /// Greedy search optimized for read-mostly workloads under shared lock.
    size_t greedy_search_layer_read_mostly(std::span<const float> query, size_t entry_point,
                                           size_t layer, const FilterFn* filter) const {
        return greedy_search_layer_shared<true>(query, entry_point, layer, filter);
    }

    /// Greedy search (direct access, for insert phase with write lock held)
    size_t greedy_search_layer(std::span<const float> query, size_t entry_point, size_t layer,
                               const FilterFn* filter) const {
        return greedy_search_layer_locked(query, entry_point, layer, filter);
    }

    template <bool ReadMostly>
    std::vector<std::pair<size_t, float>>
    beam_search_layer_shared(std::span<const float> query, size_t entry_point, size_t ef,
                             size_t layer, const FilterFn* filter) const {
        auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
        std::vector<std::pair<float, size_t>> top_storage;
        top_storage.reserve(ef + 1);
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp)>
            top_candidates(cmp, std::move(top_storage));

        auto cmp_min = [](const auto& a, const auto& b) { return a.first > b.first; };
        std::vector<std::pair<float, size_t>> candidate_storage;
        candidate_storage.reserve(ef + 1);
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp_min)>
            candidates(cmp_min, std::move(candidate_storage));

        // Use thread-local visited tracker instead of allocating unordered_set
        // This avoids heap allocation per search call
        auto& visited = ThreadLocalVisitedPool::get(nodes_.size() + 1);

        const auto* entry_node = try_get_node(entry_point);
        if (!entry_node)
            return {};

        float entry_dist = comparable_distance_query_node(query, *entry_node);
        candidates.emplace(entry_dist, entry_point);
        size_t entry_dense = entry_node->dense_id;
        if (entry_dense == kInvalidDenseId) {
            return {};
        }
        visited.visit(entry_dense);

        const bool check_deleted = !deleted_ids_.empty();
        auto passes_filter = [&](size_t id) {
            if (check_deleted && is_deleted_unlocked(id))
                return false;
            return !filter || (*filter)(id);
        };

        // Note: entry_dist can be slightly negative due to floating-point error
        // (e.g., cosine distance of identical vectors = 1 - 1.0000001 = -1e-7)
        // We allow small negative values to avoid missing exact matches.
        const float kDistanceEpsilon =
            config_.clamp_negative_distances ? -1e-5f : std::numeric_limits<float>::lowest();
        if (passes_filter(entry_point) && entry_dist >= kDistanceEpsilon) {
            float entry_score =
                config_.clamp_negative_distances ? std::max(0.0f, entry_dist) : entry_dist;
            top_candidates.emplace(entry_score, entry_point);
        }

        while (!candidates.empty()) {
            auto [current_dist, current_id] = candidates.top();
            candidates.pop();

            const auto* current_node = try_get_node(current_id);
            if (!current_node)
                continue;

            if (!top_candidates.empty() && current_dist > top_candidates.top().first &&
                top_candidates.size() >= ef) {
                break;
            }

            // Get neighbors and filter out already-visited ones
            const auto& neighbors = search_neighbors<ReadMostly>(*current_node, layer);

            // Prefetch first unvisited neighbor's vector data before entering loop
            // This hides memory latency for the first distance calculation
            for (size_t neighbor : neighbors) {
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                size_t neighbor_dense = neighbor_node->dense_id;
                if (neighbor_dense == kInvalidDenseId)
                    continue;
                if (!visited.is_visited(neighbor_dense)) {
                    prefetch_node(neighbor);
                    break;
                }
            }

            size_t prefetch_idx = 0;
            for (size_t i = 0; i < neighbors.size(); ++i) {
                size_t neighbor = neighbors[i];
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                size_t neighbor_dense = neighbor_node->dense_id;
                if (neighbor_dense == kInvalidDenseId)
                    continue;
                if (visited.is_visited(neighbor_dense))
                    continue;
                visited.visit(neighbor_dense);

                // Prefetch next unvisited neighbor while we compute distance for current
                // Look ahead to find the next neighbor that hasn't been visited
                for (size_t j = i + 1; j < neighbors.size() && prefetch_idx <= i; ++j) {
                    size_t next_neighbor = neighbors[j];
                    const auto* next_node = try_get_node(next_neighbor);
                    if (!next_node)
                        continue;
                    size_t next_dense = next_node->dense_id;
                    if (next_dense == kInvalidDenseId)
                        continue;
                    if (!visited.is_visited(next_dense)) {
                        prefetch_node(next_neighbor);
                        prefetch_idx = j;
                        break;
                    }
                }

                float neighbor_dist = comparable_distance_query_node(query, *neighbor_node);
                if (neighbor_dist < kDistanceEpsilon)
                    continue;

                bool should_explore = top_candidates.empty() || top_candidates.size() < ef ||
                                      neighbor_dist < top_candidates.top().first;

                if (should_explore) {
                    float candidate_score = config_.clamp_negative_distances
                                                ? std::max(0.0f, neighbor_dist)
                                                : neighbor_dist;
                    candidates.emplace(candidate_score, neighbor);
                }

                if (passes_filter(neighbor)) {
                    if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                        float top_score = config_.clamp_negative_distances
                                              ? std::max(0.0f, neighbor_dist)
                                              : neighbor_dist;
                        top_candidates.emplace(top_score, neighbor);
                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        std::vector<std::pair<size_t, float>> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            auto [dist, id] = top_candidates.top();
            top_candidates.pop();
            // Distances are already clamped to >= 0 when added
            result.emplace_back(id, dist);
        }

        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        return result;
    }

    /// Beam search (read-only, called under shared lock)
    std::vector<std::pair<size_t, float>> beam_search_layer_locked(std::span<const float> query,
                                                                   size_t entry_point, size_t ef,
                                                                   size_t layer,
                                                                   const FilterFn* filter) const {
        return beam_search_layer_shared<false>(query, entry_point, ef, layer, filter);
    }

    /// Beam search optimized for read-mostly workloads under shared lock.
    std::vector<std::pair<size_t, float>>
    beam_search_layer_read_mostly(std::span<const float> query, size_t entry_point, size_t ef,
                                  size_t layer, const FilterFn* filter) const {
        return beam_search_layer_shared<true>(query, entry_point, ef, layer, filter);
    }

    /// Beam search (direct access, for insert phase)
    std::vector<std::pair<size_t, float>> beam_search_layer(std::span<const float> query,
                                                            size_t entry_point, size_t ef,
                                                            size_t layer,
                                                            const FilterFn* filter) const {
        return beam_search_layer_locked(query, entry_point, ef, layer, filter);
    }

    // ========== Quantized Two-Stage Search (private, friend-accessible) ==========

    /// Two-stage search: quantized distances for graph traversal, exact FP32 for reranking.
    /// Only available for L2-family metrics. Called by HNSWQuantizedSearch.
    template <typename ApproxDistFn, typename PrefetchFn>
    requires(concepts::traits::is_l2_family_v<MetricT>)
    std::vector<std::pair<size_t, float>>
    search_quantized_rerank(std::span<const float> query, size_t k, size_t ef_search,
                            size_t rerank_factor, ApproxDistFn&& approx_dist,
                            PrefetchFn&& prefetch_fn, const FilterFn& filter = nullptr) const {
        if (nodes_.empty())
            return {};

        std::vector<float> norm_query;
        std::span<const float> effective_query = query;
        if (config_.normalize_vectors) {
            norm_query = normalize_vector(query);
            effective_query = std::span<const float>(norm_query);
        }

        size_t expanded_ef = ef_search * rerank_factor;
        expanded_ef = std::max(expanded_ef, k);

        std::shared_lock lock(nodes_mutex_);
        std::shared_lock deleted_lock(deleted_mutex_);

        const FilterFn* filter_ptr = filter ? &filter : nullptr;

        // Stage 1: Greedy descent through upper layers (cheap, use FP32)
        size_t current = entry_point_id_.load(std::memory_order_acquire);
        for (size_t lc = entry_point_layer_.load(std::memory_order_acquire); lc > 0; --lc) {
            current = greedy_search_layer_shared<true>(effective_query, current, lc, filter_ptr);
        }

        // Stage 2: Beam search at layer 0 using quantized distances
        auto approx_candidates = beam_search_layer_quantized(
            effective_query, current, expanded_ef, 0, filter_ptr, approx_dist, prefetch_fn);

        // Stage 3: Rerank with exact FP32 distances
        std::vector<std::pair<size_t, float>> reranked;
        reranked.reserve(approx_candidates.size());

        for (const auto& [id, approx_d] : approx_candidates) {
            const auto* node = try_get_node(id);
            if (!node)
                continue;
            float exact_dist = distance_query_node(effective_query, *node);
            reranked.emplace_back(id, exact_dist);
        }

        std::sort(reranked.begin(), reranked.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        if (reranked.size() > k) {
            reranked.resize(k);
        }

        return reranked;
    }

    /// Beam search using a user-supplied approximate distance function.
    /// Runs inside the HNSW internals with visited pool, prefetching, and
    /// read-mostly unlocked neighbor access — the same hot path as standard search.
    ///
    /// @tparam ApproxDistFn float(std::span<const float> query, size_t dense_id)
    /// @tparam PrefetchFn void(size_t dense_id)
    template <typename ApproxDistFn, typename PrefetchFn>
    requires(concepts::traits::is_l2_family_v<MetricT>)
    std::vector<std::pair<size_t, float>>
    beam_search_layer_quantized(std::span<const float> query, size_t entry_point, size_t ef,
                                size_t layer, const FilterFn* filter, ApproxDistFn&& approx_dist,
                                PrefetchFn&& prefetch_fn) const {
        auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp)>
            top_candidates(cmp);

        auto cmp_min = [](const auto& a, const auto& b) { return a.first > b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp_min)>
            candidates(cmp_min);

        auto& visited = ThreadLocalVisitedPool::get(nodes_.size() + 1);

        const auto* entry_node = try_get_node(entry_point);
        if (!entry_node)
            return {};

        size_t entry_dense = entry_node->dense_id;
        if (entry_dense == kInvalidDenseId)
            return {};

        float entry_dist = approx_dist(query, entry_dense);
        candidates.emplace(entry_dist, entry_point);
        visited.visit(entry_dense);

        auto passes_filter = [&](size_t id) {
            return !is_deleted_unlocked(id) && (!filter || (*filter)(id));
        };

        const float kDistanceEpsilon =
            config_.clamp_negative_distances ? -1e-5f : std::numeric_limits<float>::lowest();

        if (passes_filter(entry_point) && entry_dist >= kDistanceEpsilon) {
            float score =
                config_.clamp_negative_distances ? std::max(0.0f, entry_dist) : entry_dist;
            top_candidates.emplace(score, entry_point);
        }

        while (!candidates.empty()) {
            auto [current_dist, current_id] = candidates.top();
            candidates.pop();

            const auto* current_node = try_get_node(current_id);
            if (!current_node)
                continue;

            if (!top_candidates.empty() && current_dist > top_candidates.top().first &&
                top_candidates.size() >= ef) {
                break;
            }

            // Read-mostly: direct reference, no mutex, no copy
            const auto& neighbors = current_node->neighbors_unlocked(layer);

            // Prefetch first unvisited neighbor's quantized codes
            for (size_t neighbor : neighbors) {
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                size_t neighbor_dense = neighbor_node->dense_id;
                if (neighbor_dense == kInvalidDenseId)
                    continue;
                if (!visited.is_visited(neighbor_dense)) {
                    prefetch_fn(neighbor_dense);
                    break;
                }
            }

            size_t prefetch_idx = 0;
            for (size_t i = 0; i < neighbors.size(); ++i) {
                size_t neighbor = neighbors[i];
                const auto* neighbor_node = try_get_node(neighbor);
                if (!neighbor_node)
                    continue;
                size_t neighbor_dense = neighbor_node->dense_id;
                if (neighbor_dense == kInvalidDenseId)
                    continue;
                if (visited.is_visited(neighbor_dense))
                    continue;
                visited.visit(neighbor_dense);

                // Prefetch next unvisited neighbor's quantized codes
                for (size_t j = i + 1; j < neighbors.size() && prefetch_idx <= i; ++j) {
                    const auto* next_node = try_get_node(neighbors[j]);
                    if (!next_node)
                        continue;
                    size_t next_dense = next_node->dense_id;
                    if (next_dense == kInvalidDenseId)
                        continue;
                    if (!visited.is_visited(next_dense)) {
                        prefetch_fn(next_dense);
                        prefetch_idx = j;
                        break;
                    }
                }

                float neighbor_dist = approx_dist(query, neighbor_dense);
                if (neighbor_dist < kDistanceEpsilon)
                    continue;

                bool should_explore = top_candidates.empty() || top_candidates.size() < ef ||
                                      neighbor_dist < top_candidates.top().first;

                if (should_explore) {
                    float score = config_.clamp_negative_distances ? std::max(0.0f, neighbor_dist)
                                                                   : neighbor_dist;
                    candidates.emplace(score, neighbor);
                }

                if (passes_filter(neighbor)) {
                    if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                        float top_score = config_.clamp_negative_distances
                                              ? std::max(0.0f, neighbor_dist)
                                              : neighbor_dist;
                        top_candidates.emplace(top_score, neighbor);
                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        std::vector<std::pair<size_t, float>> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            auto [dist, id] = top_candidates.top();
            top_candidates.pop();
            result.emplace_back(id, dist);
        }

        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        return result;
    }

    /// Connect node to M nearest neighbors at layer
    void connect_neighbors(size_t node_id, const std::vector<std::pair<size_t, float>>& candidates,
                           size_t M, size_t layer) {
        size_t num_connections = std::min(M, candidates.size());
        if (num_connections == 0)
            return;

        // Cache the node lookup - it won't change during this function
        auto* node_ptr = try_get_node(node_id);
        if (!node_ptr)
            return;

        size_t M_max = (layer == 0) ? config_.M_max_0 : config_.M_max;

        for (size_t i = 0; i < num_connections; ++i) {
            size_t neighbor_id = candidates[i].first;

            // Verify neighbor exists (may not during concurrent insertion)
            auto* neighbor_ptr = try_get_node(neighbor_id);
            if (!neighbor_ptr)
                continue;

            node_ptr->add_edge(neighbor_id, layer);
            neighbor_ptr->add_edge(node_id, layer);

            if (neighbor_ptr->neighbors(layer).size() > M_max) {
                prune_connections(neighbor_id, layer);
            }
        }
    }

    /// Prune connections to maintain M_max limit
    /// Uses heuristic pruning (HNSW paper algorithm 4) for better graph diversity
    void prune_connections(size_t node_id, size_t layer) {
        auto* node_ptr = try_get_node(node_id);
        if (!node_ptr)
            return;

        size_t M_max = (layer == 0) ? config_.M_max_0 : config_.M_max;
        auto& node = *node_ptr;
        auto neighbors = node.neighbors(layer);

        if (neighbors.size() <= M_max)
            return;

        // Collect neighbor distances and sort by distance
        std::vector<std::pair<size_t, float>> neighbor_dists;
        neighbor_dists.reserve(neighbors.size());
        for (size_t neighbor : neighbors) {
            if (!try_get_node(neighbor))
                continue;
            neighbor_dists.emplace_back(neighbor, distance(node_id, neighbor));
        }

        std::sort(neighbor_dists.begin(), neighbor_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Heuristic pruning: select diverse neighbors
        // Keep neighbor if it's closer to node than to any already-selected neighbor
        // This promotes graph diversity and improves recall for clustered data
        std::vector<size_t> selected;
        selected.reserve(M_max);

        for (const auto& [neighbor_id, dist_to_node] : neighbor_dists) {
            if (selected.size() >= M_max)
                break;

            // Check if this neighbor is "diverse" - not too close to already selected
            bool is_diverse = true;
            for (size_t sel : selected) {
                float dist_to_selected = distance(neighbor_id, sel);
                // If neighbor is closer to an already-selected node than to our node,
                // it's redundant (the selected node can reach it)
                if (dist_to_selected < dist_to_node) {
                    is_diverse = false;
                    break;
                }
            }

            if (is_diverse) {
                selected.push_back(neighbor_id);
            }
        }

        // If heuristic pruning left us short, add closest remaining
        // (this ensures we always have M_max connections if possible)
        if (selected.size() < M_max) {
            for (const auto& [neighbor_id, dist] : neighbor_dists) {
                if (selected.size() >= M_max)
                    break;
                if (std::find(selected.begin(), selected.end(), neighbor_id) == selected.end()) {
                    selected.push_back(neighbor_id);
                }
            }
        }

        // Remove edges not in selected set
        std::unordered_set<size_t> selected_set(selected.begin(), selected.end());
        std::vector<size_t> to_remove;
        for (const auto& [neighbor_id, dist] : neighbor_dists) {
            if (!selected_set.contains(neighbor_id)) {
                to_remove.push_back(neighbor_id);
            }
        }

        for (size_t neighbor : to_remove) {
            node.remove_edge(neighbor, layer);
            auto* neighbor_ptr = try_get_node(neighbor);
            if (neighbor_ptr) {
                neighbor_ptr->remove_edge(node_id, layer);
            }
        }
    }

    /// Calculate distance between two nodes (returns -1 if node not found)
    float distance(size_t id1, size_t id2) const {
        const auto* n1 = try_get_node(id1);
        const auto* n2 = try_get_node(id2);
        if (!n1 || !n2)
            return -1.0f;
        return distance_nodes(*n1, *n2);
    }

    /// Calculate distance between float32 query and node (returns -1 if node not found)
    float distance(std::span<const float> query, size_t node_id) const {
        const auto* node = try_get_node(node_id);
        if (!node)
            return -1.0f;
        return distance_query_node(query, *node);
    }

    /// Try to get node by ID (returns nullptr if not found, safe for concurrent access).
    /// Uses flat lookup table when populated (O(1)), falls back to hash map.
    const NodeType* try_get_node(size_t id) const {
        if (id < flat_lookup_.size()) {
            return flat_lookup_[id];
        }
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    /// Mutable overload of try_get_node.
    NodeType* try_get_node(size_t id) {
        if (id < flat_lookup_.size()) {
            return flat_lookup_[id];
        }
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    /// Prefetch node's vector data into CPU cache for upcoming distance calculation.
    /// Call this 1-2 iterations before you need the data to hide memory latency.
    /// @param id Node ID to prefetch
    void prefetch_node(size_t id) const {
        const NodeType* node = try_get_node(id);
        if (node && !node->vector.empty()) {
            HNSW_PREFETCH_READ(node->vector.data());
            constexpr size_t kCacheLineSize = 64;
            constexpr size_t kElementsPerCacheLine = kCacheLineSize / sizeof(StorageT);
            if (node->vector.size() > kElementsPerCacheLine) {
                HNSW_PREFETCH_READ(node->vector.data() + kElementsPerCacheLine);
            }
            if (node->vector.size() > 2 * kElementsPerCacheLine) {
                HNSW_PREFETCH_READ(node->vector.data() + 2 * kElementsPerCacheLine);
            }
        }
    }

    // ========== Vector Normalization ==========

    /// L2-normalize a float vector in-place. Returns the original norm.
    static float normalize_vector_inplace(std::span<float> vec) {
        float norm_sq = 0.0f;
        for (float v : vec)
            norm_sq += v * v;
        float norm = std::sqrt(norm_sq);
        if (norm > 1e-8f) {
            float inv_norm = 1.0f / norm;
            for (float& v : vec)
                v *= inv_norm;
        }
        return norm;
    }

    /// L2-normalize a float vector, returning normalized copy.
    static std::vector<float> normalize_vector(std::span<const float> vec) {
        std::vector<float> out(vec.begin(), vec.end());
        normalize_vector_inplace(out);
        return out;
    }

    // ========== Distance Dispatch ==========

    /// Compute distance between two float spans.
    /// When normalize_vectors is enabled, uses inner product (1 - dot) for ~3x speedup.
    /// The branch is always predicted correctly (same direction every call).
    float compute_distance(std::span<const float> a, std::span<const float> b) const {
        if (config_.normalize_vectors) {
            return distances::inner_product_distance(a, b);
        }
        return config_.metric(a, b);
    }

    float compute_comparable_distance(std::span<const float> a, std::span<const float> b) const {
        if (config_.normalize_vectors) {
            return distances::inner_product_distance(a, b);
        }
        if constexpr (std::same_as<StorageT, float> &&
                      std::same_as<MetricT, distances::L2Metric<float>>) {
            return distances::l2_squared_distance(a, b);
        } else {
            return config_.metric(a, b);
        }
    }

    float output_distance(float distance) const {
        if (!config_.normalize_vectors) {
            if constexpr (std::same_as<StorageT, float> &&
                          std::same_as<MetricT, distances::L2Metric<float>>) {
                return std::sqrt(std::max(0.0f, distance));
            }
        }
        return distance;
    }

    void convert_result_distances(std::vector<std::pair<size_t, float>>& results) const {
        for (auto& [id, distance] : results) {
            (void)id;
            distance = output_distance(distance);
        }
    }

    /// Distance between two StorageT nodes
    float distance_nodes(const NodeType& n1, const NodeType& n2) const {
        if constexpr (std::same_as<StorageT, float>) {
            return compute_distance(n1.as_span(), n2.as_span());
        } else {
            auto f1 = n1.as_float32();
            auto f2 = n2.as_float32();
            return compute_distance(std::span<const float>(f1), std::span<const float>(f2));
        }
    }

    /// Distance between float32 query and StorageT node
    float distance_query_node(std::span<const float> query, const NodeType& node) const {
        if constexpr (std::same_as<StorageT, float>) {
            return compute_distance(query, node.as_span());
        } else {
            auto node_vec = node.as_float32();
            return compute_distance(query, std::span<const float>(node_vec));
        }
    }

    float comparable_distance_nodes(const NodeType& n1, const NodeType& n2) const {
        if constexpr (std::same_as<StorageT, float>) {
            return compute_comparable_distance(n1.as_span(), n2.as_span());
        } else {
            auto f1 = n1.as_float32();
            auto f2 = n2.as_float32();
            return compute_comparable_distance(std::span<const float>(f1),
                                               std::span<const float>(f2));
        }
    }

    float comparable_distance_query_node(std::span<const float> query, const NodeType& node) const {
        if constexpr (std::same_as<StorageT, float>) {
            return compute_comparable_distance(query, node.as_span());
        } else {
            auto node_vec = node.as_float32();
            return compute_comparable_distance(query, std::span<const float>(node_vec));
        }
    }

    /// Helper for read-locked operations
    template <typename Func> auto read_locked(Func&& func) const -> decltype(func()) {
        std::shared_lock lock(nodes_mutex_);
        return func();
    }

    /// Helper for write-locked operations
    template <typename Func> auto write_locked(Func&& func) -> decltype(func()) {
        std::unique_lock lock(nodes_mutex_);
        return func();
    }
};

} // namespace sqlite_vec_cpp::index
