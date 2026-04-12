#pragma once

/// Two-stage quantized HNSW search
///
/// Wraps an existing HNSWIndex and adds a quantized distance layer for fast
/// graph traversal. The full pipeline:
///
///   1. Graph traversal uses quantized distances (LVQ-8, LVQ-4, or RaBitQ)
///   2. Candidate set (top ef_search results) reranked with exact FP32 distances
///   3. Final top-K returned with exact distances
///
/// This is the architecture used by production systems (Intel SVS, Pinecone).
/// Key benefit: quantized distances are ~2-5x faster to compute and fit more
/// vectors in cache, while reranking recovers any recall loss.
///
/// Implementation: delegates to HNSWIndex::search_quantized_rerank which runs
/// inside the HNSW internals with visited pool, prefetching, and lock-free
/// neighbor access — no external beam search or hash map lookups.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <vector>

#include "../quantization/store.hpp"
#include "hnsw.hpp"

namespace sqlite_vec_cpp::index {

/// Quantization strategy selection
enum class QuantizationType {
    None,   ///< No quantization (baseline)
    LVQ8,   ///< 8-bit locally-adaptive (4x compression)
    LVQ4,   ///< 4-bit locally-adaptive (8x compression)
    RaBitQ, ///< Binary quantization (32x compression)
};

/// Two-stage quantized search wrapper for HNSWIndex
///
/// Stores quantized representations in flat, dense_id-indexed arrays
/// for O(1) cache-friendly access during beam search.
///
/// @tparam StorageT Base storage type (float or float16_t)
/// @tparam MetricT Distance metric type
template <concepts::VectorElement StorageT, typename MetricT> class HNSWQuantizedSearch {
    static_assert(concepts::traits::is_l2_family_v<MetricT>,
                  "HNSWQuantizedSearch requires an L2-family metric. "
                  "Quantized stores only implement L2 distance estimation.");

public:
    using BaseIndex = HNSWIndex<StorageT, MetricT>;
    using FilterFn = typename BaseIndex::FilterFn;

    struct Config {
        QuantizationType quantization = QuantizationType::LVQ8;
        size_t rerank_factor = 2; ///< Multiplier: fetch rerank_factor * ef_search candidates
                                  ///< from quantized search, then rerank to ef_search
    };

    /// Construct with reference to existing HNSW index
    /// The base index is NOT owned -- caller must ensure it outlives this object
    HNSWQuantizedSearch(BaseIndex& base_index, Config config = {})
        : base_index_(base_index), config_(config) {}

    /// Build quantized representations from the current index contents.
    /// Must be called after all vectors are inserted into the base index.
    void build_quantization() {
        lvq8_store_ = {};
        lvq4_store_ = {};
        rabitq_store_ = {};

        if (base_index_.empty())
            return;

        // Capture a consistent snapshot under the index read lock.
        // Stores build from the snapshot, never iterating the live graph.
        auto snap = base_index_.snapshot_for_quantization();

        switch (config_.quantization) {
            case QuantizationType::LVQ8:
                lvq8_store_.build(snap);
                break;
            case QuantizationType::LVQ4:
                lvq4_store_.build(snap);
                break;
            case QuantizationType::RaBitQ:
                rabitq_store_.build(snap);
                break;
            case QuantizationType::None:
                break;
        }

        quantization_generation_ = snap.generation;
    }

    /// Two-stage search: quantized traversal + FP32 reranking
    ///
    /// Delegates to HNSWIndex::search_quantized_rerank which uses the internal
    /// beam search loop with visited pool, prefetching, and lock-free neighbors.
    std::vector<std::pair<size_t, float>> search(std::span<const float> query, size_t k,
                                                 size_t ef_search = 50) const {
        return search_with_filter(query, k, ef_search, nullptr);
    }

    /// Two-stage search with pre-filtering
    std::vector<std::pair<size_t, float>> search_with_filter(std::span<const float> query, size_t k,
                                                             size_t ef_search,
                                                             const FilterFn& filter) const {
        // If no quantization built or index was mutated since build, fall back to base index
        if (config_.quantization == QuantizationType::None || !has_quantization() || is_stale()) {
            return filter ? base_index_.search_with_filter(query, k, ef_search, filter)
                          : base_index_.search(query, k, ef_search);
        }

        switch (config_.quantization) {
            case QuantizationType::LVQ8:
                return search_with_store(lvq8_store_, query, k, ef_search, filter);
            case QuantizationType::LVQ4:
                return search_with_store(lvq4_store_, query, k, ef_search, filter);
            case QuantizationType::RaBitQ:
                return search_rabitq(query, k, ef_search, filter);
            default:
                return base_index_.search(query, k, ef_search);
        }
    }

    /// Check if quantization data is available
    [[nodiscard]] bool has_quantization() const {
        switch (config_.quantization) {
            case QuantizationType::LVQ8:
                return lvq8_store_.count > 0;
            case QuantizationType::LVQ4:
                return lvq4_store_.count > 0;
            case QuantizationType::RaBitQ:
                return rabitq_store_.count > 0;
            case QuantizationType::None:
                return false;
        }
        return false;
    }

    /// Get the quantization type in use
    [[nodiscard]] QuantizationType quantization_type() const { return config_.quantization; }

    /// Check if the base index was mutated after build_quantization().
    /// When stale, search falls back to exact (slower but correct).
    [[nodiscard]] bool is_stale() const {
        return has_quantization() && base_index_.mutation_generation() != quantization_generation_;
    }

    /// Get memory usage of quantized data (approximate bytes)
    [[nodiscard]] size_t quantized_memory_bytes() const {
        switch (config_.quantization) {
            case QuantizationType::LVQ8:
                return lvq8_store_.memory_bytes();
            case QuantizationType::LVQ4:
                return lvq4_store_.memory_bytes();
            case QuantizationType::RaBitQ:
                return rabitq_store_.memory_bytes();
            case QuantizationType::None:
                return 0;
        }
        return 0;
    }

    /// Access the underlying base index
    BaseIndex& base_index() { return base_index_; }
    const BaseIndex& base_index() const { return base_index_; }

private:
    BaseIndex& base_index_;
    Config config_;
    uint64_t quantization_generation_ = 0; ///< Snapshot of base_index_.mutation_generation()

    // Flat, dense_id-indexed stores (only one is populated based on config)
    quantization::LVQ8Store lvq8_store_;
    quantization::LVQ4Store lvq4_store_;
    quantization::RaBitQStore rabitq_store_;

    /// Search using an LVQ store (LVQ-8 or LVQ-4)
    /// The store provides l2_distance(query, dense_id) and prefetch(dense_id)
    template <typename StoreT>
    std::vector<std::pair<size_t, float>>
    search_with_store(const StoreT& store, std::span<const float> query, size_t k, size_t ef_search,
                      const FilterFn& filter) const {
        auto approx_dist = [&store](std::span<const float> q, size_t dense_id) -> float {
            return store.l2_distance(q, dense_id);
        };
        auto prefetch_fn = [&store](size_t dense_id) { store.prefetch(dense_id); };

        return base_index_.search_quantized_rerank(query, k, ef_search, config_.rerank_factor,
                                                   approx_dist, prefetch_fn, filter);
    }

    /// Search using RaBitQ (needs precomputed query state)
    std::vector<std::pair<size_t, float>> search_rabitq(std::span<const float> query, size_t k,
                                                        size_t ef_search,
                                                        const FilterFn& filter) const {
        auto query_state = rabitq_store_.prepare_query(query);

        auto approx_dist = [this, &query_state](std::span<const float> /*q*/,
                                                size_t dense_id) -> float {
            return rabitq_store_.l2_distance(query_state, dense_id);
        };
        auto prefetch_fn = [this](size_t dense_id) { rabitq_store_.prefetch(dense_id); };

        return base_index_.search_quantized_rerank(query, k, ef_search, config_.rerank_factor,
                                                   approx_dist, prefetch_fn, filter);
    }
};

} // namespace sqlite_vec_cpp::index
