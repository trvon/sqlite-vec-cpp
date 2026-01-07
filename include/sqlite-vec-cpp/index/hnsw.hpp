#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
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
#include "hnsw_node.hpp"
#include "hnsw_threading.hpp"

namespace sqlite_vec_cpp::index {

/// Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search
/// Provides 100-1000x speedup over brute-force for large corpora (>100K vectors)
///
/// @tparam StorageT Storage type for vectors (float or float16_t)
/// @tparam MetricT Distance metric type (must work with float spans)
template <concepts::VectorElement StorageT, typename MetricT> class HNSWIndex {
public:
    /// Configuration parameters
    struct Config {
        size_t M = 16;                           ///< Connections per node per layer
        size_t M_max = 32;                       ///< Max connections (layer > 0)
        size_t M_max_0 = 32;                     ///< Max connections (layer 0)
        size_t ef_construction = 200;            ///< Exploration factor during construction
        float ml_factor = 1.0f / std::log(2.0f); ///< Layer selection multiplier
        MetricT metric{};                        ///< Distance metric (operates on float spans)
    };

    /// Alias for node type
    using NodeType = HNSWNode<StorageT>;

    /// Construct empty HNSW index
    explicit HNSWIndex(Config config = {}) : config_(config) {}

    /// Move constructor (required because std::shared_mutex is move-only)
    HNSWIndex(HNSWIndex&& other) noexcept
        : config_(other.config_), nodes_mutex_(), nodes_(std::move(other.nodes_)),
          deleted_ids_(std::move(other.deleted_ids_)),
          entry_point_id_(other.entry_point_id_.load(std::memory_order_relaxed)),
          entry_point_layer_(other.entry_point_layer_.load(std::memory_order_relaxed)),
          rng_generator_() {}

    /// Move assignment
    HNSWIndex& operator=(HNSWIndex&& other) noexcept {
        if (this != &other) {
            config_ = other.config_;
            nodes_ = std::move(other.nodes_);
            deleted_ids_ = std::move(other.deleted_ids_);
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
        return index;
    }

    /// Insert vector into index (storage type StorageT, converts to float32 for graph operations)
    void insert(size_t id, std::span<const StorageT> vector) {
        // Convert to float32 for graph search operations
        std::vector<float> vector_f32 = to_float_vector(vector);

        // Assign random layer using thread-local RNG
        size_t layer = random_layer();

        // Create and insert node with write lock
        {
            std::unique_lock lock(nodes_mutex_);
            nodes_.emplace(id, NodeType(id, vector, layer));

            // First node becomes entry point
            if (nodes_.size() == 1) {
                entry_point_id_.store(id, std::memory_order_relaxed);
                entry_point_layer_.store(layer, std::memory_order_relaxed);
                return;
            }

            // Update entry point if new node has higher layer
            if (layer > entry_point_layer_.load(std::memory_order_relaxed)) {
                size_t old_entry = entry_point_id_.load(std::memory_order_relaxed);
                size_t old_layer = entry_point_layer_.load(std::memory_order_relaxed);
                entry_point_id_.store(id, std::memory_order_relaxed);
                entry_point_layer_.store(layer, std::memory_order_relaxed);

                size_t current = old_entry;
                for (size_t lc = old_layer;; --lc) {
                    auto candidates = beam_search_layer_locked(
                        vector_f32, current, config_.ef_construction, lc, nullptr);

                    size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
                    connect_neighbors(id, candidates, M, lc);

                    if (lc == 0)
                        break;
                    if (!candidates.empty()) {
                        current = candidates[0].first;
                    }
                }
                return;
            }
        } // Release write lock before search (search uses read lock)

        // Navigate from entry point to find insertion point at each layer
        size_t current = entry_point_id_.load(std::memory_order_acquire);

        for (size_t lc = entry_point_layer_.load(std::memory_order_acquire); lc > layer; --lc) {
            current = greedy_search_layer(vector_f32, current, lc, nullptr);
        }

        {
            std::unique_lock lock(nodes_mutex_);
            for (size_t lc = layer;; --lc) {
                auto candidates = beam_search_layer_locked(vector_f32, current,
                                                           config_.ef_construction, lc, nullptr);

                size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
                connect_neighbors(id, candidates, M, lc);

                if (lc == 0)
                    break;
                if (!candidates.empty()) {
                    current = candidates[0].first;
                }
            }
        }
    }

    /// Search for k nearest neighbors (query is always float32)
    /// @param query Query vector (float32)
    /// @param k Number of neighbors to return
    /// @param ef_search Exploration factor (higher = better recall, slower)
    /// @return Vector of (id, distance) pairs, sorted by distance
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
    using FilterFn = std::function<bool(size_t node_id)>;
    std::vector<std::pair<size_t, float>> search_with_filter(std::span<const float> query, size_t k,
                                                             size_t ef_search,
                                                             FilterFn filter) const {
        if (nodes_.empty())
            return {};

        ef_search = std::max(ef_search, k);

        // Acquire shared lock for read-only operations
        std::shared_lock lock(nodes_mutex_);

        const FilterFn* filter_ptr = filter ? &filter : nullptr;

        // Phase 1: Navigate from top layer to layer 1
        size_t current = entry_point_id_.load(std::memory_order_acquire);
        for (size_t lc = entry_point_layer_.load(std::memory_order_acquire); lc > 0; --lc) {
            current = greedy_search_layer_locked(query, current, lc, filter_ptr);
        }

        // Phase 2: Beam search at layer 0
        auto candidates = beam_search_layer_locked(query, current, ef_search, 0, filter_ptr);

        // Phase 3: Return top-k
        if (candidates.size() > k) {
            candidates.resize(k);
        }

        return candidates;
    }

    /// Batch build from vectors (sequential)
    void build(std::span<const size_t> ids, std::span<const std::span<const StorageT>> vectors) {
        if (ids.size() != vectors.size()) {
            throw std::invalid_argument("ids and vectors must have same size");
        }

        for (size_t i = 0; i < ids.size(); ++i) {
            insert(ids[i], vectors[i]);
        }
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
        if (ids.size() != vectors.size()) {
            throw std::invalid_argument("ids and vectors must have same size");
        }

        if (ids.empty())
            return;

        size_t actual_threads = num_threads ? num_threads : std::thread::hardware_concurrency();
        if (actual_threads == 0) {
            actual_threads = 1;
        }

        ThreadPool pool(actual_threads);

        pool.parallel_for(ids.size(),
                          [&](size_t /*thread*/, size_t i) { insert(ids[i], vectors[i]); });
    }

    /// Parallel batch build with configurable batch size
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

        size_t actual_threads = num_threads ? num_threads : std::thread::hardware_concurrency();
        if (actual_threads == 0) {
            actual_threads = 1;
        }

        ThreadPool pool(actual_threads);

        constexpr size_t DEFAULT_BATCH_SIZE = 256;
        size_t actual_batch_size = batch_size ? batch_size : DEFAULT_BATCH_SIZE;
        size_t num_batches = (ids.size() + actual_batch_size - 1) / actual_batch_size;

        std::atomic<size_t> max_layer{0};
        std::vector<size_t> node_layers(ids.size());

        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t batch_start = batch * actual_batch_size;
            size_t batch_end = std::min(batch_start + actual_batch_size, ids.size());
            size_t batch_items = batch_end - batch_start;

            // Phase 1: Create nodes in batch (with write lock)
            pool.parallel_for(batch_items, [&](size_t /*thread*/, size_t idx) {
                size_t i = batch_start + idx;
                size_t layer = random_layer();
                node_layers[i] = layer;

                {
                    std::unique_lock lock(nodes_mutex_);
                    nodes_.emplace(ids[i], NodeType(ids[i], vectors[i], layer));

                    size_t current_max = max_layer.load(std::memory_order_relaxed);
                    while (layer > current_max) {
                        if (max_layer.compare_exchange_weak(current_max, layer,
                                                            std::memory_order_relaxed,
                                                            std::memory_order_relaxed)) {
                            break;
                        }
                    }
                }
            });

            // Phase 2: Connect nodes in batch (without global lock)
            pool.parallel_for(batch_items, [&](size_t /*thread*/, size_t idx) {
                size_t i = batch_start + idx;
                size_t layer = node_layers[i];

                // Navigate from entry point
                size_t current = entry_point_id_.load(std::memory_order_acquire);
                for (size_t lc = entry_point_layer_.load(std::memory_order_acquire); lc > layer;
                     --lc) {
                    current = greedy_search_layer_batch(current, lc, ids[i], vectors[i]);
                }

                // Find and connect to neighbors
                for (size_t lc = layer;; --lc) {
                    auto candidates = beam_search_layer_batch(current, ids[i], vectors[i], lc);

                    size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
                    size_t num_connections = std::min(M, candidates.size());

                    for (size_t c = 0; c < num_connections; ++c) {
                        size_t neighbor_id = candidates[c].first;

                        auto it_node = nodes_.find(ids[i]);
                        auto it_neighbor = nodes_.find(neighbor_id);
                        if (it_node == nodes_.end() || it_neighbor == nodes_.end())
                            continue;

                        it_node->second.add_edge(neighbor_id, lc);
                        it_neighbor->second.add_edge(ids[i], lc);

                        size_t M_max = (lc == 0) ? config_.M_max_0 : config_.M_max;
                        if (it_neighbor->second.neighbors(lc).size() > M_max) {
                            prune_connections_batch(it_neighbor->first, lc, ids[i]);
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

        // Update entry point after all batches
        size_t max_l = max_layer.load(std::memory_order_relaxed);
        entry_point_layer_.store(max_l, std::memory_order_relaxed);

        std::shared_lock lock(nodes_mutex_);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (node_layers[i] == max_l) {
                entry_point_id_.store(ids[i], std::memory_order_relaxed);
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

    /// Iterator support for serialization
    auto begin() const { return nodes_.begin(); }
    auto end() const { return nodes_.end(); }

    /// Get node by ID (for serialization)
    const NodeType* get_node(size_t id) const {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    // ========== Soft Deletion API ==========

    /// Mark node as deleted (soft delete)
    /// Graph edges remain but node is excluded from search results
    /// @param id Node ID to mark as deleted
    void remove(size_t id) {
        if (nodes_.count(id) == 0)
            return;
        deleted_ids_.insert(id);
    }

    /// Check if node is soft-deleted
    /// @param id Node ID to check
    /// @return true if node is marked as deleted
    [[nodiscard]] bool is_deleted(size_t id) const { return deleted_ids_.contains(id); }

    /// Count of soft-deleted nodes
    [[nodiscard]] size_t deleted_count() const { return deleted_ids_.size(); }

    /// Count of active (non-deleted) nodes
    [[nodiscard]] size_t active_size() const { return nodes_.size() - deleted_ids_.size(); }

    /// Get the set of deleted IDs (for serialization)
    [[nodiscard]] const std::unordered_set<size_t>& deleted_ids() const { return deleted_ids_; }

    /// Check if compaction is recommended
    /// @param threshold Fraction of deleted nodes (default 0.2 = 20%)
    /// @return true if deleted_count > threshold * size
    [[nodiscard]] bool needs_compaction(float threshold = 0.2f) const {
        if (nodes_.empty())
            return false;
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
        if (deleted_ids_.empty())
            return;

        for (auto& [id, node] : nodes_) {
            if (is_deleted(id))
                continue;

            // Remove edges to deleted nodes at each layer
            for (size_t layer = 0; layer < node.edges.size(); ++layer) {
                auto& layer_edges = node.edges[layer];
                std::erase_if(layer_edges,
                              [this](size_t neighbor) { return is_deleted(neighbor); });
            }
        }
    }

    /// Restore a soft-deleted node (undo delete)
    /// @param id Node ID to restore
    /// @return true if node was restored, false if not found or not deleted
    bool restore(size_t id) {
        if (nodes_.count(id) == 0)
            return false;
        return deleted_ids_.erase(id) > 0;
    }

    /// Clear all deletion markers (does not rebuild graph)
    void clear_deletions() { deleted_ids_.clear(); }

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
    Config config_;

    // Thread-safe graph storage
    mutable std::shared_mutex nodes_mutex_;
    std::unordered_map<size_t, NodeType> nodes_;
    std::unordered_set<size_t> deleted_ids_;

    // Atomic entry point (thread-safe updates)
    std::atomic<size_t> entry_point_id_{0};
    std::atomic<size_t> entry_point_layer_{0};

    // Per-thread RNG for parallel operations
    ThreadLocalRNG rng_generator_;

    /// Select random layer using thread-local RNG
    size_t random_layer() { return rng_generator_.random_layer(config_.ml_factor); }

    /// Greedy search (read-only, called under shared lock)
    size_t greedy_search_layer_locked(std::span<const float> query, size_t entry_point,
                                      size_t layer, const FilterFn* filter) const {
        // Small negative threshold to handle floating-point error in distance calculation
        constexpr float kDistanceEpsilon = -1e-5f;

        size_t current = entry_point;
        float current_dist = distance(query, current);

        auto passes_filter = [&](size_t id) {
            return !is_deleted(id) && (!filter || (*filter)(id));
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
            for (size_t neighbor : current_node->neighbors(layer)) {
                float neighbor_dist = distance(query, neighbor);
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

    /// Greedy search (direct access, for insert phase with write lock held)
    size_t greedy_search_layer(std::span<const float> query, size_t entry_point, size_t layer,
                               const FilterFn* filter) const {
        return greedy_search_layer_locked(query, entry_point, layer, filter);
    }

    /// Beam search (read-only, called under shared lock)
    std::vector<std::pair<size_t, float>> beam_search_layer_locked(std::span<const float> query,
                                                                   size_t entry_point, size_t ef,
                                                                   size_t layer,
                                                                   const FilterFn* filter) const {
        auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp)>
            top_candidates(cmp);

        auto cmp_min = [](const auto& a, const auto& b) { return a.first > b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp_min)>
            candidates(cmp_min);

        std::unordered_set<size_t> visited;

        const auto* entry_node = try_get_node(entry_point);
        if (!entry_node)
            return {};

        float entry_dist = distance(query, entry_point);
        candidates.emplace(entry_dist, entry_point);
        visited.insert(entry_point);

        auto passes_filter = [&](size_t id) {
            return !is_deleted(id) && (!filter || (*filter)(id));
        };

        // Note: entry_dist can be slightly negative due to floating-point error
        // (e.g., cosine distance of identical vectors = 1 - 1.0000001 = -1e-7)
        // We allow small negative values to avoid missing exact matches.
        constexpr float kDistanceEpsilon = -1e-5f;
        if (passes_filter(entry_point) && entry_dist >= kDistanceEpsilon) {
            top_candidates.emplace(std::max(0.0f, entry_dist), entry_point);
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

            for (size_t neighbor : current_node->neighbors(layer)) {
                if (visited.count(neighbor))
                    continue;
                visited.insert(neighbor);

                float neighbor_dist = distance(query, neighbor);
                if (neighbor_dist < kDistanceEpsilon)
                    continue;

                bool should_explore = top_candidates.empty() || top_candidates.size() < ef ||
                                      neighbor_dist < top_candidates.top().first;

                if (should_explore) {
                    candidates.emplace(std::max(0.0f, neighbor_dist), neighbor);
                }

                if (passes_filter(neighbor)) {
                    if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                        top_candidates.emplace(std::max(0.0f, neighbor_dist), neighbor);
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

    /// Beam search (direct access, for insert phase)
    std::vector<std::pair<size_t, float>> beam_search_layer(std::span<const float> query,
                                                            size_t entry_point, size_t ef,
                                                            size_t layer,
                                                            const FilterFn* filter) const {
        return beam_search_layer_locked(query, entry_point, ef, layer, filter);
    }

    /// Connect node to M nearest neighbors at layer
    void connect_neighbors(size_t node_id, const std::vector<std::pair<size_t, float>>& candidates,
                           size_t M, size_t layer) {
        size_t num_connections = std::min(M, candidates.size());

        for (size_t i = 0; i < num_connections; ++i) {
            size_t neighbor_id = candidates[i].first;

            // Verify both nodes exist (may not during concurrent insertion)
            auto it_node = nodes_.find(node_id);
            auto it_neighbor = nodes_.find(neighbor_id);
            if (it_node == nodes_.end() || it_neighbor == nodes_.end())
                continue;

            it_node->second.add_edge(neighbor_id, layer);
            it_neighbor->second.add_edge(node_id, layer);

            size_t M_max = (layer == 0) ? config_.M_max_0 : config_.M_max;
            if (it_neighbor->second.neighbors(layer).size() > M_max) {
                prune_connections(it_neighbor->first, layer);
            }
        }
    }

    /// Prune connections to maintain M_max limit
    void prune_connections(size_t node_id, size_t layer) {
        auto it = nodes_.find(node_id);
        if (it == nodes_.end())
            return;

        size_t M_max = (layer == 0) ? config_.M_max_0 : config_.M_max;
        auto& node = it->second;
        auto neighbors = node.neighbors(layer);

        if (neighbors.size() <= M_max)
            return;

        std::vector<std::pair<size_t, float>> neighbor_dists;
        neighbor_dists.reserve(neighbors.size());
        for (size_t neighbor : neighbors) {
            // Verify neighbor still exists
            if (nodes_.find(neighbor) == nodes_.end())
                continue;
            neighbor_dists.emplace_back(neighbor, distance(node_id, neighbor));
        }

        std::sort(neighbor_dists.begin(), neighbor_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        std::unordered_set<size_t> to_remove;
        for (size_t i = M_max; i < neighbor_dists.size(); ++i) {
            to_remove.insert(neighbor_dists[i].first);
        }

        for (size_t neighbor : to_remove) {
            node.remove_edge(neighbor, layer);
            auto it_neighbor = nodes_.find(neighbor);
            if (it_neighbor != nodes_.end()) {
                it_neighbor->second.remove_edge(node_id, layer);
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

    /// Try to get node by ID (returns nullptr if not found, safe for concurrent access)
    const NodeType* try_get_node(size_t id) const {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    /// Distance between two StorageT nodes
    float distance_nodes(const NodeType& n1, const NodeType& n2) const {
        if constexpr (std::same_as<StorageT, float>) {
            return config_.metric(n1.as_span(), n2.as_span());
        } else {
            auto f1 = n1.as_float32();
            auto f2 = n2.as_float32();
            return config_.metric(std::span<const float>(f1), std::span<const float>(f2));
        }
    }

    /// Distance between float32 query and StorageT node
    float distance_query_node(std::span<const float> query, const NodeType& node) const {
        if constexpr (std::same_as<StorageT, float>) {
            return config_.metric(query, node.as_span());
        } else {
            auto node_vec = node.as_float32();
            return config_.metric(query, std::span<const float>(node_vec));
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
