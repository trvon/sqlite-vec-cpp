#pragma once

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../concepts/distance_metric.hpp"
#include "../concepts/vector_element.hpp"
#include "hnsw_node.hpp"

namespace sqlite_vec_cpp::index {

/// Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search
/// Provides 100-1000x speedup over brute-force for large corpora (>100K vectors)
template <concepts::VectorElement T, concepts::DistanceMetric<T> Metric> class HNSWIndex {
public:
    /// Configuration parameters
    struct Config {
        size_t M = 16;                           ///< Connections per node per layer
        size_t M_max = 32;                       ///< Max connections (layer > 0)
        size_t M_max_0 = 32;                     ///< Max connections (layer 0)
        size_t ef_construction = 200;            ///< Exploration factor during construction
        float ml_factor = 1.0f / std::log(2.0f); ///< Layer selection multiplier
        Metric metric{};                         ///< Distance metric
    };

    /// Construct empty HNSW index
    explicit HNSWIndex(Config config = {})
        : config_(config), rng_(42) {} // Deterministic seed for testing

    /// Factory method for deserialization
    /// Reconstructs HNSW index from serialized state without rebuilding graph
    /// @param config Index configuration
    /// @param entry_point_id ID of entry point node
    /// @param entry_point_layer Layer of entry point
    /// @param nodes Graph nodes with edges
    /// @return Fully reconstructed HNSW index
    static HNSWIndex from_serialized(const Config& config, size_t entry_point_id,
                                     size_t entry_point_layer,
                                     std::unordered_map<size_t, HNSWNode<T>>&& nodes) {
        HNSWIndex index(config);
        index.entry_point_id_ = entry_point_id;
        index.entry_point_layer_ = entry_point_layer;
        index.nodes_ = std::move(nodes);
        return index;
    }

    /// Insert vector into index
    /// @param id External identifier (e.g., SQLite rowid)
    /// @param vector Embedding vector
    void insert(size_t id, std::span<const T> vector) {
        // Assign random layer
        size_t layer = random_layer();

        // Create node
        nodes_.emplace(id, HNSWNode<T>(id, vector, layer));

        // First node becomes entry point
        if (nodes_.size() == 1) {
            entry_point_id_ = id;
            entry_point_layer_ = layer;
            return;
        }

        // Update entry point if new node has higher layer
        if (layer > entry_point_layer_) {
            size_t old_entry = entry_point_id_;
            size_t old_layer = entry_point_layer_;
            entry_point_id_ = id;
            entry_point_layer_ = layer;

            // Insert at all layers from old entry layer down to 0
            // using old entry point as the starting point for search
            size_t current = old_entry;
            for (size_t lc = old_layer;; --lc) {
                auto candidates = beam_search_layer(vector, current, config_.ef_construction, lc);

                // Connect to M best candidates
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

        // Navigate from entry point to find insertion point at each layer
        size_t current = entry_point_id_;

        // Phase 1: Navigate layers above node's layer (greedy)
        for (size_t lc = entry_point_layer_; lc > layer; --lc) {
            current = greedy_search_layer(vector, current, lc);
        }

        // Phase 2: Insert at node's layer and below (beam search + connect)
        for (size_t lc = layer;; --lc) {
            auto candidates = beam_search_layer(vector, current, config_.ef_construction, lc);

            // Connect to M best candidates
            size_t M = (lc == 0) ? config_.M_max_0 : config_.M;
            connect_neighbors(id, candidates, M, lc);

            if (lc == 0)
                break; // Guard against underflow
            if (!candidates.empty()) {
                current = candidates[0].first; // Use best candidate as entry for next layer
            }
        }
    }

    /// Search for k nearest neighbors
    /// @param query Query vector
    /// @param k Number of neighbors to return
    /// @param ef_search Exploration factor (higher = better recall, slower)
    /// @return Vector of (id, distance) pairs, sorted by distance
    std::vector<std::pair<size_t, float>> search(std::span<const T> query, size_t k,
                                                 size_t ef_search = 50) const {
        if (nodes_.empty())
            return {};

        ef_search = std::max(ef_search, k); // ef_search must be >= k

        // Phase 1: Navigate from top layer to layer 1
        size_t current = entry_point_id_;
        for (size_t lc = entry_point_layer_; lc > 0; --lc) {
            current = greedy_search_layer(query, current, lc);
        }

        // Phase 2: Beam search at layer 0
        auto candidates = beam_search_layer(query, current, ef_search, 0);

        // Phase 3: Return top-k
        if (candidates.size() > k) {
            candidates.resize(k);
        }

        return candidates;
    }

    /// Batch build from vectors
    /// More efficient than individual inserts for initial construction
    void build(std::span<const size_t> ids, std::span<const std::span<const T>> vectors) {
        if (ids.size() != vectors.size()) {
            throw std::invalid_argument("ids and vectors must have same size");
        }

        // Insert all vectors
        for (size_t i = 0; i < ids.size(); ++i) {
            insert(ids[i], vectors[i]);
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
    const HNSWNode<T>* get_node(size_t id) const {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

private:
    Config config_;
    std::unordered_map<size_t, HNSWNode<T>> nodes_;
    size_t entry_point_id_ = 0;
    size_t entry_point_layer_ = 0;
    mutable std::mt19937 rng_; // Mutable for search (deterministic)

    /// Select random layer using exponential decay
    size_t random_layer() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        if (r == 0.0f)
            r = 1e-9f; // Avoid log(0)
        return static_cast<size_t>(-std::log(r) * config_.ml_factor);
    }

    /// Greedy search at single layer (returns 1 nearest neighbor)
    size_t greedy_search_layer(std::span<const T> query, size_t entry_point, size_t layer) const {
        size_t current = entry_point;
        float current_dist = distance(query, current);

        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t neighbor : nodes_.at(current).neighbors(layer)) {
                float neighbor_dist = distance(query, neighbor);
                if (neighbor_dist < current_dist) {
                    current = neighbor;
                    current_dist = neighbor_dist;
                    changed = true;
                }
            }
        }

        return current;
    }

    /// Beam search at single layer (returns ef nearest neighbors)
    std::vector<std::pair<size_t, float>>
    beam_search_layer(std::span<const T> query, size_t entry_point, size_t ef, size_t layer) const {
        // Priority queue: (distance, node_id) - max heap (worst candidate at top)
        auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp)>
            top_candidates(cmp);

        // Min heap for candidates to explore
        auto cmp_min = [](const auto& a, const auto& b) { return a.first > b.first; };
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                            decltype(cmp_min)>
            candidates(cmp_min);

        std::unordered_set<size_t> visited;

        float entry_dist = distance(query, entry_point);
        candidates.emplace(entry_dist, entry_point);
        top_candidates.emplace(entry_dist, entry_point);
        visited.insert(entry_point);

        while (!candidates.empty()) {
            auto [current_dist, current_id] = candidates.top();
            candidates.pop();

            // Early termination: current is worse than ef-th best
            if (current_dist > top_candidates.top().first) {
                break;
            }

            // Explore neighbors
            for (size_t neighbor : nodes_.at(current_id).neighbors(layer)) {
                if (visited.count(neighbor))
                    continue;
                visited.insert(neighbor);

                float neighbor_dist = distance(query, neighbor);

                if (neighbor_dist < top_candidates.top().first || top_candidates.size() < ef) {
                    candidates.emplace(neighbor_dist, neighbor);
                    top_candidates.emplace(neighbor_dist, neighbor);

                    // Maintain ef size
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        }

        // Convert to result format (id, distance)
        std::vector<std::pair<size_t, float>> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            auto [dist, id] = top_candidates.top();
            top_candidates.pop();
            result.emplace_back(id, dist);
        }

        // Sort by distance (ascending)
        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        return result;
    }

    /// Connect node to M nearest neighbors at layer
    void connect_neighbors(size_t node_id, const std::vector<std::pair<size_t, float>>& candidates,
                           size_t M, size_t layer) {
        // Select M best candidates
        size_t num_connections = std::min(M, candidates.size());

        for (size_t i = 0; i < num_connections; ++i) {
            size_t neighbor_id = candidates[i].first;

            // Add bidirectional edge
            nodes_.at(node_id).add_edge(neighbor_id, layer);
            nodes_.at(neighbor_id).add_edge(node_id, layer);

            // Prune neighbor's connections if exceeds M_max
            size_t M_max = (layer == 0) ? config_.M_max_0 : config_.M_max;
            if (nodes_.at(neighbor_id).neighbors(layer).size() > M_max) {
                prune_connections(neighbor_id, layer);
            }
        }
    }

    /// Prune connections to maintain M_max limit
    void prune_connections(size_t node_id, size_t layer) {
        size_t M_max = (layer == 0) ? config_.M_max_0 : config_.M_max;
        auto& node = nodes_.at(node_id);
        auto neighbors = node.neighbors(layer);

        if (neighbors.size() <= M_max)
            return;

        // Compute distances to all neighbors
        std::vector<std::pair<size_t, float>> neighbor_dists;
        neighbor_dists.reserve(neighbors.size());
        for (size_t neighbor : neighbors) {
            neighbor_dists.emplace_back(neighbor, distance(node_id, neighbor));
        }

        // Sort by distance
        std::sort(neighbor_dists.begin(), neighbor_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Keep M_max closest, remove rest
        std::unordered_set<size_t> to_remove;
        for (size_t i = M_max; i < neighbor_dists.size(); ++i) {
            to_remove.insert(neighbor_dists[i].first);
        }

        // Remove edges
        for (size_t neighbor : to_remove) {
            node.remove_edge(neighbor, layer);
            nodes_.at(neighbor).remove_edge(node_id, layer);
        }
    }

    /// Calculate distance between two nodes
    float distance(size_t id1, size_t id2) const {
        const auto& n1 = nodes_.at(id1);
        const auto& n2 = nodes_.at(id2);
        return config_.metric(n1.as_span(), n2.as_span());
    }

    /// Calculate distance between query and node
    float distance(std::span<const T> query, size_t node_id) const {
        return config_.metric(query, nodes_.at(node_id).as_span());
    }
};

} // namespace sqlite_vec_cpp::index
