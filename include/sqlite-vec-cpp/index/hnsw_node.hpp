#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <mutex>
#include <span>
#include <vector>
#include "../concepts/vector_element.hpp"
#include "../utils/float16.hpp"

namespace sqlite_vec_cpp::index {

/// Node in the HNSW graph
/// Each node represents a vector with connections to neighbors at each layer
/// @tparam T Storage type (float or float16_t)
template <concepts::VectorElement T> struct HNSWNode {
    static constexpr size_t kInvalidDenseId = std::numeric_limits<size_t>::max();

    using value_type = T;                         ///< Element type alias for generic code
    size_t id;                                    ///< External ID (e.g., SQLite rowid)
    size_t dense_id;                              ///< Dense, contiguous ID for visited tracking & flat indexing
    std::vector<T> vector;                        ///< Embedded vector (owned copy)
    std::vector<std::vector<size_t>> edges;       ///< edges[layer] = list of neighbor external IDs
    std::vector<std::vector<size_t>> dense_edges; ///< dense_edges[layer] = parallel list of neighbor dense IDs
    mutable std::mutex edge_mutex_;               ///< Protects edge list access for thread safety

    /// Construct node with vector copy
    HNSWNode(size_t node_id, std::span<const T> vec, size_t max_layer, size_t M_max = 32)
        : id(node_id), dense_id(0), vector(vec.begin(), vec.end()), edges(max_layer + 1),
          dense_edges(max_layer + 1) {
        for (size_t layer = 0; layer <= max_layer; ++layer) {
            const size_t cap = (layer == 0 ? M_max * 2 : M_max);
            edges[layer].reserve(cap);
            dense_edges[layer].reserve(cap);
        }
    }

    HNSWNode(const HNSWNode&) = delete;
    HNSWNode& operator=(const HNSWNode&) = delete;

    HNSWNode(HNSWNode&& other) noexcept
        : id(other.id), dense_id(other.dense_id), vector(std::move(other.vector)),
          edges(std::move(other.edges)), dense_edges(std::move(other.dense_edges)), edge_mutex_() {}

    HNSWNode& operator=(HNSWNode&& other) noexcept {
        if (this != &other) {
            id = other.id;
            dense_id = other.dense_id;
            vector = std::move(other.vector);
            edges = std::move(other.edges);
            dense_edges = std::move(other.dense_edges);
        }
        return *this;
    }

    /// Get connections at specific layer (returns copy for thread safety)
    std::vector<size_t> neighbors(size_t layer) const {
        std::lock_guard<std::mutex> lock(edge_mutex_);
        if (layer >= edges.size())
            return {};
        return edges[layer]; // Return copy to avoid data race after lock release
    }

    /// Get connections at specific layer (direct reference, NO lock — caller must ensure safety)
    /// Use in single-threaded insert paths to avoid per-neighbor copy + mutex overhead
    const std::vector<size_t>& neighbors_unlocked(size_t layer) const {
        static const std::vector<size_t> empty;
        if (layer >= edges.size())
            return empty;
        return edges[layer];
    }

    /// Get dense-ID connections at specific layer (returns copy for thread safety)
    std::vector<size_t> neighbors_dense(size_t layer) const {
        std::lock_guard<std::mutex> lock(edge_mutex_);
        if (layer >= dense_edges.size())
            return {};
        return dense_edges[layer];
    }

    /// Get dense-ID connections at specific layer (direct reference, NO lock)
    const std::vector<size_t>& neighbors_dense_unlocked(size_t layer) const {
        static const std::vector<size_t> empty;
        if (layer >= dense_edges.size())
            return empty;
        return dense_edges[layer];
    }

    /// Add bidirectional edge at layer (thread-safe)
    void add_edge(size_t neighbor_id, size_t layer, size_t neighbor_dense_id = kInvalidDenseId) {
        std::lock_guard<std::mutex> lock(edge_mutex_);
        if (layer >= edges.size())
            return;
        edges[layer].push_back(neighbor_id);
        if (dense_edges.size() < edges.size()) {
            dense_edges.resize(edges.size());
        }
        dense_edges[layer].push_back(neighbor_dense_id);
    }

    /// Add edge without locking (caller must ensure single-threaded access)
    void add_edge_unlocked(size_t neighbor_id, size_t layer,
                           size_t neighbor_dense_id = kInvalidDenseId) {
        if (layer >= edges.size())
            return;
        edges[layer].push_back(neighbor_id);
        if (dense_edges.size() < edges.size()) {
            dense_edges.resize(edges.size());
        }
        dense_edges[layer].push_back(neighbor_dense_id);
    }

    /// Remove edge at layer (thread-safe)
    void remove_edge(size_t neighbor_id, size_t layer) {
        std::lock_guard<std::mutex> lock(edge_mutex_);
        remove_edge_unlocked(neighbor_id, layer);
    }

    /// Remove edge without locking (caller must ensure single-threaded access)
    void remove_edge_unlocked(size_t neighbor_id, size_t layer) {
        if (layer >= edges.size())
            return;
        auto& layer_edges = edges[layer];
        const bool has_dense =
            (layer < dense_edges.size() && dense_edges[layer].size() == layer_edges.size());
        size_t write_idx = 0;
        for (size_t i = 0; i < layer_edges.size(); ++i) {
            if (layer_edges[i] != neighbor_id) {
                if (write_idx != i) {
                    layer_edges[write_idx] = layer_edges[i];
                    if (has_dense) {
                        dense_edges[layer][write_idx] = dense_edges[layer][i];
                    }
                }
                ++write_idx;
            }
        }
        layer_edges.resize(write_idx);
        if (has_dense) {
            dense_edges[layer].resize(write_idx);
        }
    }

    /// Get vector as span (same storage type)
    std::span<const T> as_span() const { return std::span{vector}; }

    /// Get vector as float32 span (converts if needed)
    template <typename U = T>
    requires std::same_as<U, utils::float16_t>
    std::vector<float> as_float32() const {
        return utils::to_float32(std::span<const utils::float16_t>(vector));
    }

    /// Number of layers (highest layer + 1)
    size_t num_layers() const { return edges.size(); }

    /// Check if node has any connections at layer
    bool has_connections(size_t layer) const {
        return layer < edges.size() && !edges[layer].empty();
    }
};

/// Convenience type aliases
using HNSWNodeF32 = HNSWNode<float>;
using HNSWNodeF16 = HNSWNode<utils::float16_t>;

} // namespace sqlite_vec_cpp::index
