#pragma once

#include <cstddef>
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
    size_t id;                              ///< External ID (e.g., SQLite rowid)
    size_t dense_id;                        ///< Dense, contiguous ID for visited tracking
    std::vector<T> vector;                  ///< Embedded vector (owned copy)
    std::vector<std::vector<size_t>> edges; ///< edges[layer] = list of neighbor IDs
    mutable std::mutex edge_mutex_;         ///< Protects edge list access for thread safety

    /// Construct node with vector copy
    HNSWNode(size_t node_id, std::span<const T> vec, size_t max_layer, size_t M_max = 32)
        : id(node_id), dense_id(0), vector(vec.begin(), vec.end()), edges(max_layer + 1) {
        for (size_t layer = 0; layer <= max_layer; ++layer) {
            edges[layer].reserve(layer == 0 ? M_max * 2 : M_max);
        }
    }

    HNSWNode(const HNSWNode&) = delete;
    HNSWNode& operator=(const HNSWNode&) = delete;

    HNSWNode(HNSWNode&& other) noexcept
        : id(other.id), dense_id(other.dense_id), vector(std::move(other.vector)),
          edges(std::move(other.edges)), edge_mutex_() {}

    HNSWNode& operator=(HNSWNode&& other) noexcept {
        if (this != &other) {
            id = other.id;
            dense_id = other.dense_id;
            vector = std::move(other.vector);
            edges = std::move(other.edges);
        }
        return *this;
    }

    /// Get connections at specific layer (returns copy for thread safety)
    std::vector<size_t> neighbors(size_t layer) const {
        std::lock_guard<std::mutex> lock(edge_mutex_);
        if (layer >= edges.size())
            return {};
        return edges[layer];  // Return copy to avoid data race after lock release
    }

    /// Add bidirectional edge at layer (thread-safe)
    void add_edge(size_t neighbor_id, size_t layer) {
        std::lock_guard<std::mutex> lock(edge_mutex_);
        if (layer >= edges.size())
            return;
        edges[layer].push_back(neighbor_id);
    }

    /// Remove edge at layer (thread-safe)
    void remove_edge(size_t neighbor_id, size_t layer) {
        std::lock_guard<std::mutex> lock(edge_mutex_);
        if (layer >= edges.size())
            return;
        auto& layer_edges = edges[layer];
        layer_edges.erase(std::remove(layer_edges.begin(), layer_edges.end(), neighbor_id),
                          layer_edges.end());
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
