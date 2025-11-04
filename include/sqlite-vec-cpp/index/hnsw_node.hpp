#pragma once

#include <cstddef>
#include <span>
#include <vector>
#include "../concepts/vector_element.hpp"

namespace sqlite_vec_cpp::index {

/// Node in the HNSW graph
/// Each node represents a vector with connections to neighbors at each layer
template <concepts::VectorElement T> struct HNSWNode {
    size_t id;                              ///< External ID (e.g., SQLite rowid)
    std::vector<T> vector;                  ///< Embedded vector (owned copy)
    std::vector<std::vector<size_t>> edges; ///< edges[layer] = list of neighbor IDs

    /// Construct node with vector copy
    HNSWNode(size_t node_id, std::span<const T> vec, size_t max_layer)
        : id(node_id), vector(vec.begin(), vec.end()), edges(max_layer + 1) {}

    /// Get connections at specific layer
    std::span<const size_t> neighbors(size_t layer) const {
        if (layer >= edges.size())
            return {};
        return edges[layer];
    }

    /// Add bidirectional edge at layer
    void add_edge(size_t neighbor_id, size_t layer) {
        if (layer >= edges.size())
            return;
        edges[layer].push_back(neighbor_id);
    }

    /// Remove edge at layer
    void remove_edge(size_t neighbor_id, size_t layer) {
        if (layer >= edges.size())
            return;
        auto& layer_edges = edges[layer];
        layer_edges.erase(std::remove(layer_edges.begin(), layer_edges.end(), neighbor_id),
                          layer_edges.end());
    }

    /// Get vector as span
    std::span<const T> as_span() const { return std::span{vector}; }

    /// Number of layers (highest layer + 1)
    size_t num_layers() const { return edges.size(); }

    /// Check if node has any connections at layer
    bool has_connections(size_t layer) const {
        return layer < edges.size() && !edges[layer].empty();
    }
};

} // namespace sqlite_vec_cpp::index
