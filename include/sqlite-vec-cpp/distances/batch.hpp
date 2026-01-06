#pragma once

#include <algorithm>
#include <execution>
#include <numeric>
#include <span>
#include <vector>
#include "../concepts/distance_metric.hpp"
#include "../concepts/vector_element.hpp"
#include "cosine.hpp"
#include "l1.hpp"
#include "l2.hpp"

namespace sqlite_vec_cpp::distances::batch {

/// Batch distance calculation: 1 query vs N database vectors
/// Optimized for cache locality and instruction-level parallelism
///
/// @param query Single query vector
/// @param database Span of database vectors (N vectors)
/// @param metric Distance metric to use
/// @return Vector of N distances
template <concepts::VectorElement T, concepts::DistanceMetric<T> Metric>
std::vector<float> batch_distance(std::span<const T> query,
                                  std::span<const std::span<const T>> database,
                                  const Metric& metric = Metric{}) {
    std::vector<float> distances;
    distances.reserve(database.size());

    // Process all database vectors against single query
    // Query stays in registers/cache throughout iteration
    for (const auto& db_vec : database) {
        distances.push_back(metric(query, db_vec));
    }

    return distances;
}

/// Batch distance with contiguous database layout
/// More cache-friendly when database vectors are stored contiguously
///
/// @param query Single query vector
/// @param database Contiguous array of database vectors (N * dim elements)
/// @param num_vectors Number of vectors in database
/// @param dim Dimension of each vector
/// @param metric Distance metric to use
template <concepts::VectorElement T, concepts::DistanceMetric<T> Metric>
std::vector<float> batch_distance_contiguous(std::span<const T> query, std::span<const T> database,
                                             size_t num_vectors, size_t dim,
                                             const Metric& metric = Metric{}) {
    assert(query.size() == dim && "Query dimension mismatch");
    assert(database.size() == num_vectors * dim && "Database size mismatch");

    std::vector<float> distances;
    distances.reserve(num_vectors);

    // Slice database into individual vectors and compute distances
    for (size_t i = 0; i < num_vectors; ++i) {
        const size_t offset = i * dim;
        std::span<const T> db_vec = database.subspan(offset, dim);
        distances.push_back(metric(query, db_vec));
    }

    return distances;
}

/// Top-K nearest neighbors using batch distance
/// Returns indices of K nearest vectors (sorted by distance)
///
/// @param query Single query vector
/// @param database Span of database vectors
/// @param k Number of nearest neighbors to return
/// @param metric Distance metric to use
template <concepts::VectorElement T, concepts::DistanceMetric<T> Metric>
std::vector<size_t> batch_top_k(std::span<const T> query,
                                std::span<const std::span<const T>> database, size_t k,
                                const Metric& metric = Metric{}) {
    // Compute all distances
    auto distances = batch_distance(query, database, metric);

    // Create index vector [0, 1, 2, ..., N-1]
    std::vector<size_t> indices(distances.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort to get top-K (smallest distances)
    const size_t actual_k = std::min(k, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
                      [&distances](size_t a, size_t b) { return distances[a] < distances[b]; });

    // Return only top-K indices
    indices.resize(actual_k);
    return indices;
}

/// Parallel batch distance (for large databases, N > 1000)
/// Uses C++17 parallel algorithms for multi-threaded processing
///
/// @param query Single query vector
/// @param database Span of database vectors
/// @param metric Distance metric to use
#ifdef __cpp_lib_parallel_algorithm
template <concepts::VectorElement T, concepts::DistanceMetric<T> Metric>
std::vector<float> batch_distance_parallel(std::span<const T> query,
                                           std::span<const std::span<const T>> database,
                                           const Metric& metric = Metric{}) {
    std::vector<float> distances(database.size());

    // Create index vector for parallel processing
    std::vector<size_t> indices(database.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Parallel transform: compute distance for each index
    std::transform(std::execution::par_unseq, indices.begin(), indices.end(), distances.begin(),
                   [&](size_t i) { return metric(query, database[i]); });

    return distances;
}
#endif

/// Batch distance with result filtering (distance threshold)
/// Only returns distances below threshold (for approximate search)
///
/// @param query Single query vector
/// @param database Span of database vectors
/// @param threshold Maximum distance threshold
/// @param metric Distance metric to use
/// @return Pairs of (index, distance) for matches below threshold
template <concepts::VectorElement T, concepts::DistanceMetric<T> Metric>
std::vector<std::pair<size_t, float>>
batch_distance_filtered(std::span<const T> query, std::span<const std::span<const T>> database,
                        float threshold, const Metric& metric = Metric{}) {
    std::vector<std::pair<size_t, float>> results;
    results.reserve(database.size() / 10); // Heuristic: expect 10% matches

    for (size_t i = 0; i < database.size(); ++i) {
        float dist = metric(query, database[i]);
        if (dist < threshold) {
            results.emplace_back(i, dist);
        }
    }

    // Sort by distance
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    return results;
}

} // namespace sqlite_vec_cpp::distances::batch
