#pragma once

#include <concepts>
#include <span>
#include <type_traits>
#include <utility> // for std::as_const
#include "vector_element.hpp"

namespace sqlite_vec_cpp::concepts {

/// Concept for distance metric functors
/// A distance metric takes two vector views and returns a scalar distance
template <typename M, typename T>
concept DistanceMetric =
    VectorElement<T> && requires(const M& metric, std::span<const T> a, std::span<const T> b) {
        // Must be callable with two spans of same type
        { metric(a, b) } -> std::convertible_to<float>;

        // Must be const-callable (stateless or immutable state)
        { std::as_const(metric)(a, b) } -> std::convertible_to<float>;
    };

/// Concept for symmetric distance metrics (distance(a,b) == distance(b,a))
template <typename M, typename T>
concept SymmetricDistanceMetric = DistanceMetric<M, T>;

/// Concept for metrics that satisfy triangle inequality
template <typename M, typename T>
concept MetricSpace = SymmetricDistanceMetric<M, T>;

/// Concept for similarity metrics (higher = more similar)
/// These need to be converted to distance (e.g., 1 - similarity)
template <typename M, typename T>
concept SimilarityMetric =
    VectorElement<T> && requires(const M& metric, std::span<const T> a, std::span<const T> b) {
        { metric(a, b) } -> std::convertible_to<float>;
        { M::is_similarity } -> std::convertible_to<bool>;
    };

/// Concept for SIMD-optimizable metrics
template <typename M, typename T>
concept SIMDOptimizable = DistanceMetric<M, T> && requires {
    // Check if type has SIMD implementation marker
    typename M::template has_simd_impl<T>;
};

namespace traits {

/// Check if a metric is symmetric
template <typename M> struct is_symmetric : std::false_type {};

/// Check if a metric satisfies triangle inequality (true metric)
template <typename M> struct is_metric_space : std::false_type {};

/// Check if a metric is a similarity (vs distance)
template <typename M> struct is_similarity : std::false_type {};

/// Helper variable templates
template <typename M> inline constexpr bool is_symmetric_v = is_symmetric<M>::value;

template <typename M> inline constexpr bool is_metric_space_v = is_metric_space<M>::value;

template <typename M> inline constexpr bool is_similarity_v = is_similarity<M>::value;

} // namespace traits

// Forward declarations for common metrics (defined in distances/)
template <VectorElement T> struct L1Metric;
template <VectorElement T> struct L2Metric;
template <VectorElement T> struct CosineMetric;
template <VectorElement T> struct HammingMetric;

// Static assertions will be added after metric definitions
// static_assert(DistanceMetric<L1Metric<float>, float>);
// static_assert(SymmetricDistanceMetric<L2Metric<float>, float>);

} // namespace sqlite_vec_cpp::concepts
