#pragma once

#include <cassert>
#include <cmath>
#include <span>
#include <type_traits>
#include "../concepts/distance_metric.hpp"
#include "../concepts/vector_element.hpp"

namespace sqlite_vec_cpp::distances {

/// Cosine distance = 1 - cosine_similarity
/// cosine_similarity = dot(a,b) / (||a|| * ||b||)
template <concepts::VectorElement T>
float cosine_distance_fallback(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size() && "Vector dimensions must match");

    float dot = 0.0f;
    float a_mag = 0.0f;
    float b_mag = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i) {
        const float a_val = static_cast<float>(a[i]);
        const float b_val = static_cast<float>(b[i]);
        dot += a_val * b_val;
        a_mag += a_val * a_val;
        b_mag += b_val * b_val;
    }

    // Return 1 - similarity to convert to distance
    return 1.0f - (dot / (std::sqrt(a_mag) * std::sqrt(b_mag)));
}

/// Specialized for floating-point (same as fallback, but explicit)
template <concepts::FloatingPointElement T>
float cosine_distance_float(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size());

    float dot = 0.0f;
    float a_mag = 0.0f;
    float b_mag = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i) {
        const float a_val = static_cast<float>(a[i]);
        const float b_val = static_cast<float>(b[i]);
        dot += a_val * b_val;
        a_mag += a_val * a_val;
        b_mag += b_val * b_val;
    }

    return 1.0f - (dot / (std::sqrt(a_mag) * std::sqrt(b_mag)));
}

/// Specialized for integer types
template <concepts::IntegerElement T>
float cosine_distance_int(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size());

    float dot = 0.0f;
    float a_mag = 0.0f;
    float b_mag = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i) {
        const float a_val = static_cast<float>(a[i]);
        const float b_val = static_cast<float>(b[i]);
        dot += a_val * b_val;
        a_mag += a_val * a_val;
        b_mag += b_val * b_val;
    }

    return 1.0f - (dot / (std::sqrt(a_mag) * std::sqrt(b_mag)));
}

// Forward declarations for SIMD implementations (if needed)
// Cosine is less commonly SIMD-optimized than L1/L2, but could be added

/// Main cosine distance function
template <concepts::VectorElement T>
float cosine_distance(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size() && "Vector dimensions must match");

    // Type-based dispatch (SIMD could be added later)
    if constexpr (concepts::FloatingPointElement<T>) {
        return cosine_distance_float(a, b);
    } else if constexpr (concepts::IntegerElement<T>) {
        return cosine_distance_int(a, b);
    } else {
        return cosine_distance_fallback(a, b);
    }
}

/// CosineMetric functor (satisfies DistanceMetric concept)
template <concepts::VectorElement T> struct CosineMetric {
    using element_type = T;
    // Note: Cosine is a similarity metric converted to distance
    // It's symmetric but doesn't satisfy triangle inequality (not a true metric)

    [[nodiscard]] float operator()(std::span<const T> a, std::span<const T> b) const {
        return cosine_distance(a, b);
    }
};

} // namespace sqlite_vec_cpp::distances

// Trait specializations (must be in concepts::traits namespace)
namespace sqlite_vec_cpp::concepts::traits {
template <typename T>
struct is_symmetric<sqlite_vec_cpp::distances::CosineMetric<T>> : std::true_type {};

// Cosine distance does NOT satisfy triangle inequality
template <typename T>
struct is_metric_space<sqlite_vec_cpp::distances::CosineMetric<T>> : std::false_type {};

// It is fundamentally a similarity metric
template <typename T>
struct is_similarity<sqlite_vec_cpp::distances::CosineMetric<T>> : std::true_type {};
} // namespace sqlite_vec_cpp::concepts::traits

namespace sqlite_vec_cpp::distances {

// Verify concept satisfaction
static_assert(concepts::DistanceMetric<CosineMetric<float>, float>);
static_assert(concepts::DistanceMetric<CosineMetric<std::int8_t>, std::int8_t>);
static_assert(concepts::SymmetricDistanceMetric<CosineMetric<float>, float>);
// Note: NOT MetricSpace (doesn't satisfy triangle inequality)

} // namespace sqlite_vec_cpp::distances
