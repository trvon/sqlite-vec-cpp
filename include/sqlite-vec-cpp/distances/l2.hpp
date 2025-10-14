#pragma once

#include <cmath>
#include <span>
#include <type_traits>
#include "../concepts/distance_metric.hpp"
#include "../concepts/vector_element.hpp"

// Include SIMD implementations
#ifdef SQLITE_VEC_ENABLE_AVX
#include "../simd/avx.hpp"
#endif

#ifdef SQLITE_VEC_ENABLE_NEON
#include "../simd/neon.hpp"
#endif

namespace sqlite_vec_cpp::distances {

/// L2 (Euclidean) distance metric - generic fallback implementation
template <concepts::VectorElement T>
float l2_distance_fallback(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size() && "Vector dimensions must match");

    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/// Specialized for floating-point types (more efficient)
template <concepts::FloatingPointElement T>
float l2_distance_float(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size());

    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = static_cast<float>(a[i] - b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/// Specialized for integer types
template <concepts::IntegerElement T>
float l2_distance_int(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size());

    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Forward declarations for SIMD implementations - now defined in simd/*.hpp
// (Declarations removed - directly included above)

/// Main L2 distance function with SIMD dispatch
template <concepts::VectorElement T> float l2_distance(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size() && "Vector dimensions must match");

    // SIMD optimizations for specific types and sizes
    if constexpr (std::is_same_v<T, float>) {
#ifdef SQLITE_VEC_ENABLE_AVX
        // AVX: require 16-element alignment and minimum size
        if (a.size() >= 16 && a.size() % 16 == 0) {
            return simd::l2_distance_float_avx(a, b);
        }
#endif
#ifdef SQLITE_VEC_ENABLE_NEON
        // NEON: require minimum size
        if (a.size() > 16) {
            return simd::l2_distance_float_neon(a, b);
        }
#endif
        return l2_distance_float(a, b);
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
#ifdef SQLITE_VEC_ENABLE_NEON
        if (a.size() > 7) {
            return simd::l2_distance_int8_neon(a, b);
        }
#endif
        return l2_distance_int(a, b);
    } else if constexpr (concepts::IntegerElement<T>) {
        return l2_distance_int(a, b);
    } else {
        return l2_distance_fallback(a, b);
    }
}

/// L2Metric functor (satisfies DistanceMetric concept)
template <concepts::VectorElement T> struct L2Metric {
    using element_type = T;
    using has_simd_impl = void; // Marker for SIMD support

    [[nodiscard]] float operator()(std::span<const T> a, std::span<const T> b) const {
        return l2_distance(a, b);
    }
};

} // namespace sqlite_vec_cpp::distances

// Trait specializations (must be in concepts::traits namespace)
namespace sqlite_vec_cpp::concepts::traits {
template <typename T>
struct is_symmetric<sqlite_vec_cpp::distances::L2Metric<T>> : std::true_type {};

template <typename T>
struct is_metric_space<sqlite_vec_cpp::distances::L2Metric<T>> : std::true_type {};
} // namespace sqlite_vec_cpp::concepts::traits

namespace sqlite_vec_cpp::distances {

// Verify concept satisfaction
static_assert(concepts::DistanceMetric<L2Metric<float>, float>);
static_assert(concepts::DistanceMetric<L2Metric<std::int8_t>, std::int8_t>);
static_assert(concepts::SymmetricDistanceMetric<L2Metric<float>, float>);
static_assert(concepts::MetricSpace<L2Metric<float>, float>);

} // namespace sqlite_vec_cpp::distances
