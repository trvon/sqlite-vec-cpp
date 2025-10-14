#pragma once

#include <cmath>
#include <cstdlib>
#include <span>
#include <type_traits>
#include "../concepts/distance_metric.hpp"
#include "../concepts/vector_element.hpp"

// Include SIMD implementations
#ifdef SQLITE_VEC_ENABLE_NEON
#include "../simd/neon.hpp"
#endif

namespace sqlite_vec_cpp::distances {

/// L1 (Manhattan) distance metric - generic fallback
template <concepts::VectorElement T>
double l1_distance_fallback(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size() && "Vector dimensions must match");

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum;
}

/// Specialized for floating-point types
template <concepts::FloatingPointElement T>
double l1_distance_float(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size());

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum;
}

/// Specialized for signed integer types (returns int64 for exact result)
template <concepts::SignedIntegerElement T>
std::int64_t l1_distance_int(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size());

    std::int64_t sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(static_cast<std::int64_t>(a[i]) - static_cast<std::int64_t>(b[i]));
    }
    return sum;
}

// Forward declarations for SIMD implementations - now defined in simd/*.hpp
// (Declarations removed - directly included above)

/// Main L1 distance function with SIMD dispatch
/// Returns float for compatibility (can represent both int and float results)
template <concepts::VectorElement T> float l1_distance(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size() && "Vector dimensions must match");

    if constexpr (std::is_same_v<T, float>) {
#ifdef SQLITE_VEC_ENABLE_NEON
        if (a.size() > 3) {
            return static_cast<float>(simd::l1_distance_float_neon(a, b));
        }
#endif
        return static_cast<float>(l1_distance_float(a, b));
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
#ifdef SQLITE_VEC_ENABLE_NEON
        if (a.size() > 15) {
            return static_cast<float>(simd::l1_distance_int8_neon(a, b));
        }
#endif
        return static_cast<float>(l1_distance_int(a, b));
    } else if constexpr (concepts::SignedIntegerElement<T>) {
        return static_cast<float>(l1_distance_int(a, b));
    } else if constexpr (concepts::FloatingPointElement<T>) {
        return static_cast<float>(l1_distance_float(a, b));
    } else {
        return static_cast<float>(l1_distance_fallback(a, b));
    }
}

/// L1Metric functor (satisfies DistanceMetric concept)
template <concepts::VectorElement T> struct L1Metric {
    using element_type = T;
    using has_simd_impl = void; // Marker for SIMD support

    [[nodiscard]] float operator()(std::span<const T> a, std::span<const T> b) const {
        return l1_distance(a, b);
    }
};

} // namespace sqlite_vec_cpp::distances

// Trait specializations (must be in concepts::traits namespace)
namespace sqlite_vec_cpp::concepts::traits {
template <typename T>
struct is_symmetric<sqlite_vec_cpp::distances::L1Metric<T>> : std::true_type {};

template <typename T>
struct is_metric_space<sqlite_vec_cpp::distances::L1Metric<T>> : std::true_type {};
} // namespace sqlite_vec_cpp::concepts::traits

namespace sqlite_vec_cpp::distances {

// Verify concept satisfaction
static_assert(concepts::DistanceMetric<L1Metric<float>, float>);
static_assert(concepts::DistanceMetric<L1Metric<std::int8_t>, std::int8_t>);
static_assert(concepts::SymmetricDistanceMetric<L1Metric<float>, float>);
static_assert(concepts::MetricSpace<L1Metric<float>, float>);

} // namespace sqlite_vec_cpp::distances
