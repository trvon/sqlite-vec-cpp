#pragma once

#include <cassert>
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

/// Inner product (dot product) distance = 1 - dot(a, b) for normalized vectors
/// For unnormalized vectors, higher dot product = more similar, so distance = -dot(a, b)
/// This implementation uses negative inner product so smaller = more similar (consistent with other
/// metrics)
template <concepts::VectorElement T>
float inner_product_distance_fallback(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size() && "Vector dimensions must match");

    float dot = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    // Negative because higher dot product = more similar
    // Adding 1.0 to keep values non-negative for normalized vectors
    return 1.0f - dot;
}

/// Specialized for floating-point types (more efficient)
template <concepts::FloatingPointElement T>
float inner_product_distance_float(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size());

    float dot = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return 1.0f - dot;
}

/// Specialized for integer types
template <concepts::IntegerElement T>
float inner_product_distance_int(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size());

    float dot = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return 1.0f - dot;
}

#ifdef SQLITE_VEC_ENABLE_NEON
namespace simd {
/// NEON-optimized inner product for float vectors
/// 4×4 unrolled: 4 accumulators × 4 floats = 16 floats per iteration.
/// For 768-dim vectors this does 192 FMA (vs 576 for full cosine).
/// Perfect for pre-normalized vectors where cosine = 1 - dot(a,b).
inline float inner_product_float_neon(std::span<const float> a, std::span<const float> b) {
    assert(a.size() == b.size());
    const std::size_t size = a.size();
    const std::size_t qty16 = size >> 4;    // size / 16
    const std::size_t end_idx = qty16 << 4; // (size / 16) * 16

    float32x4_t dot0 = vdupq_n_f32(0.0f);
    float32x4_t dot1 = vdupq_n_f32(0.0f);
    float32x4_t dot2 = vdupq_n_f32(0.0f);
    float32x4_t dot3 = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    while (i < end_idx) {
        // Process 4 sets of 4 floats = 16 floats per iteration
        float32x4_t v1 = vld1q_f32(&a[i]);
        float32x4_t v2 = vld1q_f32(&b[i]);
        dot0 = vfmaq_f32(dot0, v1, v2);
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        dot1 = vfmaq_f32(dot1, v1, v2);
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        dot2 = vfmaq_f32(dot2, v1, v2);
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        dot3 = vfmaq_f32(dot3, v1, v2);
        i += 4;
    }

    // Reduce 4 accumulators
    float32x4_t dot_total = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    float dot = vaddvq_f32(dot_total);

    // Handle remaining elements (scalar)
    while (i < size) {
        dot += a[i] * b[i];
        ++i;
    }

    return 1.0f - dot;
}
} // namespace simd
#endif

#ifdef SQLITE_VEC_ENABLE_AVX
namespace simd {
/// AVX-optimized inner product for float vectors
inline float inner_product_float_avx(std::span<const float> a, std::span<const float> b) {
    assert(a.size() == b.size());
    assert(a.size() % 8 == 0 && "Size must be multiple of 8 for AVX");

    __m256 sum_vec = _mm256_setzero_ps();

    for (std::size_t i = 0; i < a.size(); i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // Horizontal sum - extract and add lanes
    __m128 high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 low = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(high, low);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float dot = _mm_cvtss_f32(sum128);
    return 1.0f - dot;
}
} // namespace simd
#endif

/// Main inner product distance function with SIMD dispatch
template <concepts::VectorElement T>
float inner_product_distance(std::span<const T> a, std::span<const T> b) {
    assert(a.size() == b.size() && "Vector dimensions must match");

    // SIMD optimizations for specific types and sizes
    if constexpr (std::is_same_v<T, float>) {
#ifdef SQLITE_VEC_ENABLE_AVX
        // AVX: require 8-element alignment and minimum size
        if (a.size() >= 8 && a.size() % 8 == 0) {
            return simd::inner_product_float_avx(a, b);
        }
#endif
#ifdef SQLITE_VEC_ENABLE_NEON
        // NEON: 4×4 unrolled, require minimum 16 elements for efficient loop
        if (a.size() >= 16) {
            return simd::inner_product_float_neon(a, b);
        }
#endif
        return inner_product_distance_float(a, b);
    } else if constexpr (concepts::IntegerElement<T>) {
        return inner_product_distance_int(a, b);
    } else {
        return inner_product_distance_fallback(a, b);
    }
}

/// InnerProductMetric functor (satisfies DistanceMetric concept)
template <concepts::VectorElement T> struct InnerProductMetric {
    using element_type = T;
    using has_simd_impl = void; // Marker for SIMD support

    [[nodiscard]] float operator()(std::span<const T> a, std::span<const T> b) const {
        return inner_product_distance(a, b);
    }
};

} // namespace sqlite_vec_cpp::distances

// Trait specializations (must be in concepts::traits namespace)
namespace sqlite_vec_cpp::concepts::traits {
template <typename T>
struct is_symmetric<sqlite_vec_cpp::distances::InnerProductMetric<T>> : std::true_type {};

// Inner product is NOT a true metric space (doesn't satisfy triangle inequality)
template <typename T>
struct is_metric_space<sqlite_vec_cpp::distances::InnerProductMetric<T>> : std::false_type {};

// It is fundamentally a similarity metric (converted to distance)
template <typename T>
struct is_similarity<sqlite_vec_cpp::distances::InnerProductMetric<T>> : std::true_type {};
} // namespace sqlite_vec_cpp::concepts::traits

namespace sqlite_vec_cpp::distances {

// Verify concept satisfaction
static_assert(concepts::DistanceMetric<InnerProductMetric<float>, float>);
static_assert(concepts::DistanceMetric<InnerProductMetric<std::int8_t>, std::int8_t>);
static_assert(concepts::SymmetricDistanceMetric<InnerProductMetric<float>, float>);
// Note: NOT MetricSpace (doesn't satisfy triangle inequality)

} // namespace sqlite_vec_cpp::distances
