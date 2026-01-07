#pragma once

#ifdef SQLITE_VEC_ENABLE_AVX

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <span>

namespace sqlite_vec_cpp::distances::simd {

/// AVX-optimized L2 (Euclidean) distance for float vectors
/// Requires: AVX support, size % 16 == 0, size >= 16
inline float l2_distance_float_avx(std::span<const float> a, std::span<const float> b) {
    const std::size_t size = a.size();

    // Process 16 floats at a time (two __m256 vectors)
    alignas(32) float tmp_result[8];
    const std::size_t qty16 = size >> 4;    // size / 16
    const std::size_t end_idx = qty16 << 4; // (size / 16) * 16

    __m256 sum = _mm256_set1_ps(0.0f);
    std::size_t i = 0;

    while (i < end_idx) {
        // First 8 elements
        __m256 v1 = _mm256_loadu_ps(&a[i]);
        __m256 v2 = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        i += 8;

        // Next 8 elements
        v1 = _mm256_loadu_ps(&a[i]);
        v2 = _mm256_loadu_ps(&b[i]);
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        i += 8;
    }

    // Store and reduce
    _mm256_store_ps(tmp_result, sum);
    float scalar_sum = tmp_result[0] + tmp_result[1] + tmp_result[2] + tmp_result[3] +
                       tmp_result[4] + tmp_result[5] + tmp_result[6] + tmp_result[7];

    // Handle remaining elements (if any)
    while (i < size) {
        float diff = a[i] - b[i];
        scalar_sum += diff * diff;
        ++i;
    }

    return std::sqrt(scalar_sum);
}

// Note: L1 distance doesn't benefit much from AVX (no fused abs+add)
// Could add if needed for completeness

/// AVX-optimized cosine distance for float vectors
/// cosine_distance = 1 - (dot(a,b) / (||a|| * ||b||))
/// Requires: AVX support, size >= 8
inline float cosine_distance_float_avx(std::span<const float> a, std::span<const float> b) {
    const std::size_t size = a.size();
    const std::size_t qty8 = size >> 3;    // size / 8
    const std::size_t end_idx = qty8 << 3; // (size / 8) * 8

    __m256 dot_sum = _mm256_setzero_ps();
    __m256 a_sum = _mm256_setzero_ps();
    __m256 b_sum = _mm256_setzero_ps();

    std::size_t i = 0;
    while (i < end_idx) {
        __m256 v1 = _mm256_loadu_ps(&a[i]);
        __m256 v2 = _mm256_loadu_ps(&b[i]);

        // Accumulate dot product and magnitudes
        dot_sum = _mm256_fmadd_ps(v1, v2, dot_sum);
        a_sum = _mm256_fmadd_ps(v1, v1, a_sum);
        b_sum = _mm256_fmadd_ps(v2, v2, b_sum);

        i += 8;
    }

    // Horizontal sum reduction
    // dot_sum = [d0, d1, d2, d3, d4, d5, d6, d7]
    // hadd: [d0+d1, d2+d3, d4+d5, d6+d7, d0+d1, d2+d3, d4+d5, d6+d7] (not quite, but idea)
    __m128 dot_hi = _mm256_extractf128_ps(dot_sum, 1);
    __m128 dot_lo = _mm256_castps256_ps128(dot_sum);
    __m128 dot_combined = _mm_add_ps(dot_hi, dot_lo);
    dot_combined = _mm_hadd_ps(dot_combined, dot_combined);
    dot_combined = _mm_hadd_ps(dot_combined, dot_combined);
    float dot = _mm_cvtss_f32(dot_combined);

    __m128 a_hi = _mm256_extractf128_ps(a_sum, 1);
    __m128 a_lo = _mm256_castps256_ps128(a_sum);
    __m128 a_combined = _mm_add_ps(a_hi, a_lo);
    a_combined = _mm_hadd_ps(a_combined, a_combined);
    a_combined = _mm_hadd_ps(a_combined, a_combined);
    float a_mag = _mm_cvtss_f32(a_combined);

    __m128 b_hi = _mm256_extractf128_ps(b_sum, 1);
    __m128 b_lo = _mm256_castps256_ps128(b_sum);
    __m128 b_combined = _mm_add_ps(b_hi, b_lo);
    b_combined = _mm_hadd_ps(b_combined, b_combined);
    b_combined = _mm_hadd_ps(b_combined, b_combined);
    float b_mag = _mm_cvtss_f32(b_combined);

    // Handle remaining elements (scalar)
    while (i < size) {
        dot += a[i] * b[i];
        a_mag += a[i] * a[i];
        b_mag += b[i] * b[i];
        ++i;
    }

    // Compute cosine distance
    float denom = std::sqrt(a_mag) * std::sqrt(b_mag);
    if (denom < 1e-8f) {
        return 1.0f; // Avoid division by zero
    }
    return 1.0f - (dot / denom);
}

} // namespace sqlite_vec_cpp::distances::simd

#endif // SQLITE_VEC_ENABLE_AVX
