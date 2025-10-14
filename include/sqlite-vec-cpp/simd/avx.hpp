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

} // namespace sqlite_vec_cpp::distances::simd

#endif // SQLITE_VEC_ENABLE_AVX
