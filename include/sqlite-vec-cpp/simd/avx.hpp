#pragma once

#ifdef SQLITE_VEC_ENABLE_AVX

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <span>

namespace sqlite_vec_cpp::distances::simd {

inline __m256 avx_fmadd_ps(__m256 a, __m256 b, __m256 c) {
#ifdef SQLITE_VEC_ENABLE_FMA
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

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

/// AVX-optimized L1 (Manhattan) distance for float vectors
/// Uses sign-bit masking for absolute value: abs(x) = andnot(sign_mask, x)
/// Requires: AVX support, size >= 8
inline float l1_distance_float_avx(std::span<const float> a, std::span<const float> b) {
    const std::size_t size = a.size();
    const std::size_t qty8 = size >> 3;
    const std::size_t end_idx = qty8 << 3;

    // Sign-bit mask: clearing the sign bit gives absolute value
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 sum = _mm256_setzero_ps();

    std::size_t i = 0;
    while (i < end_idx) {
        __m256 v1 = _mm256_loadu_ps(&a[i]);
        __m256 v2 = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(v1, v2);
        __m256 abs_diff = _mm256_andnot_ps(sign_mask, diff);
        sum = _mm256_add_ps(sum, abs_diff);
        i += 8;
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 combined = _mm_add_ps(hi, lo);
    combined = _mm_hadd_ps(combined, combined);
    combined = _mm_hadd_ps(combined, combined);
    float result = _mm_cvtss_f32(combined);

    // Scalar tail
    while (i < size) {
        float diff = a[i] - b[i];
        result += diff < 0 ? -diff : diff;
        ++i;
    }

    return result;
}

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
        dot_sum = avx_fmadd_ps(v1, v2, dot_sum);
        a_sum = avx_fmadd_ps(v1, v1, a_sum);
        b_sum = avx_fmadd_ps(v2, v2, b_sum);

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

// =============================================================================
// AVX2 integer kernels (require AVX2 for _mm256_cvtepi8_epi16, _mm256_madd_epi16)
// =============================================================================
#ifdef SQLITE_VEC_ENABLE_AVX2

#ifndef SQLITE_VEC_ENABLE_AVX
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <span>
#endif

namespace sqlite_vec_cpp::distances::simd {

/// AVX2-optimized L2 distance for int8 vectors
/// Widens int8→int16, subtracts, squares via madd, accumulates int32.
/// Requires: AVX2, size >= 16
inline float l2_distance_int8_avx2(std::span<const std::int8_t> a,
                                    std::span<const std::int8_t> b) {
    const std::size_t size = a.size();
    const std::size_t end16 = (size / 16) * 16;

    __m256i sum = _mm256_setzero_si256();

    std::size_t i = 0;
    while (i < end16) {
        // Load 16 int8s and widen to 16 int16s
        __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
        __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
        __m256i a16 = _mm256_cvtepi8_epi16(a8);
        __m256i b16 = _mm256_cvtepi8_epi16(b8);

        // diff = a - b (int16)
        __m256i diff = _mm256_sub_epi16(a16, b16);

        // diff^2 via madd: madd(diff, diff) = diff[0]^2+diff[1]^2, ... (pairwise, int32)
        __m256i sq = _mm256_madd_epi16(diff, diff);
        sum = _mm256_add_epi32(sum, sq);

        i += 16;
    }

    // Horizontal sum of 8 int32s
    __m128i hi128 = _mm256_extracti128_si256(sum, 1);
    __m128i lo128 = _mm256_castsi256_si128(sum);
    __m128i sum128 = _mm_add_epi32(lo128, hi128);
    // Shuffle and add pairs
    __m128i shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2));
    sum128 = _mm_add_epi32(sum128, shuf);
    shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
    sum128 = _mm_add_epi32(sum128, shuf);
    float scalar_sum = static_cast<float>(_mm_cvtsi128_si32(sum128));

    // Scalar tail
    while (i < size) {
        float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        scalar_sum += diff * diff;
        ++i;
    }

    return std::sqrt(scalar_sum);
}

/// AVX2-optimized cosine distance for int8 vectors
/// Three accumulator chains: dot(a,b), mag(a), mag(b)
/// Requires: AVX2, size >= 16
inline float cosine_distance_int8_avx2(std::span<const std::int8_t> a,
                                        std::span<const std::int8_t> b) {
    const std::size_t size = a.size();
    const std::size_t end16 = (size / 16) * 16;

    __m256i dot_sum = _mm256_setzero_si256();
    __m256i a_mag_sum = _mm256_setzero_si256();
    __m256i b_mag_sum = _mm256_setzero_si256();

    std::size_t i = 0;
    while (i < end16) {
        __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
        __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
        __m256i a16 = _mm256_cvtepi8_epi16(a8);
        __m256i b16 = _mm256_cvtepi8_epi16(b8);

        // madd: pairwise multiply-accumulate int16→int32
        dot_sum = _mm256_add_epi32(dot_sum, _mm256_madd_epi16(a16, b16));
        a_mag_sum = _mm256_add_epi32(a_mag_sum, _mm256_madd_epi16(a16, a16));
        b_mag_sum = _mm256_add_epi32(b_mag_sum, _mm256_madd_epi16(b16, b16));

        i += 16;
    }

    // Horizontal reduction for all three accumulators
    auto hsum_epi32 = [](__m256i v) -> std::int32_t {
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i s = _mm_add_epi32(lo, hi);
        __m128i shuf = _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2));
        s = _mm_add_epi32(s, shuf);
        shuf = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
        s = _mm_add_epi32(s, shuf);
        return _mm_cvtsi128_si32(s);
    };

    float dot = static_cast<float>(hsum_epi32(dot_sum));
    float a_mag = static_cast<float>(hsum_epi32(a_mag_sum));
    float b_mag = static_cast<float>(hsum_epi32(b_mag_sum));

    // Scalar tail
    while (i < size) {
        float av = static_cast<float>(a[i]);
        float bv = static_cast<float>(b[i]);
        dot += av * bv;
        a_mag += av * av;
        b_mag += bv * bv;
        ++i;
    }

    float denom = std::sqrt(a_mag) * std::sqrt(b_mag);
    if (denom < 1e-8f) {
        return 1.0f;
    }
    return 1.0f - (dot / denom);
}

/// AVX2-optimized inner product distance for int8 vectors
/// Returns 1 - dot(a,b) (consistent with other IP distance functions)
/// Requires: AVX2, size >= 16
inline float inner_product_int8_avx2(std::span<const std::int8_t> a,
                                      std::span<const std::int8_t> b) {
    const std::size_t size = a.size();
    const std::size_t end16 = (size / 16) * 16;

    __m256i dot_sum = _mm256_setzero_si256();

    std::size_t i = 0;
    while (i < end16) {
        __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
        __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
        __m256i a16 = _mm256_cvtepi8_epi16(a8);
        __m256i b16 = _mm256_cvtepi8_epi16(b8);

        dot_sum = _mm256_add_epi32(dot_sum, _mm256_madd_epi16(a16, b16));
        i += 16;
    }

    // Horizontal sum
    __m128i hi = _mm256_extracti128_si256(dot_sum, 1);
    __m128i lo = _mm256_castsi256_si128(dot_sum);
    __m128i s = _mm_add_epi32(lo, hi);
    __m128i shuf = _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2));
    s = _mm_add_epi32(s, shuf);
    shuf = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
    s = _mm_add_epi32(s, shuf);
    float dot = static_cast<float>(_mm_cvtsi128_si32(s));

    // Scalar tail
    while (i < size) {
        dot += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        ++i;
    }

    return 1.0f - dot;
}

} // namespace sqlite_vec_cpp::distances::simd

#endif // SQLITE_VEC_ENABLE_AVX2
