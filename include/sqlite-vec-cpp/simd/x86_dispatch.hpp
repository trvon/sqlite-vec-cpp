#pragma once

// Runtime-dispatched AVX2+FMA float kernels for portable x86-64 builds.
//
// SQLITE_VEC_ENABLE_AVX compiles AVX kernels into every including TU, which
// requires building the whole consumer for an AVX-capable CPU. Distributed
// binaries instead target baseline x86-64 and previously fell back to scalar
// loops. These kernels carry their own target attribute, so they compile in a
// baseline TU and are only called after a one-time CPUID check confirms
// AVX2 and FMA. Define SQLITE_VEC_DISABLE_X86_DISPATCH to opt out.

#if !defined(SQLITE_VEC_ENABLE_AVX) && !defined(SQLITE_VEC_DISABLE_X86_DISPATCH) &&                \
    (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define SQLITE_VEC_X86_RUNTIME_DISPATCH 1

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#define SQLITE_VEC_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))

namespace sqlite_vec_cpp::distances::x86 {

inline bool cpu_has_avx2_fma() noexcept {
    static const bool supported = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return supported;
}

SQLITE_VEC_TARGET_AVX2_FMA inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Four independent accumulators keep the FMA pipes busy; the tail is scalar.
SQLITE_VEC_TARGET_AVX2_FMA inline float dot_avx2(const float* a, const float* b,
                                                 std::size_t n) noexcept {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float sum = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

SQLITE_VEC_TARGET_AVX2_FMA inline float l2_squared_avx2(const float* a, const float* b,
                                                        std::size_t n) noexcept {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
    }
    for (; i + 8 <= n; i += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s0 = _mm256_fmadd_ps(d, d, s0);
    }
    float sum = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// Accumulates dot(a,b), |a|^2 and |b|^2 in one pass for cosine distance.
SQLITE_VEC_TARGET_AVX2_FMA inline void cosine_terms_avx2(const float* a, const float* b,
                                                         std::size_t n, float& dot, float& aa,
                                                         float& bb) noexcept {
    __m256 sd0 = _mm256_setzero_ps();
    __m256 sd1 = _mm256_setzero_ps();
    __m256 sa0 = _mm256_setzero_ps();
    __m256 sa1 = _mm256_setzero_ps();
    __m256 sb0 = _mm256_setzero_ps();
    __m256 sb1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 a0 = _mm256_loadu_ps(a + i);
        const __m256 b0 = _mm256_loadu_ps(b + i);
        const __m256 a1 = _mm256_loadu_ps(a + i + 8);
        const __m256 b1 = _mm256_loadu_ps(b + i + 8);
        sd0 = _mm256_fmadd_ps(a0, b0, sd0);
        sd1 = _mm256_fmadd_ps(a1, b1, sd1);
        sa0 = _mm256_fmadd_ps(a0, a0, sa0);
        sa1 = _mm256_fmadd_ps(a1, a1, sa1);
        sb0 = _mm256_fmadd_ps(b0, b0, sb0);
        sb1 = _mm256_fmadd_ps(b1, b1, sb1);
    }
    for (; i + 8 <= n; i += 8) {
        const __m256 a0 = _mm256_loadu_ps(a + i);
        const __m256 b0 = _mm256_loadu_ps(b + i);
        sd0 = _mm256_fmadd_ps(a0, b0, sd0);
        sa0 = _mm256_fmadd_ps(a0, a0, sa0);
        sb0 = _mm256_fmadd_ps(b0, b0, sb0);
    }
    dot = hsum256(_mm256_add_ps(sd0, sd1));
    aa = hsum256(_mm256_add_ps(sa0, sa1));
    bb = hsum256(_mm256_add_ps(sb0, sb1));
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
}

// LVQ-8 squared L2 distance in quantized space: sum_i ((query[i] - offset) * inv_scale - codes[i])^2
SQLITE_VEC_TARGET_AVX2_FMA inline float lvq8_l2_sq_avx2(const float* query,
                                                        const std::uint8_t* codes, std::size_t dim,
                                                        float inv_scale, float offset) noexcept {
    const __m256 voffset = _mm256_set1_ps(offset);
    const __m256 vinv_scale = _mm256_set1_ps(inv_scale);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m128i bytes0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i));
        const __m128i bytes1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i + 8));
        const __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes0));
        const __m256 c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes1));

        const __m256 q0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(query + i), voffset),
                                        vinv_scale);
        const __m256 q1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(query + i + 8), voffset),
                                        vinv_scale);

        const __m256 d0 = _mm256_sub_ps(q0, c0);
        const __m256 d1 = _mm256_sub_ps(q1, c1);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m128i bytes0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i));
        const __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes0));
        const __m256 q0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(query + i), voffset),
                                        vinv_scale);
        const __m256 d0 = _mm256_sub_ps(q0, c0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
    }

    float sum = hsum256(_mm256_add_ps(sum0, sum1));
    for (; i < dim; ++i) {
        const float q_normalized = (query[i] - offset) * inv_scale;
        const float diff = q_normalized - static_cast<float>(codes[i]);
        sum += diff * diff;
    }
    return sum;
}

// LVQ-4 squared L2 distance in quantized space (2 nibbles packed per byte)
SQLITE_VEC_TARGET_AVX2_FMA inline float lvq4_l2_sq_avx2(const float* query,
                                                        const std::uint8_t* codes, std::size_t dim,
                                                        float inv_scale, float offset) noexcept {
    const __m256 voffset = _mm256_set1_ps(offset);
    const __m256 vinv_scale = _mm256_set1_ps(inv_scale);
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i / 2));
        const __m128i lo = _mm_and_si128(packed, mask_0f);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);
        const __m128i unpacked = _mm_unpacklo_epi8(lo, hi);

        const __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(unpacked));
        const __m256 c1 =
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(unpacked, 8)));

        const __m256 q0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(query + i), voffset),
                                        vinv_scale);
        const __m256 q1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(query + i + 8), voffset),
                                        vinv_scale);

        const __m256 d0 = _mm256_sub_ps(q0, c0);
        const __m256 d1 = _mm256_sub_ps(q1, c1);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
    }

    float sum = hsum256(_mm256_add_ps(sum0, sum1));
    for (; i < dim; ++i) {
        const std::uint8_t c =
            (i % 2 == 0) ? (codes[i / 2] & 0x0F) : ((codes[i / 2] >> 4) & 0x0F);
        const float q_normalized = (query[i] - offset) * inv_scale;
        const float diff = q_normalized - static_cast<float>(c);
        sum += diff * diff;
    }
    return sum;
}

// Fused LVQ-8 squared L2 distance: sum_i ((query[i] - offset) - codes[i] * scale)^2
// Uses 3 FP ops per vector (sub, fnmadd, fmadd) instead of 4 and avoids inv_scale division.
SQLITE_VEC_TARGET_AVX2_FMA inline float lvq8_l2_sq_fused_avx2(const float* query,
                                                              const std::uint8_t* codes,
                                                              std::size_t dim, float scale,
                                                              float offset) noexcept {
    const __m256 voffset = _mm256_set1_ps(offset);
    const __m256 vscale = _mm256_set1_ps(scale);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m128i bytes0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i));
        const __m128i bytes1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i + 8));
        const __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes0));
        const __m256 c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes1));

        const __m256 q0 = _mm256_sub_ps(_mm256_loadu_ps(query + i), voffset);
        const __m256 q1 = _mm256_sub_ps(_mm256_loadu_ps(query + i + 8), voffset);

        const __m256 d0 = _mm256_fnmadd_ps(c0, vscale, q0);
        const __m256 d1 = _mm256_fnmadd_ps(c1, vscale, q1);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m128i bytes0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i));
        const __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes0));
        const __m256 q0 = _mm256_sub_ps(_mm256_loadu_ps(query + i), voffset);
        const __m256 d0 = _mm256_fnmadd_ps(c0, vscale, q0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
    }

    float sum = hsum256(_mm256_add_ps(sum0, sum1));
    for (; i < dim; ++i) {
        const float diff = (query[i] - offset) - static_cast<float>(codes[i]) * scale;
        sum += diff * diff;
    }
    return sum;
}

// Fused LVQ-4 squared L2 distance: sum_i ((query[i] - offset) - c[i] * scale)^2
SQLITE_VEC_TARGET_AVX2_FMA inline float lvq4_l2_sq_fused_avx2(const float* query,
                                                              const std::uint8_t* codes,
                                                              std::size_t dim, float scale,
                                                              float offset) noexcept {
    const __m256 voffset = _mm256_set1_ps(offset);
    const __m256 vscale = _mm256_set1_ps(scale);
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i / 2));
        const __m128i lo = _mm_and_si128(packed, mask_0f);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);
        const __m128i unpacked = _mm_unpacklo_epi8(lo, hi);

        const __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(unpacked));
        const __m256 c1 =
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(unpacked, 8)));

        const __m256 q0 = _mm256_sub_ps(_mm256_loadu_ps(query + i), voffset);
        const __m256 q1 = _mm256_sub_ps(_mm256_loadu_ps(query + i + 8), voffset);

        const __m256 d0 = _mm256_fnmadd_ps(c0, vscale, q0);
        const __m256 d1 = _mm256_fnmadd_ps(c1, vscale, q1);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
    }

    float sum = hsum256(_mm256_add_ps(sum0, sum1));
    for (; i < dim; ++i) {
        const std::uint8_t c =
            (i % 2 == 0) ? (codes[i / 2] & 0x0F) : ((codes[i / 2] >> 4) & 0x0F);
        const float diff = (query[i] - offset) - static_cast<float>(c) * scale;
        sum += diff * diff;
    }
    return sum;
}

} // namespace sqlite_vec_cpp::distances::x86

#endif
