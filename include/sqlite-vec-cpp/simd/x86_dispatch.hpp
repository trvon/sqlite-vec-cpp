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

} // namespace sqlite_vec_cpp::distances::x86

#endif
