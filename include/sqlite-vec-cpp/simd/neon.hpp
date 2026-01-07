#pragma once

#ifdef SQLITE_VEC_ENABLE_NEON

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <span>

namespace sqlite_vec_cpp::distances::simd {

/// NEON-optimized L2 distance for float vectors
/// Requires: NEON support, size > 16
inline float l2_distance_float_neon(std::span<const float> a, std::span<const float> b) {
    const std::size_t size = a.size();
    const std::size_t qty16 = size >> 4;    // size / 16
    const std::size_t end_idx = qty16 << 4; // (size / 16) * 16

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    while (i < end_idx) {
        // Process 4 sets of 4 floats = 16 floats per iteration
        float32x4_t v1 = vld1q_f32(&a[i]);
        float32x4_t v2 = vld1q_f32(&b[i]);
        float32x4_t diff = vsubq_f32(v1, v2);
        sum0 = vfmaq_f32(sum0, diff, diff); // fused multiply-add
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        diff = vsubq_f32(v1, v2);
        sum1 = vfmaq_f32(sum1, diff, diff);
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        diff = vsubq_f32(v1, v2);
        sum2 = vfmaq_f32(sum2, diff, diff);
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        diff = vsubq_f32(v1, v2);
        sum3 = vfmaq_f32(sum3, diff, diff);
        i += 4;
    }

    // Reduce sums
    float32x4_t sum_total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    float scalar_sum = vaddvq_f32(sum_total);

    // Handle remaining elements
    while (i < size) {
        float diff = a[i] - b[i];
        scalar_sum += diff * diff;
        ++i;
    }

    return std::sqrt(scalar_sum);
}

/// NEON-optimized L2 distance for int8 vectors
/// Requires: NEON support, size > 7
inline float l2_distance_int8_neon(std::span<const std::int8_t> a, std::span<const std::int8_t> b) {
    const std::size_t size = a.size();
    std::int32_t sum_scalar = 0;
    std::size_t i = 0;

    // Process 8 int8s at a time
    while (i + 7 < size) {
        int8x8_t v1 = vld1_s8(&a[i]);
        int8x8_t v2 = vld1_s8(&b[i]);

        // Widen to int16 to avoid overflow
        int16x8_t v1_wide = vmovl_s8(v1);
        int16x8_t v2_wide = vmovl_s8(v2);

        int16x8_t diff = vsubq_s16(v1_wide, v2_wide);
        int16x8_t squared_diff = vmulq_s16(diff, diff);

        // Pairwise add to int32
        int32x4_t sum = vpaddlq_s16(squared_diff);

        // Extract lanes and add to scalar sum
        sum_scalar += vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1) + vgetq_lane_s32(sum, 2) +
                      vgetq_lane_s32(sum, 3);

        i += 8;
    }

    // Handle remaining elements
    while (i < size) {
        std::int16_t diff = static_cast<std::int16_t>(a[i]) - static_cast<std::int16_t>(b[i]);
        sum_scalar += diff * diff;
        ++i;
    }

    return std::sqrt(static_cast<float>(sum_scalar));
}

/// NEON-optimized L1 distance for int8 vectors
/// Requires: NEON support, size > 15
inline std::int32_t l1_distance_int8_neon(std::span<const std::int8_t> a,
                                          std::span<const std::int8_t> b) {
    const std::size_t size = a.size();

    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);
    int32x4_t acc4 = vdupq_n_s32(0);

    std::size_t i = 0;

    // Process 64 elements at a time (4 × 16)
    // Note: vabdq_s8 returns int8x16_t; reinterpret as uint8x16_t since abs diff is non-negative
    while (i + 63 < size) {
        int8x16_t v1 = vld1q_s8(&a[i]);
        int8x16_t v2 = vld1q_s8(&b[i]);
        uint8x16_t diff1 = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
        acc1 = vaddq_s32(acc1, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff1))));

        v1 = vld1q_s8(&a[i + 16]);
        v2 = vld1q_s8(&b[i + 16]);
        uint8x16_t diff2 = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
        acc2 = vaddq_s32(acc2, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff2))));

        v1 = vld1q_s8(&a[i + 32]);
        v2 = vld1q_s8(&b[i + 32]);
        uint8x16_t diff3 = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
        acc3 = vaddq_s32(acc3, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff3))));

        v1 = vld1q_s8(&a[i + 48]);
        v2 = vld1q_s8(&b[i + 48]);
        uint8x16_t diff4 = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
        acc4 = vaddq_s32(acc4, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff4))));

        i += 64;
    }

    // Process 16 elements at a time
    while (i + 15 < size) {
        int8x16_t v1 = vld1q_s8(&a[i]);
        int8x16_t v2 = vld1q_s8(&b[i]);
        uint8x16_t diff = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
        acc1 = vaddq_s32(acc1, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff))));
        i += 16;
    }

    // Reduce accumulators
    int32x4_t acc = vaddq_s32(vaddq_s32(acc1, acc2), vaddq_s32(acc3, acc4));
    std::int32_t sum = vaddvq_s32(acc);

    // Handle remaining elements
    while (i < size) {
        sum += std::abs(static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]));
        ++i;
    }

    return sum;
}

/// NEON-optimized L1 distance for float vectors
/// Requires: NEON support, size > 3
inline double l1_distance_float_neon(std::span<const float> a, std::span<const float> b) {
    const std::size_t size = a.size();
    float64x2_t acc = vdupq_n_f64(0.0);
    std::size_t i = 0;

    // Process 4 floats at a time
    while (i + 3 < size) {
        float32x4_t v1 = vld1q_f32(&a[i]);
        float32x4_t v2 = vld1q_f32(&b[i]);

        // Convert to f64 for better precision
        float64x2_t low_diff =
            vabdq_f64(vcvt_f64_f32(vget_low_f32(v1)), vcvt_f64_f32(vget_low_f32(v2)));
        float64x2_t high_diff =
            vabdq_f64(vcvt_f64_f32(vget_high_f32(v1)), vcvt_f64_f32(vget_high_f32(v2)));

        acc = vaddq_f64(acc, vaddq_f64(low_diff, high_diff));
        i += 4;
    }

    // Reduce accumulator
    double sum = vaddvq_f64(acc);

    // Handle remaining elements
    while (i < size) {
        sum += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        ++i;
    }

    return sum;
}

/// NEON-optimized cosine distance for float vectors
/// cosine_distance = 1 - (dot(a,b) / (||a|| * ||b||))
/// Requires: NEON support, size >= 4
inline float cosine_distance_float_neon(std::span<const float> a, std::span<const float> b) {
    const std::size_t size = a.size();
    const std::size_t qty16 = size >> 4;    // size / 16
    const std::size_t end_idx = qty16 << 4; // (size / 16) * 16

    float32x4_t dot0 = vdupq_n_f32(0.0f);
    float32x4_t dot1 = vdupq_n_f32(0.0f);
    float32x4_t dot2 = vdupq_n_f32(0.0f);
    float32x4_t dot3 = vdupq_n_f32(0.0f);

    float32x4_t a_mag0 = vdupq_n_f32(0.0f);
    float32x4_t a_mag1 = vdupq_n_f32(0.0f);
    float32x4_t a_mag2 = vdupq_n_f32(0.0f);
    float32x4_t a_mag3 = vdupq_n_f32(0.0f);

    float32x4_t b_mag0 = vdupq_n_f32(0.0f);
    float32x4_t b_mag1 = vdupq_n_f32(0.0f);
    float32x4_t b_mag2 = vdupq_n_f32(0.0f);
    float32x4_t b_mag3 = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    while (i < end_idx) {
        // Process 4 sets of 4 floats = 16 floats per iteration
        float32x4_t v1 = vld1q_f32(&a[i]);
        float32x4_t v2 = vld1q_f32(&b[i]);
        dot0 = vfmaq_f32(dot0, v1, v2);
        a_mag0 = vfmaq_f32(a_mag0, v1, v1);
        b_mag0 = vfmaq_f32(b_mag0, v2, v2);
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        dot1 = vfmaq_f32(dot1, v1, v2);
        a_mag1 = vfmaq_f32(a_mag1, v1, v1);
        b_mag1 = vfmaq_f32(b_mag1, v2, v2);
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        dot2 = vfmaq_f32(dot2, v1, v2);
        a_mag2 = vfmaq_f32(a_mag2, v1, v1);
        b_mag2 = vfmaq_f32(b_mag2, v2, v2);
        i += 4;

        v1 = vld1q_f32(&a[i]);
        v2 = vld1q_f32(&b[i]);
        dot3 = vfmaq_f32(dot3, v1, v2);
        a_mag3 = vfmaq_f32(a_mag3, v1, v1);
        b_mag3 = vfmaq_f32(b_mag3, v2, v2);
        i += 4;
    }

    // Reduce sums
    float32x4_t dot_total = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    float32x4_t a_mag_total = vaddq_f32(vaddq_f32(a_mag0, a_mag1), vaddq_f32(a_mag2, a_mag3));
    float32x4_t b_mag_total = vaddq_f32(vaddq_f32(b_mag0, b_mag1), vaddq_f32(b_mag2, b_mag3));

    float dot = vaddvq_f32(dot_total);
    float a_mag = vaddvq_f32(a_mag_total);
    float b_mag = vaddvq_f32(b_mag_total);

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

/// NEON-optimized dot product for int8 vectors using DotProd instruction
/// Available on ARMv8.2+ (Apple M1, Cortex-A75+, etc.)
/// Processes 16 int8 values per iteration with fused multiply-accumulate
/// @param a First int8 vector
/// @param b Second int8 vector
/// @return Dot product as int32
#ifdef __ARM_FEATURE_DOTPROD
inline std::int32_t dot_product_int8_neon_dotprod(std::span<const std::int8_t> a,
                                                  std::span<const std::int8_t> b) {
    const std::size_t size = a.size();
    const std::size_t qty64 = size >> 6;  // size / 64
    const std::size_t end64 = qty64 << 6; // Aligned to 64 bytes

    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    std::size_t i = 0;

    // Process 64 int8s per iteration (4 × 16)
    while (i < end64) {
        // Load 16 int8s and compute dot product with vdotq_s32
        int8x16_t v1_0 = vld1q_s8(&a[i]);
        int8x16_t v2_0 = vld1q_s8(&b[i]);
        acc0 = vdotq_s32(acc0, v1_0, v2_0);

        int8x16_t v1_1 = vld1q_s8(&a[i + 16]);
        int8x16_t v2_1 = vld1q_s8(&b[i + 16]);
        acc1 = vdotq_s32(acc1, v1_1, v2_1);

        int8x16_t v1_2 = vld1q_s8(&a[i + 32]);
        int8x16_t v2_2 = vld1q_s8(&b[i + 32]);
        acc2 = vdotq_s32(acc2, v1_2, v2_2);

        int8x16_t v1_3 = vld1q_s8(&a[i + 48]);
        int8x16_t v2_3 = vld1q_s8(&b[i + 48]);
        acc3 = vdotq_s32(acc3, v1_3, v2_3);

        i += 64;
    }

    // Process remaining 16-element chunks
    while (i + 15 < size) {
        int8x16_t v1 = vld1q_s8(&a[i]);
        int8x16_t v2 = vld1q_s8(&b[i]);
        acc0 = vdotq_s32(acc0, v1, v2);
        i += 16;
    }

    // Reduce accumulators
    int32x4_t acc_total = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
    std::int32_t sum = vaddvq_s32(acc_total);

    // Handle remaining elements (scalar)
    while (i < size) {
        sum += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
        ++i;
    }

    return sum;
}

/// NEON-optimized cosine distance for int8 vectors using DotProd instruction
/// Uses the fast dot product path for computing dot(a,b), ||a||², ||b||²
/// @param a First int8 vector
/// @param b Second int8 vector
/// @return Cosine distance (1 - cosine_similarity)
inline float cosine_distance_int8_neon_dotprod(std::span<const std::int8_t> a,
                                               std::span<const std::int8_t> b) {
    const std::size_t size = a.size();
    const std::size_t qty16 = size >> 4;  // size / 16
    const std::size_t end16 = qty16 << 4; // Aligned to 16 bytes

    int32x4_t dot_acc = vdupq_n_s32(0);
    int32x4_t a_mag_acc = vdupq_n_s32(0);
    int32x4_t b_mag_acc = vdupq_n_s32(0);

    std::size_t i = 0;

    // Process 16 int8s per iteration
    while (i < end16) {
        int8x16_t v1 = vld1q_s8(&a[i]);
        int8x16_t v2 = vld1q_s8(&b[i]);

        // Compute dot products: a·b, a·a, b·b
        dot_acc = vdotq_s32(dot_acc, v1, v2);
        a_mag_acc = vdotq_s32(a_mag_acc, v1, v1);
        b_mag_acc = vdotq_s32(b_mag_acc, v2, v2);

        i += 16;
    }

    // Reduce accumulators
    std::int32_t dot = vaddvq_s32(dot_acc);
    std::int32_t a_mag = vaddvq_s32(a_mag_acc);
    std::int32_t b_mag = vaddvq_s32(b_mag_acc);

    // Handle remaining elements (scalar)
    while (i < size) {
        std::int32_t ai = static_cast<std::int32_t>(a[i]);
        std::int32_t bi = static_cast<std::int32_t>(b[i]);
        dot += ai * bi;
        a_mag += ai * ai;
        b_mag += bi * bi;
        ++i;
    }

    // Compute cosine distance
    float denom = std::sqrt(static_cast<float>(a_mag)) * std::sqrt(static_cast<float>(b_mag));
    if (denom < 1e-8f) {
        return 1.0f; // Avoid division by zero
    }
    return 1.0f - (static_cast<float>(dot) / denom);
}
#endif // __ARM_FEATURE_DOTPROD

} // namespace sqlite_vec_cpp::distances::simd

#endif // SQLITE_VEC_ENABLE_NEON
