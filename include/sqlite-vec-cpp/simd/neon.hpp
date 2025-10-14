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
    while (i + 63 < size) {
        int8x16_t v1 = vld1q_s8(&a[i]);
        int8x16_t v2 = vld1q_s8(&b[i]);
        uint8x16_t diff1 = vabdq_s8(v1, v2);
        acc1 = vaddq_s32(acc1, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff1))));

        v1 = vld1q_s8(&a[i + 16]);
        v2 = vld1q_s8(&b[i + 16]);
        uint8x16_t diff2 = vabdq_s8(v1, v2);
        acc2 = vaddq_s32(acc2, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff2))));

        v1 = vld1q_s8(&a[i + 32]);
        v2 = vld1q_s8(&b[i + 32]);
        uint8x16_t diff3 = vabdq_s8(v1, v2);
        acc3 = vaddq_s32(acc3, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff3))));

        v1 = vld1q_s8(&a[i + 48]);
        v2 = vld1q_s8(&b[i + 48]);
        uint8x16_t diff4 = vabdq_s8(v1, v2);
        acc4 = vaddq_s32(acc4, vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(diff4))));

        i += 64;
    }

    // Process 16 elements at a time
    while (i + 15 < size) {
        int8x16_t v1 = vld1q_s8(&a[i]);
        int8x16_t v2 = vld1q_s8(&b[i]);
        uint8x16_t diff = vabdq_s8(v1, v2);
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

} // namespace sqlite_vec_cpp::distances::simd

#endif // SQLITE_VEC_ENABLE_NEON
