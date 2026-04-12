#pragma once

/// Locally-adaptive Vector Quantization (LVQ)
///
/// Per-vector scalar quantization with per-vector scale and offset.
/// Based on "Similarity search in the blink of an eye with compressed indices"
/// (Aguerrebere et al., 2023).
///
/// Key insight: global quantization (one scale for all vectors) introduces large
/// errors. LVQ stores per-vector min/max and quantizes each vector independently,
/// yielding much better accuracy at the same compression ratio.
///
/// Supports LVQ-8 (8-bit, 4x compression) and LVQ-4 (4-bit, 8x compression).
/// Distance estimation is performed directly on quantized values with a fused
/// scale correction, avoiding decompression overhead.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#ifdef SQLITE_VEC_ENABLE_NEON
#include <arm_neon.h>
#endif

#ifdef SQLITE_VEC_ENABLE_AVX
#include <immintrin.h>
#endif

namespace sqlite_vec_cpp::quantization {

/// Quantized vector with per-vector scale and offset
/// Stores: quantized_values[], scale, offset
/// Reconstruction: original[i] ~= quantized[i] * scale + offset
template <typename CodeT> struct LVQCode {
    std::vector<CodeT> codes; ///< Quantized values
    float scale;              ///< Per-vector scale factor
    float offset;             ///< Per-vector offset (min value)

    /// Number of dimensions
    [[nodiscard]] size_t dimensions() const { return codes.size(); }
};

/// LVQ-8: 8-bit quantization (4x compression vs FP32)
/// Each dimension mapped to [0, 255] based on per-vector min/max
struct LVQ8 {
    using code_type = uint8_t;
    static constexpr size_t bits = 8;
    static constexpr float max_code = 255.0f;

    /// Quantize a float vector to 8-bit codes with per-vector scaling
    static LVQCode<uint8_t> encode(std::span<const float> vec) {
        assert(!vec.empty());

        float vmin = vec[0];
        float vmax = vec[0];
        for (size_t i = 1; i < vec.size(); ++i) {
            vmin = std::min(vmin, vec[i]);
            vmax = std::max(vmax, vec[i]);
        }

        float range = vmax - vmin;
        float scale = range > 0.0f ? range / max_code : 0.0f;
        float inv_scale = range > 0.0f ? max_code / range : 0.0f;

        LVQCode<uint8_t> result;
        result.codes.resize(vec.size());
        result.scale = scale;
        result.offset = vmin;

        for (size_t i = 0; i < vec.size(); ++i) {
            float normalized = (vec[i] - vmin) * inv_scale;
            result.codes[i] =
                static_cast<uint8_t>(std::clamp(std::round(normalized), 0.0f, max_code));
        }

        return result;
    }

    /// Decode quantized vector back to float (for verification)
    static std::vector<float> decode(const LVQCode<uint8_t>& code) {
        std::vector<float> result(code.codes.size());
        for (size_t i = 0; i < code.codes.size(); ++i) {
            result[i] = static_cast<float>(code.codes[i]) * code.scale + code.offset;
        }
        return result;
    }

    /// Estimate L2 distance between a float query and a quantized vector.
    ///
    /// Uses the identity:
    ///   ||q - x||^2 = sum((q[i] - (c[i]*scale + offset))^2)
    ///               = scale^2 * sum((q'[i] - c[i])^2) + correction terms
    ///
    /// where q'[i] = (q[i] - offset) / scale
    ///
    /// This formulation allows computing distance directly on quantized codes
    /// with a single scale correction at the end.
    static float l2_distance(std::span<const float> query, const LVQCode<uint8_t>& code) {
        assert(query.size() == code.codes.size());

        const float scale = code.scale;
        const float offset = code.offset;

        if (scale == 0.0f) {
            // All values are identical -- distance is just ||query - offset||^2
            float sum = 0.0f;
            for (size_t i = 0; i < query.size(); ++i) {
                float diff = query[i] - offset;
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }

        const float inv_scale = 1.0f / scale;
        const size_t dim = query.size();
        float sum_sq = 0.0f;

#ifdef SQLITE_VEC_ENABLE_NEON
        if (dim >= 16) {
            sum_sq = l2_distance_neon(query, code, inv_scale, offset);
        } else
#endif
#ifdef SQLITE_VEC_ENABLE_AVX
            if (dim >= 16) {
            sum_sq = l2_distance_avx(query, code, inv_scale, offset);
        } else
#endif
        {
            sum_sq = l2_distance_scalar(query, code, inv_scale, offset);
        }

        return std::sqrt(sum_sq * scale * scale);
    }

    /// Estimate inner product between float query and quantized vector
    /// ip(q, x) ~= scale * sum(q[i] * c[i]) + offset * sum(q[i])
    static float inner_product_distance(std::span<const float> query,
                                        const LVQCode<uint8_t>& code) {
        assert(query.size() == code.codes.size());

        float dot_qc = 0.0f;
        float sum_q = 0.0f;
        for (size_t i = 0; i < query.size(); ++i) {
            dot_qc += query[i] * static_cast<float>(code.codes[i]);
            sum_q += query[i];
        }

        return 1.0f - (code.scale * dot_qc + code.offset * sum_q);
    }

private:
    /// Scalar fallback for L2 distance computation
    static float l2_distance_scalar(std::span<const float> query, const LVQCode<uint8_t>& code,
                                    float inv_scale, float offset) {
        float sum = 0.0f;
        for (size_t i = 0; i < query.size(); ++i) {
            float q_normalized = (query[i] - offset) * inv_scale;
            float diff = q_normalized - static_cast<float>(code.codes[i]);
            sum += diff * diff;
        }
        return sum;
    }

#ifdef SQLITE_VEC_ENABLE_NEON
    /// NEON-optimized L2 distance: compute in normalized space, apply scale at end
    static float l2_distance_neon(std::span<const float> query, const LVQCode<uint8_t>& code,
                                  float inv_scale, float offset) {
        const size_t dim = query.size();
        const size_t end16 = (dim / 16) * 16;

        float32x4_t voffset = vdupq_n_f32(offset);
        float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        size_t i = 0;
        while (i < end16) {
            // Load 16 uint8 codes and widen to float
            uint8x16_t codes_u8 = vld1q_u8(&code.codes[i]);

            // Widen: u8 -> u16 -> u32 -> f32 (4 groups of 4)
            uint16x8_t lo16 = vmovl_u8(vget_low_u8(codes_u8));
            uint16x8_t hi16 = vmovl_u8(vget_high_u8(codes_u8));

            float32x4_t c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            float32x4_t c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
            float32x4_t c2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            float32x4_t c3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));

            // Load query values and normalize: q' = (q - offset) * inv_scale
            float32x4_t q0 = vmulq_f32(vsubq_f32(vld1q_f32(&query[i]), voffset), vinv_scale);
            float32x4_t q1 = vmulq_f32(vsubq_f32(vld1q_f32(&query[i + 4]), voffset), vinv_scale);
            float32x4_t q2 = vmulq_f32(vsubq_f32(vld1q_f32(&query[i + 8]), voffset), vinv_scale);
            float32x4_t q3 = vmulq_f32(vsubq_f32(vld1q_f32(&query[i + 12]), voffset), vinv_scale);

            // diff = q' - c, accumulate diff^2
            float32x4_t d0 = vsubq_f32(q0, c0);
            float32x4_t d1 = vsubq_f32(q1, c1);
            float32x4_t d2 = vsubq_f32(q2, c2);
            float32x4_t d3 = vsubq_f32(q3, c3);

            sum0 = vfmaq_f32(sum0, d0, d0);
            sum1 = vfmaq_f32(sum1, d1, d1);
            sum2 = vfmaq_f32(sum2, d2, d2);
            sum3 = vfmaq_f32(sum3, d3, d3);

            i += 16;
        }

        float32x4_t total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        float sum = vaddvq_f32(total);

        // Handle remaining elements
        while (i < dim) {
            float q_normalized = (query[i] - offset) * inv_scale;
            float diff = q_normalized - static_cast<float>(code.codes[i]);
            sum += diff * diff;
            ++i;
        }

        return sum;
    }
#endif

#ifdef SQLITE_VEC_ENABLE_AVX
    /// AVX-optimized L2 distance
    static float l2_distance_avx(std::span<const float> query, const LVQCode<uint8_t>& code,
                                 float inv_scale, float offset) {
        const size_t dim = query.size();
        const size_t end16 = (dim / 16) * 16;

        __m256 voffset = _mm256_set1_ps(offset);
        __m256 vinv_scale = _mm256_set1_ps(inv_scale);
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();

        size_t i = 0;
        while (i < end16) {
            // Load 16 uint8 codes
            // Process first 8
            __m128i codes8_lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&code.codes[i]));
            __m256i codes32_lo = _mm256_cvtepu8_epi32(codes8_lo);
            __m256 cf_lo = _mm256_cvtepi32_ps(codes32_lo);

            __m256 q_lo = _mm256_loadu_ps(&query[i]);
            __m256 qn_lo = _mm256_mul_ps(_mm256_sub_ps(q_lo, voffset), vinv_scale);
            __m256 d_lo = _mm256_sub_ps(qn_lo, cf_lo);
            sum0 = _mm256_fmadd_ps(d_lo, d_lo, sum0);

            // Process next 8
            __m128i codes8_hi =
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&code.codes[i + 8]));
            __m256i codes32_hi = _mm256_cvtepu8_epi32(codes8_hi);
            __m256 cf_hi = _mm256_cvtepi32_ps(codes32_hi);

            __m256 q_hi = _mm256_loadu_ps(&query[i + 8]);
            __m256 qn_hi = _mm256_mul_ps(_mm256_sub_ps(q_hi, voffset), vinv_scale);
            __m256 d_hi = _mm256_sub_ps(qn_hi, cf_hi);
            sum1 = _mm256_fmadd_ps(d_hi, d_hi, sum1);

            i += 16;
        }

        __m256 total = _mm256_add_ps(sum0, sum1);
        // Horizontal sum
        __m128 hi128 = _mm256_extractf128_ps(total, 1);
        __m128 lo128 = _mm256_castps256_ps128(total);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        // Handle remaining elements
        while (i < dim) {
            float q_normalized = (query[i] - offset) * inv_scale;
            float diff = q_normalized - static_cast<float>(code.codes[i]);
            sum += diff * diff;
            ++i;
        }

        return sum;
    }
#endif
};

/// LVQ-4: 4-bit quantization (8x compression vs FP32)
/// Each dimension mapped to [0, 15]. Two dimensions packed into one byte.
struct LVQ4 {
    using code_type = uint8_t; // Two 4-bit codes packed per byte
    static constexpr size_t bits = 4;
    static constexpr float max_code = 15.0f;

    /// Quantize a float vector to 4-bit codes with per-vector scaling
    /// Codes are packed: byte[i] = (code[2i+1] << 4) | code[2i]
    static LVQCode<uint8_t> encode(std::span<const float> vec) {
        assert(!vec.empty());

        float vmin = vec[0];
        float vmax = vec[0];
        for (size_t i = 1; i < vec.size(); ++i) {
            vmin = std::min(vmin, vec[i]);
            vmax = std::max(vmax, vec[i]);
        }

        float range = vmax - vmin;
        float scale = range > 0.0f ? range / max_code : 0.0f;
        float inv_scale = range > 0.0f ? max_code / range : 0.0f;

        // Pack two 4-bit codes per byte
        size_t packed_size = (vec.size() + 1) / 2;
        LVQCode<uint8_t> result;
        result.codes.resize(packed_size, 0);
        result.scale = scale;
        result.offset = vmin;

        for (size_t i = 0; i < vec.size(); ++i) {
            float normalized = (vec[i] - vmin) * inv_scale;
            uint8_t code = static_cast<uint8_t>(std::clamp(std::round(normalized), 0.0f, max_code));
            if (i % 2 == 0) {
                result.codes[i / 2] = code;
            } else {
                result.codes[i / 2] |= (code << 4);
            }
        }

        return result;
    }

    /// Decode 4-bit packed codes back to float
    static std::vector<float> decode(const LVQCode<uint8_t>& code, size_t original_dim) {
        std::vector<float> result(original_dim);
        for (size_t i = 0; i < original_dim; ++i) {
            uint8_t c;
            if (i % 2 == 0) {
                c = code.codes[i / 2] & 0x0F;
            } else {
                c = (code.codes[i / 2] >> 4) & 0x0F;
            }
            result[i] = static_cast<float>(c) * code.scale + code.offset;
        }
        return result;
    }

    /// Estimate L2 distance between float query and 4-bit quantized vector
    static float l2_distance(std::span<const float> query, const LVQCode<uint8_t>& code,
                             size_t original_dim) {
        assert(query.size() == original_dim);

        const float scale = code.scale;
        const float offset = code.offset;

        if (scale == 0.0f) {
            float sum = 0.0f;
            for (size_t i = 0; i < query.size(); ++i) {
                float diff = query[i] - offset;
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }

        const float inv_scale = 1.0f / scale;
        float sum_sq = 0.0f;

#ifdef SQLITE_VEC_ENABLE_NEON
        if (original_dim >= 32) {
            sum_sq =
                l2_distance_neon(query.data(), code.codes.data(), original_dim, inv_scale, offset);
        } else
#endif
#ifdef SQLITE_VEC_ENABLE_AVX2
        if (original_dim >= 32) {
            sum_sq = l2_distance_avx2(query.data(), code.codes.data(), original_dim, inv_scale,
                                       offset);
        } else
#endif
        {
            sum_sq = l2_distance_scalar(query.data(), code.codes.data(), original_dim, inv_scale,
                                        offset);
        }

        return std::sqrt(sum_sq * scale * scale);
    }

private:
    static float l2_distance_scalar(const float* query, const uint8_t* codes, size_t dim,
                                    float inv_scale, float offset) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            uint8_t c = (i % 2 == 0) ? (codes[i / 2] & 0x0F) : ((codes[i / 2] >> 4) & 0x0F);
            float q_normalized = (query[i] - offset) * inv_scale;
            float diff = q_normalized - static_cast<float>(c);
            sum += diff * diff;
        }
        return sum;
    }

#ifdef SQLITE_VEC_ENABLE_NEON
    /// NEON LVQ-4 L2 distance: process 32 dims (16 packed bytes) per iteration
    static float l2_distance_neon(const float* query, const uint8_t* codes, size_t dim,
                                  float inv_scale, float offset) {
        const size_t end32 = (dim / 32) * 32;

        float32x4_t voffset = vdupq_n_f32(offset);
        float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
        uint8x16_t mask_lo = vdupq_n_u8(0x0F);
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        size_t i = 0;
        while (i < end32) {
            // Load 16 packed bytes = 32 nibbles = 32 dimensions
            uint8x16_t packed = vld1q_u8(codes + i / 2);

            // Extract low nibbles (even dims) and high nibbles (odd dims)
            uint8x16_t lo_nibbles = vandq_u8(packed, mask_lo);
            uint8x16_t hi_nibbles = vshrq_n_u8(packed, 4);

            // Interleave to natural dimension order:
            // zip1(lo,hi) = lo[0],hi[0],lo[1],hi[1],...,lo[7],hi[7] = dims 0..15
            // zip2(lo,hi) = lo[8],hi[8],lo[9],hi[9],...,lo[15],hi[15] = dims 16..31
            uint8x16_t unpacked_0_15 = vzip1q_u8(lo_nibbles, hi_nibbles);
            uint8x16_t unpacked_16_31 = vzip2q_u8(lo_nibbles, hi_nibbles);

            // === Process dims i..i+15 from unpacked_0_15 ===
            uint16x8_t a_lo16 = vmovl_u8(vget_low_u8(unpacked_0_15));
            uint16x8_t a_hi16 = vmovl_u8(vget_high_u8(unpacked_0_15));

            float32x4_t c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo16)));
            float32x4_t c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_lo16)));
            float32x4_t c2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_hi16)));
            float32x4_t c3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_hi16)));

            float32x4_t q0 = vmulq_f32(vsubq_f32(vld1q_f32(query + i), voffset), vinv_scale);
            float32x4_t q1 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 4), voffset), vinv_scale);
            float32x4_t q2 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 8), voffset), vinv_scale);
            float32x4_t q3 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 12), voffset), vinv_scale);

            float32x4_t d0 = vsubq_f32(q0, c0);
            float32x4_t d1 = vsubq_f32(q1, c1);
            float32x4_t d2 = vsubq_f32(q2, c2);
            float32x4_t d3 = vsubq_f32(q3, c3);

            sum0 = vfmaq_f32(sum0, d0, d0);
            sum1 = vfmaq_f32(sum1, d1, d1);
            sum2 = vfmaq_f32(sum2, d2, d2);
            sum3 = vfmaq_f32(sum3, d3, d3);

            // === Process dims i+16..i+31 from unpacked_16_31 ===
            uint16x8_t b_lo16 = vmovl_u8(vget_low_u8(unpacked_16_31));
            uint16x8_t b_hi16 = vmovl_u8(vget_high_u8(unpacked_16_31));

            c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_lo16)));
            c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_lo16)));
            c2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_hi16)));
            c3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_hi16)));

            q0 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 16), voffset), vinv_scale);
            q1 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 20), voffset), vinv_scale);
            q2 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 24), voffset), vinv_scale);
            q3 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 28), voffset), vinv_scale);

            d0 = vsubq_f32(q0, c0);
            d1 = vsubq_f32(q1, c1);
            d2 = vsubq_f32(q2, c2);
            d3 = vsubq_f32(q3, c3);

            sum0 = vfmaq_f32(sum0, d0, d0);
            sum1 = vfmaq_f32(sum1, d1, d1);
            sum2 = vfmaq_f32(sum2, d2, d2);
            sum3 = vfmaq_f32(sum3, d3, d3);

            i += 32;
        }

        float32x4_t total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        float sum = vaddvq_f32(total);

        // Scalar tail for remaining dims
        while (i < dim) {
            uint8_t c = (i % 2 == 0) ? (codes[i / 2] & 0x0F) : ((codes[i / 2] >> 4) & 0x0F);
            float q_normalized = (query[i] - offset) * inv_scale;
            float diff = q_normalized - static_cast<float>(c);
            sum += diff * diff;
            ++i;
        }

        return sum;
    }
#endif

#ifdef SQLITE_VEC_ENABLE_AVX2
    /// AVX2 LVQ-4 L2 distance: process 32 dims (16 packed bytes) per iteration
    static float l2_distance_avx2(const float* query, const uint8_t* codes, size_t dim,
                                  float inv_scale, float offset) {
        const size_t end32 = (dim / 32) * 32;

        __m256 voffset = _mm256_set1_ps(offset);
        __m256 vinv_scale = _mm256_set1_ps(inv_scale);
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();

        // Nibble mask: 0x0F repeated
        __m128i mask_lo = _mm_set1_epi8(0x0F);

        size_t i = 0;
        while (i < end32) {
            // Load 16 packed bytes = 32 nibbles = 32 dimensions
            __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i / 2));

            // Extract low nibbles (even dims) and high nibbles (odd dims)
            __m128i lo_nibbles = _mm_and_si128(packed, mask_lo);
            __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_lo);

            // Interleave to natural order: lo[0],hi[0],lo[1],hi[1],...
            __m128i unpacked_0_15 = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
            __m128i unpacked_16_31 = _mm_unpackhi_epi8(lo_nibbles, hi_nibbles);

            // === Process dims i..i+15 ===
            // Widen first 8 bytes to int32 then float
            __m256i codes32_a = _mm256_cvtepu8_epi32(unpacked_0_15);
            __m256 cf_a = _mm256_cvtepi32_ps(codes32_a);
            __m256 q_a = _mm256_loadu_ps(query + i);
            __m256 qn_a = _mm256_mul_ps(_mm256_sub_ps(q_a, voffset), vinv_scale);
            __m256 d_a = _mm256_sub_ps(qn_a, cf_a);
            sum0 = _mm256_fmadd_ps(d_a, d_a, sum0);

            // Widen next 8 bytes
            __m128i upper_0_15 = _mm_srli_si128(unpacked_0_15, 8);
            __m256i codes32_b = _mm256_cvtepu8_epi32(upper_0_15);
            __m256 cf_b = _mm256_cvtepi32_ps(codes32_b);
            __m256 q_b = _mm256_loadu_ps(query + i + 8);
            __m256 qn_b = _mm256_mul_ps(_mm256_sub_ps(q_b, voffset), vinv_scale);
            __m256 d_b = _mm256_sub_ps(qn_b, cf_b);
            sum1 = _mm256_fmadd_ps(d_b, d_b, sum1);

            // === Process dims i+16..i+31 ===
            __m256i codes32_c = _mm256_cvtepu8_epi32(unpacked_16_31);
            __m256 cf_c = _mm256_cvtepi32_ps(codes32_c);
            __m256 q_c = _mm256_loadu_ps(query + i + 16);
            __m256 qn_c = _mm256_mul_ps(_mm256_sub_ps(q_c, voffset), vinv_scale);
            __m256 d_c = _mm256_sub_ps(qn_c, cf_c);
            sum0 = _mm256_fmadd_ps(d_c, d_c, sum0);

            __m128i upper_16_31 = _mm_srli_si128(unpacked_16_31, 8);
            __m256i codes32_d = _mm256_cvtepu8_epi32(upper_16_31);
            __m256 cf_d = _mm256_cvtepi32_ps(codes32_d);
            __m256 q_d = _mm256_loadu_ps(query + i + 24);
            __m256 qn_d = _mm256_mul_ps(_mm256_sub_ps(q_d, voffset), vinv_scale);
            __m256 d_d = _mm256_sub_ps(qn_d, cf_d);
            sum1 = _mm256_fmadd_ps(d_d, d_d, sum1);

            i += 32;
        }

        __m256 total = _mm256_add_ps(sum0, sum1);
        __m128 hi128 = _mm256_extractf128_ps(total, 1);
        __m128 lo128 = _mm256_castps256_ps128(total);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        // Scalar tail
        while (i < dim) {
            uint8_t c = (i % 2 == 0) ? (codes[i / 2] & 0x0F) : ((codes[i / 2] >> 4) & 0x0F);
            float q_normalized = (query[i] - offset) * inv_scale;
            float diff = q_normalized - static_cast<float>(c);
            sum += diff * diff;
            ++i;
        }

        return sum;
    }
#endif
};

/// Batch quantizer: quantizes a collection of vectors
struct LVQBatchEncoder {
    /// Quantize all vectors in a corpus to LVQ-8
    static std::vector<LVQCode<uint8_t>>
    encode_all_lvq8(std::span<const std::span<const float>> vectors) {
        std::vector<LVQCode<uint8_t>> result;
        result.reserve(vectors.size());
        for (const auto& vec : vectors) {
            result.push_back(LVQ8::encode(vec));
        }
        return result;
    }

    /// Quantize all vectors in a corpus to LVQ-4
    static std::vector<LVQCode<uint8_t>>
    encode_all_lvq4(std::span<const std::span<const float>> vectors) {
        std::vector<LVQCode<uint8_t>> result;
        result.reserve(vectors.size());
        for (const auto& vec : vectors) {
            result.push_back(LVQ4::encode(vec));
        }
        return result;
    }
};

} // namespace sqlite_vec_cpp::quantization
