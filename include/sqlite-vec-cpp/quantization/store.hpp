#pragma once

/// QuantizedStore: flat, dense_id-indexed quantized code storage
///
/// Replaces the unordered_map<node_id, Code> pattern with a contiguous
/// vector<Code> indexed by dense_id. This gives:
///   - O(1) lookup (vs O(1) amortized with hash overhead)
///   - Cache-friendly sequential access during beam search
///   - Zero allocation per-query
///
/// The store is built once after index construction, then used for
/// all subsequent searches.

#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "../index/quantization_snapshot.hpp"
#include "lvq.hpp"
#include "rabitq.hpp"

namespace sqlite_vec_cpp::quantization {

/// Flat storage for LVQ-8 codes indexed by dense_id
struct LVQ8Store {
    /// Contiguous code storage: codes[dense_id * dim .. (dense_id+1) * dim]
    std::vector<uint8_t> codes;
    /// Per-vector scale factors: scales[dense_id]
    std::vector<float> scales;
    /// Per-vector offsets: offsets[dense_id]
    std::vector<float> offsets;
    /// Dimensionality
    size_t dim = 0;
    /// Number of vectors stored
    size_t count = 0;

    /// Build from a locked quantization snapshot (preferred — no live graph iteration)
    void build(const index::QuantizationSnapshot& snap) {
        if (snap.entries.empty())
            return;

        dim = snap.dim;
        size_t max_dense_id = 0;
        for (const auto& e : snap.entries)
            max_dense_id = std::max(max_dense_id, e.dense_id);

        count = max_dense_id + 1;
        codes.resize(count * dim, 0);
        scales.resize(count, 0.0f);
        offsets.resize(count, 0.0f);

        for (const auto& e : snap.entries) {
            auto code = LVQ8::encode(std::span<const float>(e.vector));
            size_t offset = e.dense_id * dim;
            std::copy(code.codes.begin(), code.codes.end(), codes.begin() + offset);
            scales[e.dense_id] = code.scale;
            offsets[e.dense_id] = code.offset;
        }
    }

    /// Estimate L2 distance between float query and quantized vector at dense_id
    [[nodiscard]] float l2_distance(std::span<const float> query, size_t dense_id) const {
        assert(dense_id < count);
        assert(query.size() == dim);

        const float scale = scales[dense_id];
        const float offset = offsets[dense_id];
        const uint8_t* code_ptr = codes.data() + dense_id * dim;

        if (scale == 0.0f) {
            float sum = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float diff = query[i] - offset;
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }

        const float inv_scale = 1.0f / scale;
        float sum_sq = l2_sq_impl(query.data(), code_ptr, dim, inv_scale, offset);
        return std::sqrt(sum_sq * scale * scale);
    }

    /// Prefetch the quantized code for a given dense_id into L1 cache
    void prefetch(size_t dense_id) const {
        if (dense_id < count) {
            const void* addr = codes.data() + dense_id * dim;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(addr, 0, 3);
#endif
        }
    }

    /// Memory usage in bytes
    [[nodiscard]] size_t memory_bytes() const {
        return codes.size() + scales.size() * sizeof(float) + offsets.size() * sizeof(float);
    }

private:
    static float l2_sq_impl(const float* query, const uint8_t* codes, size_t dim, float inv_scale,
                            float offset) {
#ifdef SQLITE_VEC_ENABLE_NEON
        if (dim >= 16) {
            return l2_sq_neon(query, codes, dim, inv_scale, offset);
        }
#endif
        return l2_sq_scalar(query, codes, dim, inv_scale, offset);
    }

    static float l2_sq_scalar(const float* query, const uint8_t* codes, size_t dim, float inv_scale,
                              float offset) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float q_normalized = (query[i] - offset) * inv_scale;
            float diff = q_normalized - static_cast<float>(codes[i]);
            sum += diff * diff;
        }
        return sum;
    }

#ifdef SQLITE_VEC_ENABLE_NEON
    static float l2_sq_neon(const float* query, const uint8_t* codes, size_t dim, float inv_scale,
                            float offset) {
        const size_t end16 = (dim / 16) * 16;

        float32x4_t voffset = vdupq_n_f32(offset);
        float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        size_t i = 0;
        while (i < end16) {
            uint8x16_t codes_u8 = vld1q_u8(codes + i);

            uint16x8_t lo16 = vmovl_u8(vget_low_u8(codes_u8));
            uint16x8_t hi16 = vmovl_u8(vget_high_u8(codes_u8));

            float32x4_t c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            float32x4_t c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
            float32x4_t c2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            float32x4_t c3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));

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

            i += 16;
        }

        float32x4_t total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        float sum = vaddvq_f32(total);

        while (i < dim) {
            float q_normalized = (query[i] - offset) * inv_scale;
            float diff = q_normalized - static_cast<float>(codes[i]);
            sum += diff * diff;
            ++i;
        }

        return sum;
    }
#endif
};

/// Flat storage for LVQ-4 codes indexed by dense_id
/// Each vector is packed: 2 nibbles per byte, so bytes_per_vec = (dim + 1) / 2
struct LVQ4Store {
    /// Contiguous packed code storage: codes[dense_id * bytes_per_vec .. ]
    std::vector<uint8_t> codes;
    /// Per-vector scale factors
    std::vector<float> scales;
    /// Per-vector offsets
    std::vector<float> offsets;
    /// Dimensionality
    size_t dim = 0;
    /// Packed bytes per vector
    size_t bytes_per_vec = 0;
    /// Number of vectors stored
    size_t count = 0;

    /// Build from a locked quantization snapshot (preferred — no live graph iteration)
    void build(const index::QuantizationSnapshot& snap) {
        if (snap.entries.empty())
            return;

        dim = snap.dim;
        bytes_per_vec = (dim + 1) / 2;
        size_t max_dense_id = 0;
        for (const auto& e : snap.entries)
            max_dense_id = std::max(max_dense_id, e.dense_id);

        count = max_dense_id + 1;
        codes.resize(count * bytes_per_vec, 0);
        scales.resize(count, 0.0f);
        offsets.resize(count, 0.0f);

        for (const auto& e : snap.entries) {
            auto code = LVQ4::encode(std::span<const float>(e.vector));
            size_t offset = e.dense_id * bytes_per_vec;
            std::copy(code.codes.begin(), code.codes.end(), codes.begin() + offset);
            scales[e.dense_id] = code.scale;
            offsets[e.dense_id] = code.offset;
        }
    }

    /// Estimate L2 distance between float query and quantized vector at dense_id
    [[nodiscard]] float l2_distance(std::span<const float> query, size_t dense_id) const {
        assert(dense_id < count);
        assert(query.size() == dim);

        const float scale = scales[dense_id];
        const float offset = offsets[dense_id];
        const uint8_t* code_ptr = codes.data() + dense_id * bytes_per_vec;

        if (scale == 0.0f) {
            float sum = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float diff = query[i] - offset;
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }

        const float inv_scale = 1.0f / scale;
        float sum_sq = l2_sq_impl(query.data(), code_ptr, dim, inv_scale, offset);
        return std::sqrt(sum_sq * scale * scale);
    }

    /// Prefetch the quantized code for a given dense_id into L1 cache
    void prefetch(size_t dense_id) const {
        if (dense_id < count) {
            const void* addr = codes.data() + dense_id * bytes_per_vec;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(addr, 0, 3);
#endif
        }
    }

    /// Memory usage in bytes
    [[nodiscard]] size_t memory_bytes() const {
        return codes.size() + scales.size() * sizeof(float) + offsets.size() * sizeof(float);
    }

private:
    static float l2_sq_impl(const float* query, const uint8_t* codes, size_t dim, float inv_scale,
                            float offset) {
#ifdef SQLITE_VEC_ENABLE_NEON
        if (dim >= 32) {
            return l2_sq_neon(query, codes, dim, inv_scale, offset);
        }
#endif
        return l2_sq_scalar(query, codes, dim, inv_scale, offset);
    }

    static float l2_sq_scalar(const float* query, const uint8_t* codes, size_t dim, float inv_scale,
                              float offset) {
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
    static float l2_sq_neon(const float* query, const uint8_t* codes, size_t dim, float inv_scale,
                            float offset) {
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
            uint8x16_t packed = vld1q_u8(codes + i / 2);

            uint8x16_t lo_nibbles = vandq_u8(packed, mask_lo);
            uint8x16_t hi_nibbles = vshrq_n_u8(packed, 4);

            uint8x16_t unpacked_0_15 = vzip1q_u8(lo_nibbles, hi_nibbles);
            uint8x16_t unpacked_16_31 = vzip2q_u8(lo_nibbles, hi_nibbles);

            // Dims i..i+15
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

            // Dims i+16..i+31
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

/// Flat storage for RaBitQ codes indexed by dense_id.
/// Implements true RaBitQ (arXiv:2405.12497): rotated unit residuals are
/// binarized; per-vector residual norms and correction factors drive an
/// unbiased inner-product estimator. Query-side uses 4-bit scalar
/// quantization of the rotated query so estimation is pure popcounts.
struct RaBitQStore {
    /// Packed binary codes: bits[dense_id * bytes_per_vec .. ]
    std::vector<uint8_t> bits;
    /// Per-vector distance to centroid: ||v - c||
    std::vector<float> dist_to_centroid;
    /// Per-vector correction factor <x_bar, o_rot>
    std::vector<float> ip_quant;
    /// Dataset centroid
    std::vector<float> centroid;
    /// Bytes per vector (padded_dim / 8)
    size_t bytes_per_vec = 0;
    /// Dimensionality (unpadded)
    size_t dim = 0;
    /// Number of vectors stored
    size_t count = 0;

    /// Build from a locked quantization snapshot (preferred — no live graph iteration)
    void build(const index::QuantizationSnapshot& snap) {
        if (snap.entries.empty())
            return;

        dim = snap.dim;
        rotation_ = std::make_shared<FastRotation>(dim, kRaBitQDefaultSeed);
        const size_t padded = rotation_->padded_dim();
        bytes_per_vec = padded / 8;

        size_t max_dense_id = 0;
        for (const auto& e : snap.entries)
            max_dense_id = std::max(max_dense_id, e.dense_id);

        centroid.assign(dim, 0.0f);
        for (const auto& e : snap.entries) {
            for (size_t i = 0; i < dim; ++i)
                centroid[i] += e.vector[i];
        }
        float inv_n = 1.0f / static_cast<float>(snap.entries.size());
        for (size_t i = 0; i < dim; ++i)
            centroid[i] *= inv_n;

        count = max_dense_id + 1;
        bits.assign(count * bytes_per_vec, 0);
        dist_to_centroid.assign(count, 0.0f);
        ip_quant.assign(count, 1.0f);

        std::vector<float> residual(dim);
        std::vector<float> rotated(padded);
        for (const auto& e : snap.entries) {
            float dist_sq = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                residual[i] = e.vector[i] - centroid[i];
                dist_sq += residual[i] * residual[i];
            }
            const float dist = std::sqrt(dist_sq);
            dist_to_centroid[e.dense_id] = dist;
            if (dist <= kRaBitQEpsilon) {
                dist_to_centroid[e.dense_id] = 0.0f;
                continue;
            }

            const float inv_dist = 1.0f / dist;
            for (size_t i = 0; i < dim; ++i)
                residual[i] *= inv_dist;

            rotation_->apply(std::span<const float>(residual), std::span<float>(rotated));

            uint8_t* code_ptr = bits.data() + e.dense_id * bytes_per_vec;
            float abs_sum = 0.0f;
            for (size_t i = 0; i < padded; ++i) {
                abs_sum += std::abs(rotated[i]);
                if (rotated[i] >= 0.0f) {
                    code_ptr[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
                }
            }
            ip_quant[e.dense_id] =
                std::max(abs_sum * rotation_->inv_sqrt_padded(), kRaBitQEpsilon);
        }
    }

    /// Precomputed query state for fast distance estimation
    struct QueryState {
        std::vector<uint8_t> planes; ///< 4 bit-planes of the quantized rotated query
        size_t plane_bytes = 0;
        float dist_to_centroid = 0.0f;
        float vl = 0.0f;
        float delta = 0.0f;
        float sum_q = 0.0f;
    };

    /// Prepare query for fast estimation
    [[nodiscard]] QueryState prepare_query(std::span<const float> query) const {
        QueryState state;
        if (!rotation_)
            return state;
        const size_t padded = rotation_->padded_dim();
        state.plane_bytes = padded / 8;
        state.planes.assign(4 * state.plane_bytes, 0);

        float dist_sq = 0.0f;
        std::vector<float> residual(dim);
        for (size_t i = 0; i < dim; ++i) {
            residual[i] = query[i] - centroid[i];
            dist_sq += residual[i] * residual[i];
        }
        state.dist_to_centroid = std::sqrt(dist_sq);
        if (state.dist_to_centroid <= kRaBitQEpsilon) {
            state.dist_to_centroid = 0.0f;
            return state;
        }

        const float inv_dist = 1.0f / state.dist_to_centroid;
        for (size_t i = 0; i < dim; ++i)
            residual[i] *= inv_dist;

        std::vector<float> rotated(padded);
        rotation_->apply(std::span<const float>(residual), std::span<float>(rotated));

        float vl = rotated[0];
        float vr = rotated[0];
        for (size_t i = 1; i < padded; ++i) {
            vl = std::min(vl, rotated[i]);
            vr = std::max(vr, rotated[i]);
        }
        state.vl = vl;
        state.delta = (vr - vl) / 15.0f;

        const float inv_delta = state.delta > 0.0f ? 1.0f / state.delta : 0.0f;
        uint64_t sum_u = 0;
        for (size_t i = 0; i < padded; ++i) {
            const float scaled = (rotated[i] - vl) * inv_delta;
            const int u = std::clamp(static_cast<int>(scaled + 0.5f), 0, 15);
            sum_u += static_cast<uint64_t>(u);
            for (size_t j = 0; j < 4; ++j) {
                if ((u >> j) & 1) {
                    state.planes[j * state.plane_bytes + i / 8] |=
                        static_cast<uint8_t>(1u << (i % 8));
                }
            }
        }
        state.sum_q =
            static_cast<float>(padded) * vl + state.delta * static_cast<float>(sum_u);
        return state;
    }

    /// Estimate L2 distance using precomputed query state
    [[nodiscard]] float l2_distance(const QueryState& qs, size_t dense_id) const {
        assert(dense_id < count);

        const float qd = qs.dist_to_centroid;
        const float cd = dist_to_centroid[dense_id];
        if (qd <= 0.0f || cd <= 0.0f) {
            return std::sqrt(std::max(0.0f, qd * qd + cd * cd));
        }

        const uint8_t* code_ptr = bits.data() + dense_id * bytes_per_vec;
        const size_t nb = qs.plane_bytes;

        const uint32_t pc = detail::popcount_bytes(code_ptr, nb);
        uint64_t su = 0;
        for (size_t j = 0; j < 4; ++j) {
            su += static_cast<uint64_t>(1u << j) *
                  detail::and_popcount_bytes(code_ptr, qs.planes.data() + j * nb, nb);
        }

        const float sum_selected =
            qs.vl * static_cast<float>(pc) + qs.delta * static_cast<float>(su);
        const float ip_xq =
            (2.0f * sum_selected - qs.sum_q) * rotation_->inv_sqrt_padded();
        const float est = std::clamp(ip_xq / ip_quant[dense_id], -1.0f, 1.0f);

        const float dist_sq = qd * qd + cd * cd - 2.0f * qd * cd * est;
        return std::sqrt(std::max(0.0f, dist_sq));
    }

    /// Prefetch bits for a given dense_id
    void prefetch(size_t dense_id) const {
        if (dense_id < count) {
            const void* addr = bits.data() + dense_id * bytes_per_vec;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(addr, 0, 3);
#endif
        }
    }

    /// Memory usage in bytes
    [[nodiscard]] size_t memory_bytes() const {
        return bits.size() + dist_to_centroid.size() * sizeof(float) +
               ip_quant.size() * sizeof(float) + centroid.size() * sizeof(float) +
               (rotation_ ? rotation_->memory_bytes() : 0);
    }

private:
    std::shared_ptr<FastRotation> rotation_;
};

} // namespace sqlite_vec_cpp::quantization
