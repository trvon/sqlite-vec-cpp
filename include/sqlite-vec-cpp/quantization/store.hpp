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

    /// Build from an HNSW index. Iterates nodes, encodes each vector,
    /// and stores the code at the position of node.dense_id.
    template <typename IndexT> void build(const IndexT& index) {
        if (index.empty())
            return;

        // Determine max dense_id and dimensionality
        size_t max_dense_id = 0;
        for (const auto& [id, node] : index) {
            max_dense_id = std::max(max_dense_id, node.dense_id);
            if (dim == 0)
                dim = node.vector.size();
        }

        count = max_dense_id + 1;
        codes.resize(count * dim, 0);
        scales.resize(count, 0.0f);
        offsets.resize(count, 0.0f);

        for (const auto& [id, node] : index) {
            // Get float vector
            std::vector<float> fvec;
            std::span<const float> vec_span;
            if constexpr (std::same_as<typename IndexT::NodeType::value_type, float>) {
                vec_span = node.as_span();
            } else {
                fvec = index.to_float_vector(node.as_span());
                vec_span = std::span<const float>(fvec);
            }

            auto code = LVQ8::encode(vec_span);
            size_t offset = node.dense_id * dim;
            std::copy(code.codes.begin(), code.codes.end(), codes.begin() + offset);
            scales[node.dense_id] = code.scale;
            offsets[node.dense_id] = code.offset;
        }
    }

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

    /// Build from an HNSW index
    template <typename IndexT> void build(const IndexT& index) {
        if (index.empty())
            return;

        size_t max_dense_id = 0;
        for (const auto& [id, node] : index) {
            max_dense_id = std::max(max_dense_id, node.dense_id);
            if (dim == 0)
                dim = node.vector.size();
        }

        bytes_per_vec = (dim + 1) / 2;
        count = max_dense_id + 1;
        codes.resize(count * bytes_per_vec, 0);
        scales.resize(count, 0.0f);
        offsets.resize(count, 0.0f);

        for (const auto& [id, node] : index) {
            std::vector<float> fvec;
            std::span<const float> vec_span;
            if constexpr (std::same_as<typename IndexT::NodeType::value_type, float>) {
                vec_span = node.as_span();
            } else {
                fvec = index.to_float_vector(node.as_span());
                vec_span = std::span<const float>(fvec);
            }

            auto code = LVQ4::encode(vec_span);
            size_t offset = node.dense_id * bytes_per_vec;
            std::copy(code.codes.begin(), code.codes.end(), codes.begin() + offset);
            scales[node.dense_id] = code.scale;
            offsets[node.dense_id] = code.offset;
        }
    }

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

/// Flat storage for RaBitQ codes indexed by dense_id
struct RaBitQStore {
    /// Packed binary codes: bits[dense_id * bytes_per_vec .. ]
    std::vector<uint8_t> bits;
    /// Per-vector norms
    std::vector<float> norms;
    /// Dataset centroid
    std::vector<float> centroid;
    /// Bytes per vector
    size_t bytes_per_vec = 0;
    /// Dimensionality
    size_t dim = 0;
    /// Number of vectors stored
    size_t count = 0;

    /// Build from an HNSW index
    template <typename IndexT> void build(const IndexT& index) {
        if (index.empty())
            return;

        // Collect vectors for centroid computation
        size_t max_dense_id = 0;
        std::vector<std::vector<float>> all_vecs;
        std::vector<size_t> dense_ids;

        for (const auto& [id, node] : index) {
            max_dense_id = std::max(max_dense_id, node.dense_id);
            if (dim == 0)
                dim = node.vector.size();

            std::vector<float> fvec;
            if constexpr (std::same_as<typename IndexT::NodeType::value_type, float>) {
                fvec.assign(node.vector.begin(), node.vector.end());
            } else {
                fvec = index.to_float_vector(node.as_span());
            }
            all_vecs.push_back(std::move(fvec));
            dense_ids.push_back(node.dense_id);
        }

        // Compute centroid
        centroid.assign(dim, 0.0f);
        for (const auto& v : all_vecs) {
            for (size_t i = 0; i < dim; ++i)
                centroid[i] += v[i];
        }
        float inv_n = 1.0f / static_cast<float>(all_vecs.size());
        for (size_t i = 0; i < dim; ++i)
            centroid[i] *= inv_n;

        // Encode all vectors
        bytes_per_vec = (dim + 7) / 8;
        count = max_dense_id + 1;
        bits.resize(count * bytes_per_vec, 0);
        norms.resize(count, 0.0f);

        for (size_t idx = 0; idx < all_vecs.size(); ++idx) {
            const auto& vec = all_vecs[idx];
            size_t did = dense_ids[idx];
            size_t bit_offset = did * bytes_per_vec;

            float norm_sq = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float centered = vec[i] - centroid[i];
                norm_sq += vec[i] * vec[i];
                if (centered >= 0.0f) {
                    bits[bit_offset + i / 8] |= (1u << (i % 8));
                }
            }
            norms[did] = std::sqrt(norm_sq);
        }
    }

    /// Build from a locked quantization snapshot (preferred — no live graph iteration)
    void build(const index::QuantizationSnapshot& snap) {
        if (snap.entries.empty())
            return;

        dim = snap.dim;
        size_t max_dense_id = 0;
        for (const auto& e : snap.entries)
            max_dense_id = std::max(max_dense_id, e.dense_id);

        // Compute centroid
        centroid.assign(dim, 0.0f);
        for (const auto& e : snap.entries) {
            for (size_t i = 0; i < dim; ++i)
                centroid[i] += e.vector[i];
        }
        float inv_n = 1.0f / static_cast<float>(snap.entries.size());
        for (size_t i = 0; i < dim; ++i)
            centroid[i] *= inv_n;

        // Encode all vectors
        bytes_per_vec = (dim + 7) / 8;
        count = max_dense_id + 1;
        bits.resize(count * bytes_per_vec, 0);
        norms.resize(count, 0.0f);

        for (const auto& e : snap.entries) {
            size_t bit_offset = e.dense_id * bytes_per_vec;
            float norm_sq = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float centered = e.vector[i] - centroid[i];
                norm_sq += e.vector[i] * e.vector[i];
                if (centered >= 0.0f) {
                    bits[bit_offset + i / 8] |= (1u << (i % 8));
                }
            }
            norms[e.dense_id] = std::sqrt(norm_sq);
        }
    }

    /// Precomputed query state for fast distance estimation
    struct QueryState {
        std::vector<uint8_t> bits;
        float norm;
    };

    /// Prepare query for fast estimation
    [[nodiscard]] QueryState prepare_query(std::span<const float> query) const {
        QueryState state;
        state.bits.resize(bytes_per_vec, 0);
        float norm_sq = 0.0f;

        for (size_t i = 0; i < dim; ++i) {
            float centered = query[i] - centroid[i];
            norm_sq += query[i] * query[i];
            if (centered >= 0.0f) {
                state.bits[i / 8] |= (1u << (i % 8));
            }
        }
        state.norm = std::sqrt(norm_sq);
        return state;
    }

    /// Estimate L2 distance using precomputed query state
    [[nodiscard]] float l2_distance(const QueryState& qs, size_t dense_id) const {
        assert(dense_id < count);

        const uint8_t* code_ptr = bits.data() + dense_id * bytes_per_vec;
        uint32_t hamming = hamming_distance_impl(qs.bits.data(), code_ptr, bytes_per_vec);

        float binary_ip = static_cast<float>(dim) - 2.0f * static_cast<float>(hamming);
        float norm_factor = (dim > 0) ? (qs.norm * norms[dense_id] / static_cast<float>(dim)) : 0.0f;
        float ip_estimate = norm_factor * binary_ip;
        float dist_sq = qs.norm * qs.norm + norms[dense_id] * norms[dense_id] - 2.0f * ip_estimate;

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
        return bits.size() + norms.size() * sizeof(float) + centroid.size() * sizeof(float);
    }

private:
    static uint32_t hamming_distance_impl(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
#ifdef SQLITE_VEC_ENABLE_NEON
        if (num_bytes >= 16)
            return hamming_neon(a, b, num_bytes);
#endif
        return hamming_scalar(a, b, num_bytes);
    }

    static uint32_t hamming_scalar(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
        static constexpr uint8_t popcount_table[256] = {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3,
            4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4,
            4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
            3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
            4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3,
            4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3,
            3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5,
            6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
            4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5,
            6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
        uint32_t count = 0;
        for (size_t i = 0; i < num_bytes; ++i)
            count += popcount_table[a[i] ^ b[i]];
        return count;
    }

#ifdef SQLITE_VEC_ENABLE_NEON
    static uint32_t hamming_neon(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
        uint32_t total = 0;
        size_t i = 0;
        const size_t end16 = (num_bytes / 16) * 16;
        uint8x16_t acc = vdupq_n_u8(0);
        size_t iter_count = 0;

        while (i < end16) {
            uint8x16_t va = vld1q_u8(a + i);
            uint8x16_t vb = vld1q_u8(b + i);
            acc = vaddq_u8(acc, vcntq_u8(veorq_u8(va, vb)));
            i += 16;
            if (++iter_count == 31) {
                uint16x8_t s16 = vpaddlq_u8(acc);
                uint32x4_t s32 = vpaddlq_u16(s16);
                uint64x2_t s64 = vpaddlq_u32(s32);
                total += vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1);
                acc = vdupq_n_u8(0);
                iter_count = 0;
            }
        }

        uint16x8_t s16 = vpaddlq_u8(acc);
        uint32x4_t s32 = vpaddlq_u16(s16);
        uint64x2_t s64 = vpaddlq_u32(s32);
        total += vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1);

        while (i < num_bytes) {
            static constexpr uint8_t pt[256] = {
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2,
                3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3,
                3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3,
                4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4,
                3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5,
                6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4,
                4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
                6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5,
                3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3,
                4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6,
                6, 7, 6, 7, 7, 8};
            total += pt[a[i] ^ b[i]];
            ++i;
        }
        return total;
    }
#endif
};

} // namespace sqlite_vec_cpp::quantization
