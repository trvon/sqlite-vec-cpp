#pragma once

/// RaBitQ: Randomized Binary Quantization
///
/// Based on "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
/// Error Bound for Approximate Nearest Neighbor Search" (Gao & Long, 2024).
///
/// Quantizes D-dimensional vectors into D-bit strings. Distance estimation
/// uses popcount (Hamming distance) plus scalar corrections for the L2 case.
///
/// Key properties:
/// - 32x compression vs FP32 (D dims -> D/8 bytes)
/// - Sharp theoretical error bounds (unlike PQ)
/// - SIMD-friendly: popcount on NEON is a single instruction (vcnt)
/// - Pairs naturally with HNSW: binary distance for graph traversal,
///   FP32 rerank for final top-K

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#ifdef SQLITE_VEC_ENABLE_NEON
#include <arm_neon.h>
#endif

namespace sqlite_vec_cpp::quantization {

/// Binary code produced by RaBitQ encoding
struct RaBitQCode {
    std::vector<uint8_t> bits; ///< Packed binary code (D/8 bytes for D dimensions)
    float norm;                ///< L2 norm of the original vector
    float centroid_dot;        ///< Dot product of original vector with dataset centroid

    /// Number of packed bytes
    [[nodiscard]] size_t byte_size() const { return bits.size(); }

    /// Number of dimensions encoded
    [[nodiscard]] size_t dimensions() const { return bits.size() * 8; }
};

/// RaBitQ encoder/distance estimator
class RaBitQ {
public:
    /// Initialize RaBitQ with dataset statistics
    /// @param centroid The mean vector of the dataset (D dimensions)
    explicit RaBitQ(std::vector<float> centroid) : centroid_(std::move(centroid)) {}

    /// Compute centroid from a set of vectors
    static std::vector<float> compute_centroid(std::span<const std::span<const float>> vectors) {
        if (vectors.empty())
            return {};

        size_t dim = vectors[0].size();
        std::vector<float> centroid(dim, 0.0f);

        for (const auto& vec : vectors) {
            assert(vec.size() == dim);
            for (size_t i = 0; i < dim; ++i) {
                centroid[i] += vec[i];
            }
        }

        float inv_n = 1.0f / static_cast<float>(vectors.size());
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] *= inv_n;
        }

        return centroid;
    }

    /// Encode a vector to binary code
    /// Process: center the vector (subtract centroid), then binarize each dimension
    /// bit[i] = 1 if (vec[i] - centroid[i]) >= 0, else 0
    [[nodiscard]] RaBitQCode encode(std::span<const float> vec) const {
        assert(vec.size() == centroid_.size());
        const size_t dim = vec.size();

        RaBitQCode code;
        code.bits.resize((dim + 7) / 8, 0);

        // Compute norm and centroid dot product for distance correction
        float norm_sq = 0.0f;
        float centroid_dot = 0.0f;

        for (size_t i = 0; i < dim; ++i) {
            float centered = vec[i] - centroid_[i];
            norm_sq += vec[i] * vec[i];
            centroid_dot += vec[i] * centroid_[i];

            // Binarize: bit = 1 if centered value >= 0
            if (centered >= 0.0f) {
                code.bits[i / 8] |= (1u << (i % 8));
            }
        }

        code.norm = std::sqrt(norm_sq);
        code.centroid_dot = centroid_dot;

        return code;
    }

    /// Batch encode all vectors
    [[nodiscard]] std::vector<RaBitQCode>
    encode_all(std::span<const std::span<const float>> vectors) const {
        std::vector<RaBitQCode> codes;
        codes.reserve(vectors.size());
        for (const auto& vec : vectors) {
            codes.push_back(encode(vec));
        }
        return codes;
    }

    /// Precompute query-side data for fast distance estimation
    /// Call once per query, then use estimate_l2_distance for each candidate
    struct QueryState {
        std::vector<uint8_t> bits; ///< Binary code of centered query
        float norm;                ///< L2 norm of query
        float centroid_dot;        ///< Dot product of query with centroid
        float centroid_norm_sq;    ///< Squared norm of centroid
    };

    /// Prepare query for distance estimation
    [[nodiscard]] QueryState prepare_query(std::span<const float> query) const {
        assert(query.size() == centroid_.size());
        const size_t dim = query.size();

        QueryState state;
        state.bits.resize((dim + 7) / 8, 0);
        state.norm = 0.0f;
        state.centroid_dot = 0.0f;
        state.centroid_norm_sq = 0.0f;

        for (size_t i = 0; i < dim; ++i) {
            float centered = query[i] - centroid_[i];
            state.norm += query[i] * query[i];
            state.centroid_dot += query[i] * centroid_[i];
            state.centroid_norm_sq += centroid_[i] * centroid_[i];

            if (centered >= 0.0f) {
                state.bits[i / 8] |= (1u << (i % 8));
            }
        }

        state.norm = std::sqrt(state.norm);

        return state;
    }

    /// Estimate L2 distance between prepared query and encoded vector
    ///
    /// The estimation uses:
    ///   ||q - x||^2 ~= ||q||^2 + ||x||^2 - 2 * correction(hamming, norms)
    ///
    /// The binary dot product (D - 2*hamming) provides a fast estimate of the
    /// actual inner product direction, scaled by the norms for magnitude.
    [[nodiscard]] float estimate_l2_distance(const QueryState& query,
                                             const RaBitQCode& code) const {
        const size_t dim = centroid_.size();
        const size_t num_bytes = query.bits.size();

        // Compute Hamming distance between binary codes
        uint32_t hamming = hamming_distance(query.bits.data(), code.bits.data(), num_bytes);

        // Binary inner product estimate: sign agreement count
        // binary_ip = D - 2 * hamming  (ranges from -D to D)
        float binary_ip = static_cast<float>(dim) - 2.0f * static_cast<float>(hamming);

        // Scale by norms: the binary codes capture direction, norms capture magnitude
        // ip_estimate = (query.norm * code.norm / D) * binary_ip
        float norm_factor =
            (dim > 0) ? (query.norm * code.norm / static_cast<float>(dim)) : 0.0f;
        float ip_estimate = norm_factor * binary_ip;

        // L2^2 = ||q||^2 + ||x||^2 - 2 * ip_estimate
        float dist_sq =
            query.norm * query.norm + code.norm * code.norm - 2.0f * ip_estimate;

        // Clamp to prevent sqrt of negative (numerical errors)
        return std::sqrt(std::max(0.0f, dist_sq));
    }

    /// Convenience: estimate L2 distance from raw query vector to encoded vector
    [[nodiscard]] float l2_distance(std::span<const float> query,
                                    const RaBitQCode& code) const {
        auto state = prepare_query(query);
        return estimate_l2_distance(state, code);
    }

    /// Get the centroid
    [[nodiscard]] std::span<const float> centroid() const { return centroid_; }

    /// Get dimensionality
    [[nodiscard]] size_t dimensions() const { return centroid_.size(); }

private:
    std::vector<float> centroid_; ///< Dataset centroid (mean vector)

    /// Compute Hamming distance between two packed bit arrays
    static uint32_t hamming_distance(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
#ifdef SQLITE_VEC_ENABLE_NEON
        if (num_bytes >= 16) {
            return hamming_distance_neon(a, b, num_bytes);
        }
#endif
        return hamming_distance_scalar(a, b, num_bytes);
    }

    /// Scalar Hamming distance using lookup table
    static uint32_t hamming_distance_scalar(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
        // Precomputed popcount lookup table
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
        for (size_t i = 0; i < num_bytes; ++i) {
            count += popcount_table[a[i] ^ b[i]];
        }
        return count;
    }

#ifdef SQLITE_VEC_ENABLE_NEON
    /// NEON Hamming distance using vcnt (byte-level popcount)
    static uint32_t hamming_distance_neon(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
        uint32_t total = 0;
        size_t i = 0;

        // Process 16 bytes at a time
        const size_t end16 = (num_bytes / 16) * 16;
        uint8x16_t acc = vdupq_n_u8(0);

        while (i < end16) {
            uint8x16_t va = vld1q_u8(a + i);
            uint8x16_t vb = vld1q_u8(b + i);
            uint8x16_t vxor = veorq_u8(va, vb);
            uint8x16_t vpopcnt = vcntq_u8(vxor); // Byte-level popcount
            acc = vaddq_u8(acc, vpopcnt);
            i += 16;

            // Reduce periodically to prevent uint8 overflow (every 31 iterations)
            if ((i / 16) % 31 == 0) {
                // Horizontal sum of acc into total
                uint16x8_t sum16 = vpaddlq_u8(acc);
                uint32x4_t sum32 = vpaddlq_u16(sum16);
                uint64x2_t sum64 = vpaddlq_u32(sum32);
                total += vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
                acc = vdupq_n_u8(0);
            }
        }

        // Final reduction of accumulator
        uint16x8_t sum16 = vpaddlq_u8(acc);
        uint32x4_t sum32 = vpaddlq_u16(sum16);
        uint64x2_t sum64 = vpaddlq_u32(sum32);
        total += vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);

        // Handle remaining bytes
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

        while (i < num_bytes) {
            total += popcount_table[a[i] ^ b[i]];
            ++i;
        }

        return total;
    }
#endif
};

} // namespace sqlite_vec_cpp::quantization
