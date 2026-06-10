#pragma once

/// RaBitQ: Randomized Binary Quantization
///
/// Based on "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
/// Error Bound for Approximate Nearest Neighbor Search" (Gao & Long, 2024,
/// arXiv:2405.12497).
///
/// Pipeline (per the paper, structured-rotation variant):
///   1. Center each vector on the dataset centroid and normalize the residual.
///   2. Apply a seeded random orthogonal rotation (sign flips + fast
///      Walsh-Hadamard transform, two rounds).
///   3. Binarize the rotated unit residual: bit i = sign(rot_i).
///      The quantized vector is x_bar = (2b - 1) / sqrt(D_pad).
///   4. Store per vector: packed bits, the residual norm ||v - c||, and the
///      correction factor <x_bar, o_rot> (computable exactly at encode time).
///
/// Distance estimation:
///   <o, q>      ~= <x_bar, q_rot> / <x_bar, o_rot>     (unbiased estimator)
///   ||v - q||^2  = ||v - c||^2 + ||q - c||^2
///                  - 2 ||v - c|| ||q - c|| <o, q>
///
/// The query-side inner product <x_bar, q_rot> is computed with bitwise ops:
/// the rotated query is scalar-quantized to 4 bits and the inner product
/// reduces to five popcounts over the packed code (one plain + four ANDed
/// bit-planes), exactly as in the reference implementation.
///
/// Key properties:
/// - ~32x compression vs FP32 (D dims -> D_pad/8 bytes + 8 bytes corrections)
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

inline constexpr uint32_t kRaBitQDefaultSeed = 0x5EEDB175u;
inline constexpr float kRaBitQEpsilon = 1e-12f;

namespace detail {

inline constexpr uint8_t kPopcountTable[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4,
    5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
    5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4,
    5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
    5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
    5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5,
    5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4,
    5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7,
    5, 6, 6, 7, 6, 7, 7, 8};

inline uint32_t popcount_scalar(const uint8_t* a, size_t num_bytes) {
    uint32_t count = 0;
    for (size_t i = 0; i < num_bytes; ++i) {
        count += kPopcountTable[a[i]];
    }
    return count;
}

inline uint32_t and_popcount_scalar(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
    uint32_t count = 0;
    for (size_t i = 0; i < num_bytes; ++i) {
        count += kPopcountTable[a[i] & b[i]];
    }
    return count;
}

inline uint32_t xor_popcount_scalar(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
    uint32_t count = 0;
    for (size_t i = 0; i < num_bytes; ++i) {
        count += kPopcountTable[a[i] ^ b[i]];
    }
    return count;
}

#ifdef SQLITE_VEC_ENABLE_NEON
inline uint32_t reduce_u8x16(uint8x16_t acc) {
    uint16x8_t sum16 = vpaddlq_u8(acc);
    uint32x4_t sum32 = vpaddlq_u16(sum16);
    uint64x2_t sum64 = vpaddlq_u32(sum32);
    return static_cast<uint32_t>(vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1));
}

inline uint32_t popcount_neon(const uint8_t* a, size_t num_bytes) {
    uint32_t total = 0;
    size_t i = 0;
    const size_t end16 = (num_bytes / 16) * 16;
    uint8x16_t acc = vdupq_n_u8(0);
    size_t iters = 0;
    while (i < end16) {
        acc = vaddq_u8(acc, vcntq_u8(vld1q_u8(a + i)));
        i += 16;
        if (++iters == 31) {
            total += reduce_u8x16(acc);
            acc = vdupq_n_u8(0);
            iters = 0;
        }
    }
    total += reduce_u8x16(acc);
    while (i < num_bytes) {
        total += kPopcountTable[a[i]];
        ++i;
    }
    return total;
}

inline uint32_t and_popcount_neon(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
    uint32_t total = 0;
    size_t i = 0;
    const size_t end16 = (num_bytes / 16) * 16;
    uint8x16_t acc = vdupq_n_u8(0);
    size_t iters = 0;
    while (i < end16) {
        acc = vaddq_u8(acc, vcntq_u8(vandq_u8(vld1q_u8(a + i), vld1q_u8(b + i))));
        i += 16;
        if (++iters == 31) {
            total += reduce_u8x16(acc);
            acc = vdupq_n_u8(0);
            iters = 0;
        }
    }
    total += reduce_u8x16(acc);
    while (i < num_bytes) {
        total += kPopcountTable[a[i] & b[i]];
        ++i;
    }
    return total;
}

inline uint32_t xor_popcount_neon(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
    uint32_t total = 0;
    size_t i = 0;
    const size_t end16 = (num_bytes / 16) * 16;
    uint8x16_t acc = vdupq_n_u8(0);
    size_t iters = 0;
    while (i < end16) {
        acc = vaddq_u8(acc, vcntq_u8(veorq_u8(vld1q_u8(a + i), vld1q_u8(b + i))));
        i += 16;
        if (++iters == 31) {
            total += reduce_u8x16(acc);
            acc = vdupq_n_u8(0);
            iters = 0;
        }
    }
    total += reduce_u8x16(acc);
    while (i < num_bytes) {
        total += kPopcountTable[a[i] ^ b[i]];
        ++i;
    }
    return total;
}
#endif

inline uint32_t popcount_bytes(const uint8_t* a, size_t num_bytes) {
#ifdef SQLITE_VEC_ENABLE_NEON
    if (num_bytes >= 16) {
        return popcount_neon(a, num_bytes);
    }
#endif
    return popcount_scalar(a, num_bytes);
}

inline uint32_t and_popcount_bytes(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
#ifdef SQLITE_VEC_ENABLE_NEON
    if (num_bytes >= 16) {
        return and_popcount_neon(a, b, num_bytes);
    }
#endif
    return and_popcount_scalar(a, b, num_bytes);
}

inline uint32_t xor_popcount_bytes(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
#ifdef SQLITE_VEC_ENABLE_NEON
    if (num_bytes >= 16) {
        return xor_popcount_neon(a, b, num_bytes);
    }
#endif
    return xor_popcount_scalar(a, b, num_bytes);
}

} // namespace detail

/// Seeded random orthogonal rotation: rounds of {random sign flips, FWHT}.
/// Each round is orthogonal (norm-preserving after 1/sqrt(D) scaling), so the
/// composition is a structured random rotation a la the fast Johnson-
/// Lindenstrauss transform. Deterministic for a given (dim, seed).
class FastRotation {
public:
    explicit FastRotation(size_t dim, uint32_t seed = kRaBitQDefaultSeed) : dim_(dim) {
        padded_dim_ = 8;
        while (padded_dim_ < dim_) {
            padded_dim_ <<= 1;
        }
        inv_sqrt_padded_ = 1.0f / std::sqrt(static_cast<float>(padded_dim_));

        std::mt19937 rng(seed);
        const size_t sign_bytes = kRounds * (padded_dim_ / 8);
        sign_bits_.resize(sign_bytes);
        std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
        for (auto& b : sign_bits_) {
            b = static_cast<uint8_t>(byte_dist(rng));
        }
    }

    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t padded_dim() const { return padded_dim_; }
    [[nodiscard]] float inv_sqrt_padded() const { return inv_sqrt_padded_; }

    /// Rotate `in` (dim elements) into `out` (padded_dim elements, zero-padded).
    void apply(std::span<const float> in, std::span<float> out) const {
        assert(in.size() == dim_);
        assert(out.size() == padded_dim_);

        std::fill(out.begin(), out.end(), 0.0f);
        std::copy(in.begin(), in.end(), out.begin());

        for (size_t r = 0; r < kRounds; ++r) {
            const uint8_t* signs = sign_bits_.data() + r * (padded_dim_ / 8);
            for (size_t i = 0; i < padded_dim_; ++i) {
                if ((signs[i / 8] >> (i % 8)) & 1u) {
                    out[i] = -out[i];
                }
            }
            fwht(out);
            for (size_t i = 0; i < padded_dim_; ++i) {
                out[i] *= inv_sqrt_padded_;
            }
        }
    }

    [[nodiscard]] size_t memory_bytes() const { return sign_bits_.size(); }

private:
    static constexpr size_t kRounds = 2;

    static void fwht(std::span<float> data) {
        const size_t n = data.size();
        for (size_t len = 1; len < n; len <<= 1) {
            for (size_t i = 0; i < n; i += (len << 1)) {
                for (size_t j = i; j < i + len; ++j) {
                    const float a = data[j];
                    const float b = data[j + len];
                    data[j] = a + b;
                    data[j + len] = a - b;
                }
            }
        }
    }

    size_t dim_;
    size_t padded_dim_;
    float inv_sqrt_padded_;
    std::vector<uint8_t> sign_bits_;
};

/// Binary code produced by RaBitQ encoding
struct RaBitQCode {
    std::vector<uint8_t> bits; ///< Packed binary code (padded_dim/8 bytes)
    float norm = 0.0f;         ///< L2 norm of the original vector
    float dist_to_centroid = 0.0f; ///< ||v - c||, used in the distance identity
    float ip_quant = 1.0f;     ///< <x_bar, o_rot>, the per-vector correction factor

    /// Number of packed bytes
    [[nodiscard]] size_t byte_size() const { return bits.size(); }

    /// Number of dimensions encoded (padded)
    [[nodiscard]] size_t dimensions() const { return bits.size() * 8; }
};

/// RaBitQ encoder/distance estimator
class RaBitQ {
public:
    /// Initialize RaBitQ with dataset statistics
    /// @param centroid The mean vector of the dataset (D dimensions)
    explicit RaBitQ(std::vector<float> centroid, uint32_t seed = kRaBitQDefaultSeed)
        : centroid_(std::move(centroid)), rotation_(centroid_.size(), seed) {}

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

    /// Encode a vector to binary code: rotate the normalized centered residual
    /// and binarize each dimension; store the exact correction factor.
    [[nodiscard]] RaBitQCode encode(std::span<const float> vec) const {
        assert(vec.size() == centroid_.size());
        const size_t dim = vec.size();
        const size_t padded = rotation_.padded_dim();

        RaBitQCode code;
        code.bits.assign(padded / 8, 0);

        float norm_sq = 0.0f;
        float dist_sq = 0.0f;
        std::vector<float> residual(dim);
        for (size_t i = 0; i < dim; ++i) {
            norm_sq += vec[i] * vec[i];
            residual[i] = vec[i] - centroid_[i];
            dist_sq += residual[i] * residual[i];
        }
        code.norm = std::sqrt(norm_sq);
        code.dist_to_centroid = std::sqrt(dist_sq);

        if (code.dist_to_centroid <= kRaBitQEpsilon) {
            code.dist_to_centroid = 0.0f;
            code.ip_quant = 1.0f;
            return code;
        }

        const float inv_dist = 1.0f / code.dist_to_centroid;
        for (size_t i = 0; i < dim; ++i) {
            residual[i] *= inv_dist;
        }

        std::vector<float> rotated(padded);
        rotation_.apply(std::span<const float>(residual), std::span<float>(rotated));

        float abs_sum = 0.0f;
        for (size_t i = 0; i < padded; ++i) {
            abs_sum += std::abs(rotated[i]);
            if (rotated[i] >= 0.0f) {
                code.bits[i / 8] |= (1u << (i % 8));
            }
        }
        code.ip_quant = std::max(abs_sum * rotation_.inv_sqrt_padded(), kRaBitQEpsilon);

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

    /// Precomputed query-side data for fast distance estimation.
    /// The rotated unit residual of the query is scalar-quantized to 4 bits;
    /// inner products against binary codes then reduce to popcounts over the
    /// four bit-planes (plus one popcount of the code itself).
    struct QueryState {
        std::vector<uint8_t> planes; ///< 4 bit-planes, each padded_dim/8 bytes
        size_t plane_bytes = 0;      ///< Bytes per plane
        float dist_to_centroid = 0.0f; ///< ||q - c||
        float vl = 0.0f;             ///< Quantization grid lower bound
        float delta = 0.0f;          ///< Quantization grid step
        float sum_q = 0.0f;          ///< Sum of quantized rotated query values
    };

    /// Prepare query for distance estimation
    [[nodiscard]] QueryState prepare_query(std::span<const float> query) const {
        assert(query.size() == centroid_.size());
        const size_t dim = query.size();
        const size_t padded = rotation_.padded_dim();

        QueryState state;
        state.plane_bytes = padded / 8;
        state.planes.assign(4 * state.plane_bytes, 0);

        float dist_sq = 0.0f;
        std::vector<float> residual(dim);
        for (size_t i = 0; i < dim; ++i) {
            residual[i] = query[i] - centroid_[i];
            dist_sq += residual[i] * residual[i];
        }
        state.dist_to_centroid = std::sqrt(dist_sq);

        if (state.dist_to_centroid <= kRaBitQEpsilon) {
            state.dist_to_centroid = 0.0f;
            return state;
        }

        const float inv_dist = 1.0f / state.dist_to_centroid;
        for (size_t i = 0; i < dim; ++i) {
            residual[i] *= inv_dist;
        }

        std::vector<float> rotated(padded);
        rotation_.apply(std::span<const float>(residual), std::span<float>(rotated));

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
        state.sum_q = static_cast<float>(padded) * vl + state.delta * static_cast<float>(sum_u);

        return state;
    }

    /// Estimate L2 distance between prepared query and encoded vector using
    /// the geometric identity over centered residuals with the unbiased
    /// inner-product estimator <x_bar, q_rot> / <x_bar, o_rot>.
    [[nodiscard]] float estimate_l2_distance(const QueryState& query,
                                             const RaBitQCode& code) const {
        const float qd = query.dist_to_centroid;
        const float cd = code.dist_to_centroid;
        if (qd <= 0.0f || cd <= 0.0f) {
            return std::sqrt(std::max(0.0f, qd * qd + cd * cd));
        }

        const size_t nb = query.plane_bytes;
        const uint8_t* bits = code.bits.data();

        const uint32_t pc = detail::popcount_bytes(bits, nb);
        uint64_t su = 0;
        for (size_t j = 0; j < 4; ++j) {
            su += static_cast<uint64_t>(1u << j) *
                  detail::and_popcount_bytes(bits, query.planes.data() + j * nb, nb);
        }

        const float sum_selected =
            query.vl * static_cast<float>(pc) + query.delta * static_cast<float>(su);
        const float ip_xq =
            (2.0f * sum_selected - query.sum_q) * rotation_.inv_sqrt_padded();
        const float est = std::clamp(ip_xq / code.ip_quant, -1.0f, 1.0f);

        const float dist_sq = qd * qd + cd * cd - 2.0f * qd * cd * est;
        return std::sqrt(std::max(0.0f, dist_sq));
    }

    /// Convenience: estimate L2 distance from raw query vector to encoded vector
    [[nodiscard]] float l2_distance(std::span<const float> query, const RaBitQCode& code) const {
        auto state = prepare_query(query);
        return estimate_l2_distance(state, code);
    }

    /// Get the centroid
    [[nodiscard]] std::span<const float> centroid() const { return centroid_; }

    /// Get dimensionality
    [[nodiscard]] size_t dimensions() const { return centroid_.size(); }

    /// Get the shared rotation
    [[nodiscard]] const FastRotation& rotation() const { return rotation_; }

private:
    std::vector<float> centroid_; ///< Dataset centroid (mean vector)
    FastRotation rotation_;       ///< Seeded structured random rotation
};

} // namespace sqlite_vec_cpp::quantization
