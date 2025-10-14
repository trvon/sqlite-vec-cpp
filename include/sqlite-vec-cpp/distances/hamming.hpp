#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <climits>
#include <cstdint>
#include <span>
#include "../concepts/distance_metric.hpp"
#include "../concepts/vector_element.hpp"

namespace sqlite_vec_cpp::distances {

/// Constexpr hamming distance lookup table
/// Counts the number of set bits in a byte (popcount for uint8_t)
/// From: https://github.com/facebookresearch/faiss/blob/main/faiss/utils/hamming_distance/common.h
constexpr std::array<std::uint8_t, 256> hamming_lookup_table = {
    {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3,
     4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
     4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
     5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5,
     4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
     3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
     5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
     5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
     4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8}};

// Verify constexpr evaluation at compile time
static_assert(hamming_lookup_table[0] == 0);
static_assert(hamming_lookup_table[1] == 1);
static_assert(hamming_lookup_table[255] == 8);
static_assert(hamming_lookup_table[0b10101010] == 4);

/// Hamming distance for uint8_t bitvectors using lookup table
inline float hamming_distance_u8(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
    assert(a.size() == b.size() && "Bitvector dimensions must match");

    std::size_t count = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        count += hamming_lookup_table[a[i] ^ b[i]];
    }
    return static_cast<float>(count);
}

/// Hamming distance for uint64_t bitvectors using popcount
inline float hamming_distance_u64(std::span<const std::uint64_t> a,
                                  std::span<const std::uint64_t> b) {
    assert(a.size() == b.size() && "Bitvector dimensions must match");

    std::size_t count = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        // C++20 std::popcount for efficient bit counting
        count += std::popcount(a[i] ^ b[i]);
    }
    return static_cast<float>(count);
}

/// Main hamming distance function
/// Note: Input is bitvector data (bits packed into bytes)
/// The 'dimensions' parameter in original code is the number of BITS, not bytes
inline float hamming_distance(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b,
                              std::size_t bit_dimensions) {
    assert(a.size() == b.size());
    assert(a.size() * CHAR_BIT >= bit_dimensions && "Buffer too small for bit dimensions");

    const std::size_t byte_size = (bit_dimensions + CHAR_BIT - 1) / CHAR_BIT;

    // Optimize for uint64_t if dimensions align
    if (bit_dimensions % 64 == 0) {
        const std::size_t u64_size = bit_dimensions / 64;
        auto a_u64 = std::span<const std::uint64_t>(
            reinterpret_cast<const std::uint64_t*>(a.data()), u64_size);
        auto b_u64 = std::span<const std::uint64_t>(
            reinterpret_cast<const std::uint64_t*>(b.data()), u64_size);
        return hamming_distance_u64(a_u64, b_u64);
    }

    // Fallback to byte-wise lookup table
    return hamming_distance_u8(a.subspan(0, byte_size), b.subspan(0, byte_size));
}

/// HammingMetric functor (for bitvector types only)
/// Note: This operates on uint8_t (packed bits), not arbitrary vector elements
struct HammingMetric {
    using element_type = std::uint8_t;
    std::size_t bit_dimensions; // Number of bits in the bitvector

    explicit HammingMetric(std::size_t bits) : bit_dimensions(bits) {}

    [[nodiscard]] float operator()(std::span<const std::uint8_t> a,
                                   std::span<const std::uint8_t> b) const {
        return hamming_distance(a, b, bit_dimensions);
    }
};

} // namespace sqlite_vec_cpp::distances

// Trait specializations (must be in concepts::traits namespace)
namespace sqlite_vec_cpp::concepts::traits {
template <> struct is_symmetric<sqlite_vec_cpp::distances::HammingMetric> : std::true_type {};

template <> struct is_metric_space<sqlite_vec_cpp::distances::HammingMetric> : std::true_type {};
} // namespace sqlite_vec_cpp::concepts::traits

namespace sqlite_vec_cpp::distances {

// Verify concept satisfaction
static_assert(concepts::DistanceMetric<HammingMetric, std::uint8_t>);
static_assert(concepts::SymmetricDistanceMetric<HammingMetric, std::uint8_t>);
static_assert(concepts::MetricSpace<HammingMetric, std::uint8_t>);

} // namespace sqlite_vec_cpp::distances
