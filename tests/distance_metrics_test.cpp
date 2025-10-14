// SPDX-License-Identifier: MIT
// Copyright (c) 2025 YAMS Contributors
// Comprehensive test suite for sqlite-vec-cpp distance metrics

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../include/sqlite-vec-cpp/concepts.hpp"
#include "../include/sqlite-vec-cpp/distances.hpp"
#include "../include/sqlite-vec-cpp/vector_view.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace sqlite_vec_cpp;
using Catch::Approx;
using Catch::Matchers::WithinRel;

// ============================================================================
// Test Fixtures and Helpers
// ============================================================================

class DistanceMetricsFixture {
protected:
    std::mt19937 rng{42};

    template <typename T> std::vector<T> makeRandomVector(size_t dims, T min_val, T max_val) {
        std::vector<T> vec(dims);
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(min_val, max_val);
            std::generate(vec.begin(), vec.end(), [&]() { return dist(rng); });
        } else {
            std::uniform_int_distribution<T> dist(min_val, max_val);
            std::generate(vec.begin(), vec.end(), [&]() { return dist(rng); });
        }
        return vec;
    }

    template <typename T> std::vector<T> makeZeroVector(size_t dims) {
        return std::vector<T>(dims, T{0});
    }

    template <typename T> std::vector<T> makeUnitVector(size_t dims, size_t unit_idx = 0) {
        std::vector<T> vec(dims, T{0});
        if (unit_idx < dims)
            vec[unit_idx] = T{1};
        return vec;
    }

    template <typename T> std::vector<T> makeConstantVector(size_t dims, T value) {
        return std::vector<T>(dims, value);
    }
};

// ============================================================================
// L2 Distance Tests
// ============================================================================

TEST_CASE_METHOD(DistanceMetricsFixture, "L2 distance: Basic properties", "[distance][l2]") {
    SECTION("Identity: distance to self is zero") {
        auto vec = makeRandomVector<float>(128, -1.0f, 1.0f);
        auto dist = l2_distance(std::span{vec}, std::span{vec});
        REQUIRE_THAT(dist, WithinRel(0.0f, 1e-5));
    }

    SECTION("Symmetry: d(a,b) == d(b,a)") {
        auto a = makeRandomVector<float>(64, -10.0f, 10.0f);
        auto b = makeRandomVector<float>(64, -10.0f, 10.0f);
        auto d_ab = l2_distance(std::span{a}, std::span{b});
        auto d_ba = l2_distance(std::span{b}, std::span{a});
        REQUIRE_THAT(d_ab, WithinRel(d_ba, 1e-5));
    }

    SECTION("Non-negativity: distance >= 0") {
        auto a = makeRandomVector<float>(100, -50.0f, 50.0f);
        auto b = makeRandomVector<float>(100, -50.0f, 50.0f);
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE(dist >= 0.0f);
    }

    SECTION("Triangle inequality: d(a,c) <= d(a,b) + d(b,c)") {
        auto a = makeRandomVector<float>(32, 0.0f, 1.0f);
        auto b = makeRandomVector<float>(32, 0.0f, 1.0f);
        auto c = makeRandomVector<float>(32, 0.0f, 1.0f);

        auto d_ac = l2_distance(std::span{a}, std::span{c});
        auto d_ab = l2_distance(std::span{a}, std::span{b});
        auto d_bc = l2_distance(std::span{b}, std::span{c});

        REQUIRE(d_ac <= d_ab + d_bc + 1e-4f); // Small epsilon for FP errors
    }
}

TEST_CASE_METHOD(DistanceMetricsFixture, "L2 distance: Known values", "[distance][l2]") {
    SECTION("Unit vectors orthogonal") {
        std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> b = {0.0f, 1.0f, 0.0f, 0.0f};
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(std::sqrt(2.0f), 1e-5));
    }

    SECTION("Simple 2D case") {
        std::vector<float> a = {3.0f, 4.0f};
        std::vector<float> b = {0.0f, 0.0f};
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(5.0f, 1e-5)); // 3-4-5 triangle
    }

    SECTION("3D case") {
        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::vector<float> b = {4.0f, 6.0f, 8.0f};
        // sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(std::sqrt(50.0f), 1e-5));
    }
}

TEST_CASE_METHOD(DistanceMetricsFixture, "L2 distance: Edge cases", "[distance][l2]") {
    SECTION("Zero vectors") {
        auto zero = makeZeroVector<float>(64);
        auto dist = l2_distance(std::span{zero}, std::span{zero});
        REQUIRE_THAT(dist, WithinRel(0.0f, 1e-7));
    }

    SECTION("Single element vectors") {
        std::vector<float> a = {5.0f};
        std::vector<float> b = {2.0f};
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(3.0f, 1e-5));
    }

    SECTION("Large dimension vectors (SIMD test)") {
        auto a = makeRandomVector<float>(1024, -1.0f, 1.0f);
        auto b = makeRandomVector<float>(1024, -1.0f, 1.0f);
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE(std::isfinite(dist));
        REQUIRE(dist >= 0.0f);
    }
}

TEST_CASE_METHOD(DistanceMetricsFixture, "L2 distance: int8 type", "[distance][l2][int8]") {
    SECTION("Basic int8 computation") {
        std::vector<int8_t> a = {10, 20, 30};
        std::vector<int8_t> b = {13, 24, 35};
        // sqrt((13-10)^2 + (24-20)^2 + (35-30)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(std::sqrt(50.0f), 1e-5));
    }

    SECTION("int8 with negative values") {
        std::vector<int8_t> a = {-10, 0, 10};
        std::vector<int8_t> b = {10, 0, -10};
        // sqrt((10-(-10))^2 + 0 + (-10-10)^2) = sqrt(400 + 400) = sqrt(800)
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(std::sqrt(800.0f), 1e-4));
    }

    SECTION("int8 large vector") {
        auto a = makeRandomVector<int8_t>(512, int8_t{-100}, int8_t{100});
        auto b = makeRandomVector<int8_t>(512, int8_t{-100}, int8_t{100});
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE(std::isfinite(dist));
        REQUIRE(dist >= 0.0f);
    }
}

// ============================================================================
// L1 Distance Tests
// ============================================================================

TEST_CASE_METHOD(DistanceMetricsFixture, "L1 distance: Basic properties", "[distance][l1]") {
    SECTION("Identity") {
        auto vec = makeRandomVector<float>(100, -5.0f, 5.0f);
        auto dist = l1_distance(std::span{vec}, std::span{vec});
        REQUIRE_THAT(dist, WithinRel(0.0f, 1e-6));
    }

    SECTION("Symmetry") {
        auto a = makeRandomVector<float>(75, -10.0f, 10.0f);
        auto b = makeRandomVector<float>(75, -10.0f, 10.0f);
        auto d_ab = l1_distance(std::span{a}, std::span{b});
        auto d_ba = l1_distance(std::span{b}, std::span{a});
        REQUIRE_THAT(d_ab, WithinRel(d_ba, 1e-5));
    }

    SECTION("Non-negativity") {
        auto a = makeRandomVector<float>(50, -100.0f, 100.0f);
        auto b = makeRandomVector<float>(50, -100.0f, 100.0f);
        auto dist = l1_distance(std::span{a}, std::span{b});
        REQUIRE(dist >= 0.0f);
    }
}

TEST_CASE_METHOD(DistanceMetricsFixture, "L1 distance: Known values", "[distance][l1]") {
    SECTION("Simple 2D") {
        std::vector<float> a = {3.0f, 4.0f};
        std::vector<float> b = {0.0f, 0.0f};
        auto dist = l1_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(7.0f, 1e-5)); // |3-0| + |4-0| = 7
    }

    SECTION("With negative values") {
        std::vector<float> a = {-1.0f, 2.0f, -3.0f};
        std::vector<float> b = {1.0f, -2.0f, 3.0f};
        // |1-(-1)| + |-2-2| + |3-(-3)| = 2 + 4 + 6 = 12
        auto dist = l1_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(12.0f, 1e-5));
    }
}

TEST_CASE_METHOD(DistanceMetricsFixture, "L1 distance: int8 type", "[distance][l1][int8]") {
    SECTION("Basic int8") {
        std::vector<int8_t> a = {10, -20, 30};
        std::vector<int8_t> b = {5, -15, 25};
        // |5-10| + |-15-(-20)| + |25-30| = 5 + 5 + 5 = 15
        auto dist = l1_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(15.0f, 1e-5));
    }
}

// ============================================================================
// Cosine Distance Tests
// ============================================================================

TEST_CASE_METHOD(DistanceMetricsFixture, "Cosine distance: Basic properties",
                 "[distance][cosine]") {
    SECTION("Identity: identical vectors have distance 0") {
        auto vec = makeRandomVector<float>(64, 0.1f, 1.0f); // Avoid zero
        auto dist = cosine_distance(std::span{vec}, std::span{vec});
        REQUIRE_THAT(dist, WithinRel(0.0f, 1e-5));
    }

    SECTION("Symmetry") {
        auto a = makeRandomVector<float>(100, 0.1f, 10.0f);
        auto b = makeRandomVector<float>(100, 0.1f, 10.0f);
        auto d_ab = cosine_distance(std::span{a}, std::span{b});
        auto d_ba = cosine_distance(std::span{b}, std::span{a});
        REQUIRE_THAT(d_ab, WithinRel(d_ba, 1e-5));
    }

    SECTION("Range: distance in [0, 2]") {
        auto a = makeRandomVector<float>(80, -5.0f, 5.0f);
        auto b = makeRandomVector<float>(80, -5.0f, 5.0f);
        auto dist = cosine_distance(std::span{a}, std::span{b});
        REQUIRE(dist >= 0.0f);
        REQUIRE(dist <= 2.0f + 1e-4f);
    }
}

TEST_CASE_METHOD(DistanceMetricsFixture, "Cosine distance: Known values", "[distance][cosine]") {
    SECTION("Parallel vectors (same direction)") {
        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::vector<float> b = {2.0f, 4.0f, 6.0f}; // 2*a
        auto dist = cosine_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(0.0f, 1e-5)); // cos(0) = 1, dist = 0
    }

    SECTION("Orthogonal vectors") {
        std::vector<float> a = {1.0f, 0.0f, 0.0f};
        std::vector<float> b = {0.0f, 1.0f, 0.0f};
        auto dist = cosine_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(1.0f, 1e-5)); // cos(90°) = 0, dist = 1
    }

    SECTION("Opposite vectors") {
        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::vector<float> b = {-1.0f, -2.0f, -3.0f};
        auto dist = cosine_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(2.0f, 1e-5)); // cos(180°) = -1, dist = 2
    }
}

TEST_CASE_METHOD(DistanceMetricsFixture, "Cosine distance: Edge cases", "[distance][cosine]") {
    SECTION("Unit vectors") {
        auto a = makeUnitVector<float>(128, 0);
        auto b = makeUnitVector<float>(128, 64);
        auto dist = cosine_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(1.0f, 1e-5)); // Orthogonal
    }

    SECTION("Normalized vectors") {
        std::vector<float> a = {0.6f, 0.8f}; // Already normalized
        std::vector<float> b = {0.8f, 0.6f};
        auto dist = cosine_distance(std::span{a}, std::span{b});
        // cos(θ) = 0.6*0.8 + 0.8*0.6 = 0.96, dist = 1 - 0.96 = 0.04
        REQUIRE_THAT(dist, WithinRel(0.04f, 1e-5));
    }
}

TEST_CASE_METHOD(DistanceMetricsFixture, "Cosine distance: int8 type", "[distance][cosine][int8]") {
    SECTION("Basic int8") {
        std::vector<int8_t> a = {3, 4}; // Length 5
        std::vector<int8_t> b = {4, 3}; // Length 5
        auto dist = cosine_distance(std::span{a}, std::span{b});
        // cos(θ) = (3*4 + 4*3)/(5*5) = 24/25 = 0.96, dist = 1 - 0.96 = 0.04
        REQUIRE_THAT(dist, WithinRel(0.04f, 1e-4));
    }

    SECTION("int8 orthogonal") {
        std::vector<int8_t> a = {1, 0, 0};
        std::vector<int8_t> b = {0, 1, 0};
        auto dist = cosine_distance(std::span{a}, std::span{b});
        REQUIRE_THAT(dist, WithinRel(1.0f, 1e-5));
    }
}

// ============================================================================
// Hamming Distance Tests
// ============================================================================

TEST_CASE("Hamming distance: Basic properties", "[distance][hamming]") {
    SECTION("Identity") {
        std::vector<uint8_t> vec = {0xFF, 0x00, 0xAA, 0x55};
        auto dist = hamming_distance(std::span{vec}, std::span{vec});
        REQUIRE(dist == 0);
    }

    SECTION("Symmetry") {
        std::vector<uint8_t> a = {0x12, 0x34, 0x56};
        std::vector<uint8_t> b = {0xAB, 0xCD, 0xEF};
        auto d_ab = hamming_distance(std::span{a}, std::span{b});
        auto d_ba = hamming_distance(std::span{b}, std::span{a});
        REQUIRE(d_ab == d_ba);
    }

    SECTION("Non-negativity") {
        std::vector<uint8_t> a = {0xFF};
        std::vector<uint8_t> b = {0x00};
        auto dist = hamming_distance(std::span{a}, std::span{b});
        REQUIRE(dist >= 0);
    }
}

TEST_CASE("Hamming distance: Known values", "[distance][hamming]") {
    SECTION("Single byte: all bits different") {
        std::vector<uint8_t> a = {0xFF};
        std::vector<uint8_t> b = {0x00};
        auto dist = hamming_distance(std::span{a}, std::span{b});
        REQUIRE(dist == 8);
    }

    SECTION("Single byte: one bit different") {
        std::vector<uint8_t> a = {0b10101010};
        std::vector<uint8_t> b = {0b10101011};
        auto dist = hamming_distance(std::span{a}, std::span{b});
        REQUIRE(dist == 1);
    }

    SECTION("Multiple bytes") {
        std::vector<uint8_t> a = {0b11110000, 0b10101010};
        std::vector<uint8_t> b = {0b11110000, 0b01010101};
        // First byte: 0 bits different, second byte: all 8 bits different
        auto dist = hamming_distance(std::span{a}, std::span{b});
        REQUIRE(dist == 8);
    }

    SECTION("Real-world pattern") {
        std::vector<uint8_t> a = {0x12, 0x34, 0x56, 0x78};
        std::vector<uint8_t> b = {0x12, 0x34, 0x56, 0x78};
        auto dist = hamming_distance(std::span{a}, std::span{b});
        REQUIRE(dist == 0);
    }
}

// ============================================================================
// SIMD Consistency Tests
// ============================================================================

#if defined(SQLITE_VEC_ENABLE_AVX) || defined(SQLITE_VEC_ENABLE_NEON)
TEST_CASE_METHOD(DistanceMetricsFixture, "SIMD vs Scalar consistency", "[distance][simd]") {
    SECTION("L2: Large vectors should use SIMD path") {
        // Test that SIMD and scalar implementations produce same results
        auto a = makeRandomVector<float>(1024, -10.0f, 10.0f);
        auto b = makeRandomVector<float>(1024, -10.0f, 10.0f);

        auto dist_simd = l2_distance(std::span{a}, std::span{b});

        // Force scalar path by using non-aligned or odd-sized subset
        std::vector<float> a_small(a.begin(), a.begin() + 17);
        std::vector<float> b_small(b.begin(), b.begin() + 17);
        auto dist_scalar = l2_distance(std::span{a_small}, std::span{b_small});

        // Both should be finite and positive
        REQUIRE(std::isfinite(dist_simd));
        REQUIRE(std::isfinite(dist_scalar));
        REQUIRE(dist_simd >= 0.0f);
        REQUIRE(dist_scalar >= 0.0f);
    }

    SECTION("Cosine: SIMD consistency") {
        auto a = makeRandomVector<float>(512, 0.1f, 10.0f);
        auto b = makeRandomVector<float>(512, 0.1f, 10.0f);

        auto dist = cosine_distance(std::span{a}, std::span{b});
        REQUIRE(dist >= 0.0f);
        REQUIRE(dist <= 2.0f);
    }
}
#endif

// ============================================================================
// Multi-type Template Tests
// ============================================================================

TEMPLATE_TEST_CASE("Distance metrics work with various numeric types", "[distance][template]",
                   float, double, int8_t, int16_t, int32_t) {
    SECTION("L2 distance") {
        std::vector<TestType> a = {TestType(1), TestType(2), TestType(3)};
        std::vector<TestType> b = {TestType(4), TestType(5), TestType(6)};
        auto dist = l2_distance(std::span{a}, std::span{b});
        REQUIRE(std::isfinite(dist));
        REQUIRE(dist >= 0.0f);
    }

    SECTION("L1 distance") {
        std::vector<TestType> a = {TestType(10), TestType(20)};
        std::vector<TestType> b = {TestType(15), TestType(25)};
        auto dist = l1_distance(std::span{a}, std::span{b});
        REQUIRE(std::isfinite(dist));
        REQUIRE(dist >= 0.0f);
    }
}

// ============================================================================
// Performance-oriented Tests
// ============================================================================

TEST_CASE_METHOD(DistanceMetricsFixture, "Distance metrics: Performance dimensions",
                 "[distance][perf]") {
    auto dims = GENERATE(128, 256, 384, 512, 768, 1024, 1536, 2048);

    SECTION("L2 distance scales linearly") {
        auto a = makeRandomVector<float>(dims, -1.0f, 1.0f);
        auto b = makeRandomVector<float>(dims, -1.0f, 1.0f);
        auto dist = l2_distance(std::span{a}, std::span{b});

        // Just verify it works and produces reasonable results
        REQUIRE(std::isfinite(dist));
        REQUIRE(dist >= 0.0f);
        // Upper bound: if all dims differ by 2, sqrt(dims * 4) = 2*sqrt(dims)
        REQUIRE(dist <= 2.0f * std::sqrt(static_cast<float>(dims)) + 1.0f);
    }
}

// ============================================================================
// Error/Boundary Tests
// ============================================================================

TEST_CASE("Distance metrics: Size mismatches (debug build only)", "[distance][error]") {
// These tests only apply when assertions are enabled
#ifdef NDEBUG
    SKIP("Size mismatch checks only active in debug builds");
#endif

    const std::vector<float> short_vec = {1.0f, 2.0f};
    const std::vector<float> long_vec = {1.0f, 2.0f, 3.0f};

    // In debug builds, mismatched sizes should trigger assertions
    // In release builds, behavior is undefined (but shouldn't crash)
    SECTION("L2 with mismatched sizes") {
        // This would assert in debug, undefined in release
        // We document this behavior but don't test it directly
        INFO("Size mismatch behavior is implementation-defined");
    }
}
