// SPDX-License-Identifier: MIT
// Copyright (c) 2025 YAMS Contributors
// Catch2-based benchmarks for sqlite-vec-cpp (simpler integration)

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../include/sqlite-vec-cpp/distances.hpp"
#include "../include/sqlite-vec-cpp/vector_view.hpp"

#include <random>
#include <vector>

using namespace sqlite_vec_cpp;

// ============================================================================
// Benchmark Helper
// ============================================================================

template <typename T> std::vector<T> makeRandomVector(size_t dims, T min_val, T max_val) {
    static std::mt19937 rng{42};
    std::vector<T> vec(dims);
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        std::generate(vec.begin(), vec.end(), [&]() { return dist(rng); });
    } else {
        std::uniform_int_distribution<int> dist(min_val, max_val);
        std::generate(vec.begin(), vec.end(), [&]() { return static_cast<T>(dist(rng)); });
    }
    return vec;
}

// ============================================================================
// L2 Distance Benchmarks
// ============================================================================

TEST_CASE("Benchmark: L2 Distance", "[!benchmark][l2]") {
    BENCHMARK("L2 float 128d") {
        auto a = makeRandomVector<float>(128, -1.0f, 1.0f);
        auto b = makeRandomVector<float>(128, -1.0f, 1.0f);
        return l2_distance(std::span{a}, std::span{b});
    };

    BENCHMARK("L2 float 384d (ada-002 size)") {
        auto a = makeRandomVector<float>(384, -1.0f, 1.0f);
        auto b = makeRandomVector<float>(384, -1.0f, 1.0f);
        return l2_distance(std::span{a}, std::span{b});
    };

    BENCHMARK("L2 float 1536d (text-embedding-3-small)") {
        auto a = makeRandomVector<float>(1536, -1.0f, 1.0f);
        auto b = makeRandomVector<float>(1536, -1.0f, 1.0f);
        return l2_distance(std::span{a}, std::span{b});
    };

    BENCHMARK("L2 int8 384d (quantized)") {
        auto a = makeRandomVector<int8_t>(384, int8_t{-128}, int8_t{127});
        auto b = makeRandomVector<int8_t>(384, int8_t{-128}, int8_t{127});
        return l2_distance(std::span{a}, std::span{b});
    };
}

// ============================================================================
// L1 Distance Benchmarks
// ============================================================================

TEST_CASE("Benchmark: L1 Distance", "[!benchmark][l1]") {
    BENCHMARK("L1 float 128d") {
        auto a = makeRandomVector<float>(128, -1.0f, 1.0f);
        auto b = makeRandomVector<float>(128, -1.0f, 1.0f);
        return l1_distance(std::span{a}, std::span{b});
    };

    BENCHMARK("L1 float 384d") {
        auto a = makeRandomVector<float>(384, -1.0f, 1.0f);
        auto b = makeRandomVector<float>(384, -1.0f, 1.0f);
        return l1_distance(std::span{a}, std::span{b});
    };

    BENCHMARK("L1 int8 384d") {
        auto a = makeRandomVector<int8_t>(384, int8_t{-128}, int8_t{127});
        auto b = makeRandomVector<int8_t>(384, int8_t{-128}, int8_t{127});
        return l1_distance(std::span{a}, std::span{b});
    };
}

// ============================================================================
// Cosine Distance Benchmarks
// ============================================================================

TEST_CASE("Benchmark: Cosine Distance", "[!benchmark][cosine]") {
    BENCHMARK("Cosine float 128d") {
        auto a = makeRandomVector<float>(128, 0.01f, 1.0f);
        auto b = makeRandomVector<float>(128, 0.01f, 1.0f);
        return cosine_distance(std::span{a}, std::span{b});
    };

    BENCHMARK("Cosine float 384d") {
        auto a = makeRandomVector<float>(384, 0.01f, 1.0f);
        auto b = makeRandomVector<float>(384, 0.01f, 1.0f);
        return cosine_distance(std::span{a}, std::span{b});
    };

    BENCHMARK("Cosine float 1536d") {
        auto a = makeRandomVector<float>(1536, 0.01f, 1.0f);
        auto b = makeRandomVector<float>(1536, 0.01f, 1.0f);
        return cosine_distance(std::span{a}, std::span{b});
    };
}

// ============================================================================
// Hamming Distance Benchmarks
// ============================================================================

TEST_CASE("Benchmark: Hamming Distance", "[!benchmark][hamming]") {
    BENCHMARK("Hamming 128 bytes") {
        auto a = makeRandomVector<uint8_t>(128, uint8_t{0}, uint8_t{255});
        auto b = makeRandomVector<uint8_t>(128, uint8_t{0}, uint8_t{255});
        return hamming_distance(std::span{a}, std::span{b});
    };

    BENCHMARK("Hamming 384 bytes") {
        auto a = makeRandomVector<uint8_t>(384, uint8_t{0}, uint8_t{255});
        auto b = makeRandomVector<uint8_t>(384, uint8_t{0}, uint8_t{255});
        return hamming_distance(std::span{a}, std::span{b});
    };
}

// ============================================================================
// Batch Operations
// ============================================================================

TEST_CASE("Benchmark: Batch Operations", "[!benchmark][batch]") {
    BENCHMARK("L2: 1 vs 100 vectors (384d)") {
        auto query = makeRandomVector<float>(384, -1.0f, 1.0f);
        std::vector<std::vector<float>> batch;
        for (int i = 0; i < 100; ++i) {
            batch.push_back(makeRandomVector<float>(384, -1.0f, 1.0f));
        }

        return std::accumulate(batch.begin(), batch.end(), 0.0f, [&](float acc, const auto& vec) {
            return acc + l2_distance(std::span{query}, std::span{vec});
        });
    };
}
