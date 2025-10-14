// SPDX-License-Identifier: Apache-2.0 OR MIT
// Copyright 2025 - Present, Trevon Hanna

#include <random>
#include <vector>
#include "vec_core.hpp"
#include <benchmark/benchmark.h>

namespace sqlite_vec {

// Random vector generator
template <typename T>
std::vector<T> generate_random_vector(size_t dim, T min = T{0}, T max = T{1}) {
    std::mt19937 gen(42); // Fixed seed for reproducibility

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min, max);
        std::vector<T> vec(dim);
        std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
        return vec;
    } else {
        std::uniform_int_distribution<int> dist(static_cast<int>(min), static_cast<int>(max));
        std::vector<T> vec(dim);
        std::generate(vec.begin(), vec.end(), [&]() { return static_cast<T>(dist(gen)); });
        return vec;
    }
}

// ============================================================================
// L2 Distance Benchmarks
// ============================================================================

static void BM_L2_Float_128(benchmark::State& state) {
    auto a = generate_random_vector<float>(128);
    auto b = generate_random_vector<float>(128);

    for (auto _ : state) {
        float result =
            distance_l2_sqeuclidean(std::span<const float>(a), std::span<const float>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 128);
}
BENCHMARK(BM_L2_Float_128);

static void BM_L2_Float_256(benchmark::State& state) {
    auto a = generate_random_vector<float>(256);
    auto b = generate_random_vector<float>(256);

    for (auto _ : state) {
        float result =
            distance_l2_sqeuclidean(std::span<const float>(a), std::span<const float>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 256);
}
BENCHMARK(BM_L2_Float_256);

static void BM_L2_Float_512(benchmark::State& state) {
    auto a = generate_random_vector<float>(512);
    auto b = generate_random_vector<float>(512);

    for (auto _ : state) {
        float result =
            distance_l2_sqeuclidean(std::span<const float>(a), std::span<const float>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 512);
}
BENCHMARK(BM_L2_Float_512);

static void BM_L2_Float_1536(benchmark::State& state) {
    auto a = generate_random_vector<float>(1536); // OpenAI embedding size
    auto b = generate_random_vector<float>(1536);

    for (auto _ : state) {
        float result =
            distance_l2_sqeuclidean(std::span<const float>(a), std::span<const float>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 1536);
}
BENCHMARK(BM_L2_Float_1536);

static void BM_L2_Int8_128(benchmark::State& state) {
    auto a = generate_random_vector<int8_t>(128, int8_t{-128}, int8_t{127});
    auto b = generate_random_vector<int8_t>(128, int8_t{-128}, int8_t{127});

    for (auto _ : state) {
        float result =
            distance_l2_sqeuclidean(std::span<const int8_t>(a), std::span<const int8_t>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 128);
}
BENCHMARK(BM_L2_Int8_128);

// ============================================================================
// L1 Distance Benchmarks
// ============================================================================

static void BM_L1_Float_128(benchmark::State& state) {
    auto a = generate_random_vector<float>(128);
    auto b = generate_random_vector<float>(128);

    for (auto _ : state) {
        float result = distance_l1(std::span<const float>(a), std::span<const float>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 128);
}
BENCHMARK(BM_L1_Float_128);

static void BM_L1_Float_1536(benchmark::State& state) {
    auto a = generate_random_vector<float>(1536);
    auto b = generate_random_vector<float>(1536);

    for (auto _ : state) {
        float result = distance_l1(std::span<const float>(a), std::span<const float>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 1536);
}
BENCHMARK(BM_L1_Float_1536);

// ============================================================================
// Cosine Distance Benchmarks
// ============================================================================

static void BM_Cosine_Float_128(benchmark::State& state) {
    auto a = generate_random_vector<float>(128);
    auto b = generate_random_vector<float>(128);

    for (auto _ : state) {
        float result = distance_cosine(std::span<const float>(a), std::span<const float>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 128);
}
BENCHMARK(BM_Cosine_Float_128);

static void BM_Cosine_Float_1536(benchmark::State& state) {
    auto a = generate_random_vector<float>(1536);
    auto b = generate_random_vector<float>(1536);

    for (auto _ : state) {
        float result = distance_cosine(std::span<const float>(a), std::span<const float>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 1536);
}
BENCHMARK(BM_Cosine_Float_1536);

// ============================================================================
// Hamming Distance Benchmarks
// ============================================================================

static void BM_Hamming_128(benchmark::State& state) {
    auto a = generate_random_vector<unsigned char>(128, (unsigned char)0, (unsigned char)255);
    auto b = generate_random_vector<unsigned char>(128, (unsigned char)0, (unsigned char)255);

    for (auto _ : state) {
        int result =
            distance_hamming(std::span<const unsigned char>(a), std::span<const unsigned char>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 128);
}
BENCHMARK(BM_Hamming_128);

static void BM_Hamming_1536(benchmark::State& state) {
    auto a = generate_random_vector<unsigned char>(1536, (unsigned char)0, (unsigned char)255);
    auto b = generate_random_vector<unsigned char>(1536, (unsigned char)0, (unsigned char)255);

    for (auto _ : state) {
        int result =
            distance_hamming(std::span<const unsigned char>(a), std::span<const unsigned char>(b));
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * 1536);
}
BENCHMARK(BM_Hamming_1536);

// ============================================================================
// Batch Operations (if we implement 057-104)
// ============================================================================

static void BM_L2_Float_Batch_1000x128(benchmark::State& state) {
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 1000; ++i) {
        queries.push_back(generate_random_vector<float>(128));
    }
    auto target = generate_random_vector<float>(128);

    for (auto _ : state) {
        for (const auto& query : queries) {
            float result = distance_l2_sqeuclidean(std::span<const float>(query),
                                                   std::span<const float>(target));
            benchmark::DoNotOptimize(result);
        }
    }
    state.SetItemsProcessed(state.iterations() * 1000 * 128);
}
BENCHMARK(BM_L2_Float_Batch_1000x128);

} // namespace sqlite_vec

BENCHMARK_MAIN();
