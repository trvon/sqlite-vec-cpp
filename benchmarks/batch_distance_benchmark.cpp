// SPDX-License-Identifier: Apache-2.0 OR MIT
// Benchmark: Batch distance vs Sequential distance

#include <random>
#include <vector>
#include "../include/sqlite-vec-cpp/distances/batch.hpp"
#include "../include/sqlite-vec-cpp/distances/l2.hpp"
#include <benchmark/benchmark.h>

using namespace sqlite_vec_cpp::distances;

// Generate random vector
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
// Sequential vs Batch: Small Database (100 vectors)
// ============================================================================

static void BM_Sequential_100x384(benchmark::State& state) {
    const size_t num_vectors = 100;
    const size_t dim = 384;

    auto query = generate_random_vector<float>(dim);
    std::vector<std::vector<float>> database;
    for (size_t i = 0; i < num_vectors; ++i) {
        database.push_back(generate_random_vector<float>(dim));
    }

    for (auto _ : state) {
        std::vector<float> distances;
        distances.reserve(num_vectors);
        for (const auto& db_vec : database) {
            float dist = l2_distance(std::span<const float>{query}, std::span<const float>{db_vec});
            distances.push_back(dist);
            benchmark::DoNotOptimize(dist);
        }
        benchmark::DoNotOptimize(distances);
    }
    state.SetItemsProcessed(state.iterations() * num_vectors);
}
BENCHMARK(BM_Sequential_100x384);

static void BM_Batch_100x384(benchmark::State& state) {
    const size_t num_vectors = 100;
    const size_t dim = 384;

    auto query = generate_random_vector<float>(dim);
    std::vector<std::vector<float>> database;
    std::vector<std::span<const float>> db_spans;
    for (size_t i = 0; i < num_vectors; ++i) {
        database.push_back(generate_random_vector<float>(dim));
        db_spans.emplace_back(database.back());
    }

    for (auto _ : state) {
        auto distances = batch::batch_distance(std::span<const float>{query},
                                               std::span<const std::span<const float>>{db_spans},
                                               L2Metric<float>{});
        benchmark::DoNotOptimize(distances);
    }
    state.SetItemsProcessed(state.iterations() * num_vectors);
}
BENCHMARK(BM_Batch_100x384);

// ============================================================================
// Sequential vs Batch: Medium Database (1K vectors)
// ============================================================================

static void BM_Sequential_1Kx384(benchmark::State& state) {
    const size_t num_vectors = 1000;
    const size_t dim = 384;

    auto query = generate_random_vector<float>(dim);
    std::vector<std::vector<float>> database;
    for (size_t i = 0; i < num_vectors; ++i) {
        database.push_back(generate_random_vector<float>(dim));
    }

    for (auto _ : state) {
        std::vector<float> distances;
        distances.reserve(num_vectors);
        for (const auto& db_vec : database) {
            float dist = l2_distance(std::span<const float>{query}, std::span<const float>{db_vec});
            distances.push_back(dist);
            benchmark::DoNotOptimize(dist);
        }
        benchmark::DoNotOptimize(distances);
    }
    state.SetItemsProcessed(state.iterations() * num_vectors);
}
BENCHMARK(BM_Sequential_1Kx384);

static void BM_Batch_1Kx384(benchmark::State& state) {
    const size_t num_vectors = 1000;
    const size_t dim = 384;

    auto query = generate_random_vector<float>(dim);
    std::vector<std::vector<float>> database;
    std::vector<std::span<const float>> db_spans;
    for (size_t i = 0; i < num_vectors; ++i) {
        database.push_back(generate_random_vector<float>(dim));
        db_spans.emplace_back(database.back());
    }

    for (auto _ : state) {
        auto distances = batch::batch_distance(std::span<const float>{query},
                                               std::span<const std::span<const float>>{db_spans},
                                               L2Metric<float>{});
        benchmark::DoNotOptimize(distances);
    }
    state.SetItemsProcessed(state.iterations() * num_vectors);
}
BENCHMARK(BM_Batch_1Kx384);

// ============================================================================
// Top-K Performance
// ============================================================================

static void BM_TopK_1Kx384_K10(benchmark::State& state) {
    const size_t num_vectors = 1000;
    const size_t dim = 384;
    const size_t k = 10;

    auto query = generate_random_vector<float>(dim);
    std::vector<std::vector<float>> database;
    std::vector<std::span<const float>> db_spans;
    for (size_t i = 0; i < num_vectors; ++i) {
        database.push_back(generate_random_vector<float>(dim));
        db_spans.emplace_back(database.back());
    }

    for (auto _ : state) {
        auto top_k = batch::batch_top_k(std::span<const float>{query},
                                        std::span<const std::span<const float>>{db_spans}, k,
                                        L2Metric<float>{});
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * num_vectors);
}
BENCHMARK(BM_TopK_1Kx384_K10);

// ============================================================================
// Contiguous Layout Performance
// ============================================================================

static void BM_Batch_Contiguous_1Kx384(benchmark::State& state) {
    const size_t num_vectors = 1000;
    const size_t dim = 384;

    auto query = generate_random_vector<float>(dim);

    // Flatten database into contiguous array
    std::vector<float> database_flat;
    database_flat.reserve(num_vectors * dim);
    for (size_t i = 0; i < num_vectors; ++i) {
        auto vec = generate_random_vector<float>(dim);
        database_flat.insert(database_flat.end(), vec.begin(), vec.end());
    }

    for (auto _ : state) {
        auto distances = batch::batch_distance_contiguous(std::span<const float>{query},
                                                          std::span<const float>{database_flat},
                                                          num_vectors, dim, L2Metric<float>{});
        benchmark::DoNotOptimize(distances);
    }
    state.SetItemsProcessed(state.iterations() * num_vectors);
}
BENCHMARK(BM_Batch_Contiguous_1Kx384);

// ============================================================================
// int8 Quantized Vectors
// ============================================================================

static void BM_Batch_Int8_1Kx384(benchmark::State& state) {
    const size_t num_vectors = 1000;
    const size_t dim = 384;

    auto query = generate_random_vector<int8_t>(dim, int8_t{-128}, int8_t{127});
    std::vector<std::vector<int8_t>> database;
    std::vector<std::span<const int8_t>> db_spans;
    for (size_t i = 0; i < num_vectors; ++i) {
        database.push_back(generate_random_vector<int8_t>(dim, int8_t{-128}, int8_t{127}));
        db_spans.emplace_back(database.back());
    }

    for (auto _ : state) {
        auto distances = batch::batch_distance(std::span<const int8_t>{query},
                                               std::span<const std::span<const int8_t>>{db_spans},
                                               L2Metric<int8_t>{});
        benchmark::DoNotOptimize(distances);
    }
    state.SetItemsProcessed(state.iterations() * num_vectors);
}
BENCHMARK(BM_Batch_Int8_1Kx384);

// ============================================================================
// Large embeddings (1536d - text-embedding-3-small)
// ============================================================================

static void BM_Batch_1Kx1536(benchmark::State& state) {
    const size_t num_vectors = 1000;
    const size_t dim = 1536;

    auto query = generate_random_vector<float>(dim);
    std::vector<std::vector<float>> database;
    std::vector<std::span<const float>> db_spans;
    for (size_t i = 0; i < num_vectors; ++i) {
        database.push_back(generate_random_vector<float>(dim));
        db_spans.emplace_back(database.back());
    }

    for (auto _ : state) {
        auto distances = batch::batch_distance(std::span<const float>{query},
                                               std::span<const std::span<const float>>{db_spans},
                                               L2Metric<float>{});
        benchmark::DoNotOptimize(distances);
    }
    state.SetItemsProcessed(state.iterations() * num_vectors);
}
BENCHMARK(BM_Batch_1Kx1536);

BENCHMARK_MAIN();
