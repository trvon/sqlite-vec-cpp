// SPDX-License-Identifier: Apache-2.0 OR MIT
// Benchmark: filtered search overhead in HNSW

#include <random>
#include <unordered_set>
#include <vector>
#include <benchmark/benchmark.h>
#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>

using namespace sqlite_vec_cpp::distances;
using namespace sqlite_vec_cpp::index;

static std::vector<float> generate_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(rng);
    }
    return vec;
}

struct FilterBenchData {
    HNSWIndex<float, L2Metric<float>> index;
    std::vector<std::vector<float>> queries;
    std::vector<uint8_t> allowed;
    std::unordered_set<size_t> allowed_set;
};

static FilterBenchData build_data(size_t corpus_size, size_t dim, size_t num_queries,
                                  uint32_t allowed_pct) {
    std::mt19937 rng(42);
    FilterBenchData data;

    // Build index
    std::vector<std::vector<float>> vectors;
    vectors.reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        data.index.insert(i, std::span<const float>{vectors.back()});
    }

    // Queries
    data.queries.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        data.queries.push_back(generate_vector(dim, rng));
    }

    // Allowed mask + set
    data.allowed.assign(corpus_size, 0);
    std::uniform_int_distribution<int> pct(0, 99);
    for (size_t i = 0; i < corpus_size; ++i) {
        if (static_cast<uint32_t>(pct(rng)) < allowed_pct) {
            data.allowed[i] = 1;
            data.allowed_set.insert(i);
        }
    }

    return data;
}

static void BM_HNSW_Search_NoFilter(benchmark::State& state) {
    constexpr size_t k = 10;
    constexpr size_t ef = 50;
    constexpr size_t corpus_size = 10000;
    constexpr size_t dim = 128;
    constexpr size_t num_queries = 100;

    auto data = build_data(corpus_size, dim, num_queries, 100);

    for (auto _ : state) {
        for (const auto& q : data.queries) {
            auto results = data.index.search(std::span<const float>{q}, k, ef);
            benchmark::DoNotOptimize(results);
        }
    }

    state.SetItemsProcessed(state.iterations() * num_queries);
}

static void BM_HNSW_Search_Filter_Bitset(benchmark::State& state) {
    constexpr size_t k = 10;
    constexpr size_t ef = 50;
    constexpr size_t corpus_size = 10000;
    constexpr size_t dim = 128;
    constexpr size_t num_queries = 100;
    uint32_t allowed_pct = static_cast<uint32_t>(state.range(0));

    auto data = build_data(corpus_size, dim, num_queries, allowed_pct);

    for (auto _ : state) {
        for (const auto& q : data.queries) {
            auto results =
                data.index.search_with_filter(std::span<const float>{q}, k, ef,
                                              [&](size_t id) { return data.allowed[id] != 0; });
            benchmark::DoNotOptimize(results);
        }
    }

    state.counters["allowed_pct"] = allowed_pct;
    state.SetItemsProcessed(state.iterations() * num_queries);
}

static void BM_HNSW_Search_Filter_Set(benchmark::State& state) {
    constexpr size_t k = 10;
    constexpr size_t ef = 50;
    constexpr size_t corpus_size = 10000;
    constexpr size_t dim = 128;
    constexpr size_t num_queries = 100;
    uint32_t allowed_pct = static_cast<uint32_t>(state.range(0));

    auto data = build_data(corpus_size, dim, num_queries, allowed_pct);

    for (auto _ : state) {
        for (const auto& q : data.queries) {
            auto results = data.index.search_with_filter(
                std::span<const float>{q}, k, ef,
                [&](size_t id) { return data.allowed_set.contains(id); });
            benchmark::DoNotOptimize(results);
        }
    }

    state.counters["allowed_pct"] = allowed_pct;
    state.SetItemsProcessed(state.iterations() * num_queries);
}

BENCHMARK(BM_HNSW_Search_NoFilter);
BENCHMARK(BM_HNSW_Search_Filter_Bitset)->Arg(10)->Arg(50)->Arg(90);
BENCHMARK(BM_HNSW_Search_Filter_Set)->Arg(10)->Arg(50)->Arg(90);

BENCHMARK_MAIN();
