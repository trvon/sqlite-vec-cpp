// SPDX-License-Identifier: Apache-2.0 OR MIT
// Benchmark for HNSW index performance

#include <random>
#include <span>
#include <vector>
#include <benchmark/benchmark.h>
#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

// ============================================================================
// Helpers
// ============================================================================

std::vector<float> generate_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(rng);
    }
    return vec;
}

// ============================================================================
// Benchmark: HNSW Index Build
// ============================================================================

static void BM_HNSW_Build(benchmark::State& state) {
    size_t num_vectors = state.range(0);
    size_t dim = state.range(1);
    std::mt19937 rng(42);

    // Generate vectors
    std::vector<std::vector<float>> vectors;
    vectors.reserve(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(generate_vector(dim, rng));
    }

    for (auto _ : state) {
        HNSWIndex<float, L2Metric<float>> index;
        for (size_t i = 0; i < num_vectors; ++i) {
            index.insert(i, std::span<const float>{vectors[i]});
        }
        benchmark::DoNotOptimize(index);
    }

    state.counters["vectors"] = num_vectors;
    state.counters["vectors/sec"] =
        benchmark::Counter(num_vectors, benchmark::Counter::kIsIterationInvariantRate);
}

// Build time scaling: 1K, 10K, 100K vectors
BENCHMARK(BM_HNSW_Build)->Args({1000, 384});
BENCHMARK(BM_HNSW_Build)->Args({10000, 384});
BENCHMARK(BM_HNSW_Build)->Args({100000, 384});

// ============================================================================
// Benchmark: HNSW Search Latency
// ============================================================================

static void BM_HNSW_Search(benchmark::State& state) {
    size_t corpus_size = state.range(0);
    size_t dim = state.range(1);
    size_t k = state.range(2);
    size_t ef_search = state.range(3);

    std::mt19937 rng(42);

    // Build index
    HNSWIndex<float, L2Metric<float>> index;
    std::vector<std::vector<float>> vectors;
    vectors.reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        index.insert(i, std::span<const float>{vectors[i]});
    }

    // Generate query
    auto query = generate_vector(dim, rng);

    // Benchmark search
    for (auto _ : state) {
        auto results = index.search(std::span<const float>{query}, k, ef_search);
        benchmark::DoNotOptimize(results);
    }

    state.counters["corpus"] = corpus_size;
    state.counters["k"] = k;
    state.counters["ef"] = ef_search;
    state.counters["QPS"] = benchmark::Counter(1, benchmark::Counter::kIsRate);
}

// Corpus size scaling (384d, k=10, ef=50)
BENCHMARK(BM_HNSW_Search)->Args({1000, 384, 10, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 50});
BENCHMARK(BM_HNSW_Search)->Args({100000, 384, 10, 50});

// ef_search scaling (10K corpus, 384d, k=10)
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 10});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 20});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 100});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 200});

// k-value scaling (10K corpus, 384d, ef=50)
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 1, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 5, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 50, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 100, 100});

// Dimension scaling (10K corpus, k=10, ef=50)
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 768, 10, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 1536, 10, 50});

// ============================================================================
// Benchmark: HNSW vs Brute-Force Comparison
// ============================================================================

static void BM_Brute_Force_Search(benchmark::State& state) {
    size_t corpus_size = state.range(0);
    size_t dim = state.range(1);
    size_t k = state.range(2);

    std::mt19937 rng(42);

    // Generate corpus
    std::vector<std::vector<float>> vectors;
    vectors.reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) {
        vectors.push_back(generate_vector(dim, rng));
    }

    // Generate query
    auto query = generate_vector(dim, rng);
    L2Metric<float> metric;

    // Benchmark brute-force search
    for (auto _ : state) {
        std::vector<std::pair<size_t, float>> results;
        results.reserve(corpus_size);
        for (size_t i = 0; i < corpus_size; ++i) {
            float dist = metric(std::span<const float>{query}, std::span<const float>{vectors[i]});
            results.emplace_back(i, dist);
        }
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });
        results.resize(k);
        benchmark::DoNotOptimize(results);
    }

    state.counters["corpus"] = corpus_size;
    state.counters["k"] = k;
    state.counters["QPS"] = benchmark::Counter(1, benchmark::Counter::kIsRate);
}

// Brute-force baseline (for speedup comparison)
BENCHMARK(BM_Brute_Force_Search)->Args({1000, 384, 10});
BENCHMARK(BM_Brute_Force_Search)->Args({10000, 384, 10});
BENCHMARK(BM_Brute_Force_Search)->Args({100000, 384, 10});

// ============================================================================
// Benchmark: Recall Quality vs ef_search
// ============================================================================

static void BM_HNSW_Recall_Quality(benchmark::State& state) {
    size_t corpus_size = 10000;
    size_t dim = 128;
    size_t k = 10;
    size_t ef_search = state.range(0);

    std::mt19937 rng(42);

    // Build index
    HNSWIndex<float, L2Metric<float>> index;
    std::vector<std::vector<float>> vectors;
    vectors.reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        index.insert(i, std::span<const float>{vectors[i]});
    }

    // Generate query
    auto query = generate_vector(dim, rng);

    // Compute ground truth (once)
    L2Metric<float> metric;
    std::vector<std::pair<size_t, float>> ground_truth;
    ground_truth.reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) {
        float dist = metric(std::span<const float>{query}, std::span<const float>{vectors[i]});
        ground_truth.emplace_back(i, dist);
    }
    std::sort(ground_truth.begin(), ground_truth.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    ground_truth.resize(k);

    std::unordered_set<size_t> gt_ids;
    for (const auto& [id, _] : ground_truth) {
        gt_ids.insert(id);
    }

    // Benchmark search and calculate recall
    size_t total_hits = 0;
    size_t iterations = 0;
    for (auto _ : state) {
        auto results = index.search(std::span<const float>{query}, k, ef_search);

        size_t hits = 0;
        for (const auto& [id, _] : results) {
            if (gt_ids.count(id))
                ++hits;
        }
        total_hits += hits;
        ++iterations;

        benchmark::DoNotOptimize(results);
    }

    float avg_recall = static_cast<float>(total_hits) / (iterations * k);
    state.counters["recall"] = avg_recall * 100.0f;
    state.counters["ef"] = ef_search;
}

// Recall vs ef_search trade-off
BENCHMARK(BM_HNSW_Recall_Quality)->Arg(10);
BENCHMARK(BM_HNSW_Recall_Quality)->Arg(20);
BENCHMARK(BM_HNSW_Recall_Quality)->Arg(50);
BENCHMARK(BM_HNSW_Recall_Quality)->Arg(100);
BENCHMARK(BM_HNSW_Recall_Quality)->Arg(200);
BENCHMARK(BM_HNSW_Recall_Quality)->Arg(500);

// ============================================================================
// Benchmark: Batch Search Throughput
// ============================================================================

static void BM_HNSW_Batch_Search(benchmark::State& state) {
    size_t corpus_size = state.range(0);
    size_t dim = state.range(1);
    size_t num_queries = state.range(2);
    size_t num_threads = state.range(3);
    size_t k = 10;
    size_t ef_search = 50;

    std::mt19937 rng(42);

    // Build index
    HNSWIndex<float, L2Metric<float>> index;
    std::vector<std::vector<float>> vectors;
    vectors.reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        index.insert(i, std::span<const float>{vectors[i]});
    }

    // Generate queries
    std::vector<std::vector<float>> query_data;
    std::vector<std::span<const float>> queries;
    query_data.reserve(num_queries);
    queries.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        query_data.push_back(generate_vector(dim, rng));
    }
    for (auto& q : query_data) {
        queries.push_back(std::span<const float>{q});
    }

    // Benchmark batch search
    for (auto _ : state) {
        auto results = index.search_batch(queries, k, ef_search, num_threads);
        benchmark::DoNotOptimize(results);
    }

    state.counters["corpus"] = corpus_size;
    state.counters["queries"] = num_queries;
    state.counters["threads"] = num_threads;
    state.counters["QPS"] =
        benchmark::Counter(num_queries, benchmark::Counter::kIsIterationInvariantRate);
}

// Batch search thread scaling (10K corpus, 384d, 100 queries)
BENCHMARK(BM_HNSW_Batch_Search)->Args({10000, 384, 100, 1});
BENCHMARK(BM_HNSW_Batch_Search)->Args({10000, 384, 100, 2});
BENCHMARK(BM_HNSW_Batch_Search)->Args({10000, 384, 100, 4});
BENCHMARK(BM_HNSW_Batch_Search)->Args({10000, 384, 100, 8});

// Batch size scaling (10K corpus, 384d, 4 threads)
BENCHMARK(BM_HNSW_Batch_Search)->Args({10000, 384, 10, 4});
BENCHMARK(BM_HNSW_Batch_Search)->Args({10000, 384, 100, 4});
BENCHMARK(BM_HNSW_Batch_Search)->Args({10000, 384, 1000, 4});

BENCHMARK_MAIN();
