// SPDX-License-Identifier: Apache-2.0 OR MIT
// Benchmark for HNSW index performance

#include <algorithm>
#include <memory>
#include <random>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <benchmark/benchmark.h>
#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>
#include <sqlite-vec-cpp/utils/float16.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

// ============================================================================
// Helpers
// ============================================================================

std::vector<float> generate_vector(size_t dim, std::mt19937& rng);

namespace {

struct DatasetKey {
    size_t corpus = 0;
    size_t dim = 0;

    bool operator==(const DatasetKey& other) const {
        return corpus == other.corpus && dim == other.dim;
    }
};

struct DatasetKeyHash {
    size_t operator()(const DatasetKey& key) const {
        return (key.corpus * 1315423911u) ^ (key.dim + 0x9e3779b97f4a7c15ULL);
    }
};

struct QueryKey {
    size_t count = 0;
    size_t dim = 0;
    uint32_t seed = 0;

    bool operator==(const QueryKey& other) const {
        return count == other.count && dim == other.dim && seed == other.seed;
    }
};

struct QueryKeyHash {
    size_t operator()(const QueryKey& key) const {
        size_t h = key.count * 1315423911u;
        h ^= (key.dim + 0x9e3779b97f4a7c15ULL);
        h ^= (static_cast<size_t>(key.seed) << 1);
        return h;
    }
};

constexpr size_t kMaxCorpusDefault = 50000;

const std::vector<std::vector<float>>& get_vectors(size_t corpus_size, size_t dim) {
    static std::unordered_map<DatasetKey, std::shared_ptr<std::vector<std::vector<float>>>,
                              DatasetKeyHash>
        cache;

    DatasetKey key{corpus_size, dim};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return *it->second;
    }

    std::mt19937 rng(42);
    auto vectors = std::make_shared<std::vector<std::vector<float>>>();
    vectors->reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) {
        vectors->push_back(generate_vector(dim, rng));
    }

    cache.emplace(key, vectors);
    return *vectors;
}

const std::vector<std::vector<float>>& get_queries(size_t count, size_t dim, uint32_t seed) {
    static std::unordered_map<QueryKey, std::shared_ptr<std::vector<std::vector<float>>>,
                              QueryKeyHash>
        cache;

    QueryKey key{count, dim, seed};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return *it->second;
    }

    std::mt19937 rng(seed);
    auto queries = std::make_shared<std::vector<std::vector<float>>>();
    queries->reserve(count);
    for (size_t i = 0; i < count; ++i) {
        queries->push_back(generate_vector(dim, rng));
    }

    cache.emplace(key, queries);
    return *queries;
}

} // namespace

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
    size_t requested_vectors = num_vectors;
    if (num_vectors > kMaxCorpusDefault) {
        num_vectors = kMaxCorpusDefault;
    }

    const auto& vectors = get_vectors(num_vectors, dim);

    for (auto _ : state) {
        HNSWIndex<float, L2Metric<float>> index;
        for (size_t i = 0; i < num_vectors; ++i) {
            index.insert(i, std::span<const float>{vectors[i]});
        }
        benchmark::DoNotOptimize(index);
    }

    state.counters["vectors"] = num_vectors;
    if (requested_vectors != num_vectors) {
        state.counters["vectors_requested"] = requested_vectors;
    }
    state.counters["vectors/sec"] =
        benchmark::Counter(num_vectors, benchmark::Counter::kIsIterationInvariantRate);
}

// Build time scaling: 1K, 10K, 100K vectors
BENCHMARK(BM_HNSW_Build)->Args({1000, 384});
BENCHMARK(BM_HNSW_Build)->Args({10000, 384});
BENCHMARK(BM_HNSW_Build)->Args({50000, 384});

// ============================================================================
// Benchmark: HNSW Search Latency
// ============================================================================

static void BM_HNSW_Search(benchmark::State& state) {
    size_t corpus_size = state.range(0);
    size_t dim = state.range(1);
    size_t k = state.range(2);
    size_t ef_search = state.range(3);
    size_t requested_corpus = corpus_size;
    if (corpus_size > kMaxCorpusDefault) {
        corpus_size = kMaxCorpusDefault;
    }

    // Build index
    HNSWIndex<float, L2Metric<float>> index;
    const auto& vectors = get_vectors(corpus_size, dim);
    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert(i, std::span<const float>{vectors[i]});
    }

    // Generate query
    const auto& queries = get_queries(1, dim, 43);
    const auto& query = queries[0];

    // Benchmark search
    for (auto _ : state) {
        auto results = index.search(std::span<const float>{query}, k, ef_search);
        benchmark::DoNotOptimize(results);
    }

    state.counters["corpus"] = corpus_size;
    if (requested_corpus != corpus_size) {
        state.counters["corpus_requested"] = requested_corpus;
    }
    state.counters["k"] = k;
    state.counters["ef"] = ef_search;
    state.counters["QPS"] = benchmark::Counter(1, benchmark::Counter::kIsRate);
}

// Corpus size scaling (384d, k=10, ef=50)
BENCHMARK(BM_HNSW_Search)->Args({1000, 384, 10, 50});
BENCHMARK(BM_HNSW_Search)->Args({10000, 384, 10, 50});
BENCHMARK(BM_HNSW_Search)->Args({50000, 384, 10, 50});

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
    size_t requested_corpus = corpus_size;
    if (corpus_size > kMaxCorpusDefault) {
        corpus_size = kMaxCorpusDefault;
    }

    // Generate corpus
    const auto& vectors = get_vectors(corpus_size, dim);

    // Generate query
    const auto& queries = get_queries(1, dim, 43);
    const auto& query = queries[0];
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
    if (requested_corpus != corpus_size) {
        state.counters["corpus_requested"] = requested_corpus;
    }
    state.counters["k"] = k;
    state.counters["QPS"] = benchmark::Counter(1, benchmark::Counter::kIsRate);
}

// Brute-force baseline (for speedup comparison)
BENCHMARK(BM_Brute_Force_Search)->Args({1000, 384, 10});
BENCHMARK(BM_Brute_Force_Search)->Args({10000, 384, 10});
BENCHMARK(BM_Brute_Force_Search)->Args({50000, 384, 10});

// ============================================================================
// Benchmark: Recall Quality vs ef_search
// ============================================================================

static void BM_HNSW_Recall_Quality(benchmark::State& state) {
    size_t corpus_size = std::min<size_t>(10000, kMaxCorpusDefault);
    size_t dim = 128;
    size_t k = 10;
    size_t ef_search = state.range(0);

    // Build index
    HNSWIndex<float, L2Metric<float>> index;
    const auto& vectors = get_vectors(corpus_size, dim);
    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert(i, std::span<const float>{vectors[i]});
    }

    // Generate query
    const auto& queries = get_queries(1, dim, 43);
    const auto& query = queries[0];

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
// Benchmark: Recall/Precision Quality vs ef_search (Multi-query)
// ============================================================================

static void BM_HNSW_Recall_Quality_Multi(benchmark::State& state) {
    size_t corpus_size = std::min<size_t>(10000, kMaxCorpusDefault);
    size_t dim = 128;
    size_t k = 10;
    size_t ef_search = state.range(0);
    size_t num_queries = 100;

    // Build index
    HNSWIndex<float, L2Metric<float>> index;
    const auto& vectors = get_vectors(corpus_size, dim);
    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert(i, std::span<const float>{vectors[i]});
    }

    // Generate queries
    const auto& queries = get_queries(num_queries, dim, 44);

    // Precompute ground truth sets
    L2Metric<float> metric;
    std::vector<std::unordered_set<size_t>> gt_sets(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        std::vector<std::pair<size_t, float>> ground_truth;
        ground_truth.reserve(corpus_size);
        for (size_t i = 0; i < corpus_size; ++i) {
            float dist =
                metric(std::span<const float>{queries[q]}, std::span<const float>{vectors[i]});
            ground_truth.emplace_back(i, dist);
        }
        std::sort(ground_truth.begin(), ground_truth.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        ground_truth.resize(k);
        for (const auto& [id, _] : ground_truth) {
            gt_sets[q].insert(id);
        }
    }

    double recall = 0.0;
    for (auto _ : state) {
        state.PauseTiming();
        size_t total_hits = 0;
        for (size_t q = 0; q < num_queries; ++q) {
            auto results = index.search(std::span<const float>{queries[q]}, k, ef_search);
            for (const auto& [id, _] : results) {
                if (gt_sets[q].count(id)) {
                    ++total_hits;
                }
            }
        }
        recall = static_cast<double>(total_hits) / static_cast<double>(num_queries * k);
        state.ResumeTiming();
        benchmark::DoNotOptimize(recall);
    }

    state.counters["recall"] = recall * 100.0;
    state.counters["precision"] = recall * 100.0;
    state.counters["queries"] = static_cast<double>(num_queries);
    state.counters["ef"] = static_cast<double>(ef_search);
}

BENCHMARK(BM_HNSW_Recall_Quality_Multi)->Arg(10);
BENCHMARK(BM_HNSW_Recall_Quality_Multi)->Arg(50);
BENCHMARK(BM_HNSW_Recall_Quality_Multi)->Arg(100);
BENCHMARK(BM_HNSW_Recall_Quality_Multi)->Arg(200);

// ============================================================================
// Benchmark: fp16 drift vs fp32 (ranking + distance ratio)
// ============================================================================

static void BM_HNSW_FP16_Drift(benchmark::State& state) {
    size_t corpus_size = std::min<size_t>(10000, kMaxCorpusDefault);
    size_t dim = 128;
    size_t k = 10;
    size_t ef_search = 50;
    size_t num_queries = 100;

    // Build float32 index
    HNSWIndex<float, L2Metric<float>> f32_index;
    const auto& vectors = get_vectors(corpus_size, dim);
    for (size_t i = 0; i < corpus_size; ++i) {
        f32_index.insert(i, std::span<const float>{vectors[i]});
    }

    // Build fp16 index with the same vectors
    HNSWIndex<sqlite_vec_cpp::utils::float16_t, L2Metric<float>> f16_index;
    for (size_t i = 0; i < corpus_size; ++i) {
        auto f16_vec = sqlite_vec_cpp::utils::to_float16(std::span<const float>{vectors[i]});
        f16_index.insert(i, std::span<const sqlite_vec_cpp::utils::float16_t>{f16_vec});
    }

    // Generate queries
    const auto& queries = get_queries(num_queries, dim, 44);

    double overlap = 0.0;
    double ratio = 0.0;
    for (auto _ : state) {
        state.PauseTiming();
        size_t total_hits = 0;
        double ratio_sum = 0.0;
        size_t ratio_count = 0;

        for (size_t q = 0; q < num_queries; ++q) {
            auto f32_results = f32_index.search(std::span<const float>{queries[q]}, k, ef_search);
            auto f16_results = f16_index.search(std::span<const float>{queries[q]}, k, ef_search);

            std::unordered_map<size_t, float> f32_dist;
            f32_dist.reserve(f32_results.size());
            for (const auto& [id, dist] : f32_results) {
                f32_dist.emplace(id, dist);
            }

            for (const auto& [id, dist] : f16_results) {
                auto it = f32_dist.find(id);
                if (it != f32_dist.end()) {
                    ++total_hits;
                    if (it->second > 0.0f) {
                        ratio_sum += static_cast<double>(dist) /
                                     static_cast<double>(it->second);
                        ++ratio_count;
                    }
                }
            }
        }

        overlap = static_cast<double>(total_hits) / static_cast<double>(num_queries * k);
        ratio = ratio_count ? (ratio_sum / static_cast<double>(ratio_count)) : 0.0;
        state.ResumeTiming();
        benchmark::DoNotOptimize(overlap);
    }

    state.counters["overlap"] = overlap * 100.0;
    state.counters["mean_dist_ratio"] = ratio;
    state.counters["queries"] = static_cast<double>(num_queries);
}

BENCHMARK(BM_HNSW_FP16_Drift);

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
    size_t requested_corpus = corpus_size;
    if (corpus_size > kMaxCorpusDefault) {
        corpus_size = kMaxCorpusDefault;
    }

    // Build index
    HNSWIndex<float, L2Metric<float>> index;
    const auto& vectors = get_vectors(corpus_size, dim);
    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert(i, std::span<const float>{vectors[i]});
    }

    // Generate queries
    const auto& query_data = get_queries(num_queries, dim, 45);
    std::vector<std::span<const float>> queries;
    queries.reserve(num_queries);
    for (auto& q : query_data) {
        queries.push_back(std::span<const float>{q});
    }

    // Benchmark batch search
    for (auto _ : state) {
        auto results = index.search_batch(queries, k, ef_search, num_threads);
        benchmark::DoNotOptimize(results);
    }

    state.counters["corpus"] = corpus_size;
    if (requested_corpus != corpus_size) {
        state.counters["corpus_requested"] = requested_corpus;
    }
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
