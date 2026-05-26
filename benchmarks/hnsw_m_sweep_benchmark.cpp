// SPDX-License-Identifier: Apache-2.0 OR MIT
// Benchmark: HNSW M-parameter sweep for finding the optimal default
//
// Systematically measures recall@10, build time, and query latency across
// M values from 8 to 32 (stepping by 4) to identify the recall/cost knee.
//
// This benchmark directly validates the audit finding that M=16 may be too
// conservative for modern high-dimensional embedding vectors (384-1536d).
//
// Research basis: Elliott & Clark (2024), "The Impacts of Data, Ordering, and
// Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds"
// found that real embedding vectors need different parameters than SIFT1M.
//
// Usage:
//   ./hnsw_m_sweep_benchmark [--corpus=N] [--queries=N] [--dim=N]
//
// Output: CSV table suitable for plotting recall-vs-build-time Pareto frontier.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

#include <sqlite-vec-cpp/distances/cosine.hpp>
#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

// ============================================================================
// Configuration
// ============================================================================

struct SweepConfig {
    size_t corpus_size = 10000;
    size_t num_queries = 100;
    size_t dim = 768;
    size_t k = 10;
    size_t ef_construction = 200;
    // M values to sweep
    std::vector<size_t> m_values = {8, 12, 16, 20, 24, 28, 32};
    // ef_search values to test per M
    std::vector<size_t> ef_search_values = {50, 100, 200};
    uint32_t seed = 42;
    bool use_l2 = false; // false = cosine (default for embeddings)
};

static SweepConfig parseArgs(int argc, char* argv[]) {
    SweepConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--corpus=", 9) == 0)
            cfg.corpus_size = std::stoul(argv[i] + 9);
        else if (strncmp(argv[i], "--queries=", 10) == 0)
            cfg.num_queries = std::stoul(argv[i] + 10);
        else if (strncmp(argv[i], "--dim=", 6) == 0)
            cfg.dim = std::stoul(argv[i] + 6);
        else if (strncmp(argv[i], "--k=", 4) == 0)
            cfg.k = std::stoul(argv[i] + 4);
        else if (strncmp(argv[i], "--ef-construction=", 18) == 0)
            cfg.ef_construction = std::stoul(argv[i] + 18);
        else if (strncmp(argv[i], "--seed=", 7) == 0)
            cfg.seed = std::stoul(argv[i] + 7);
        else if (strcmp(argv[i], "--l2") == 0)
            cfg.use_l2 = true;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --corpus=N            Corpus size (default: 10000)\n");
            printf("  --queries=N           Number of queries (default: 100)\n");
            printf("  --dim=N               Vector dimension (default: 768)\n");
            printf("  --k=N                 Top-K results (default: 10)\n");
            printf("  --ef-construction=N   HNSW ef_construction (default: 200)\n");
            printf("  --seed=N              Random seed (default: 42)\n");
            printf("  --l2                  Use L2 distance (default: cosine)\n");
            exit(0);
        }
    }
    return cfg;
}

// ============================================================================
// Data Generation
// ============================================================================

static std::vector<float> generateVector(size_t dim, std::mt19937& rng, bool normalize = true) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    float norm = 0.0f;
    for (auto& v : vec) {
        v = dist(rng);
        norm += v * v;
    }
    if (normalize && norm > 0.0f) {
        norm = std::sqrt(norm);
        for (auto& v : vec)
            v /= norm;
    }
    return vec;
}

static std::vector<std::vector<float>> generateDataset(size_t count, size_t dim,
                                                       std::mt19937& rng) {
    std::vector<std::vector<float>> data;
    data.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        data.push_back(generateVector(dim, rng));
    }
    return data;
}

// ============================================================================
// Ground Truth (parallel brute-force)
// ============================================================================

static std::vector<std::unordered_set<size_t>>
computeGroundTruth(const std::vector<std::vector<float>>& queries,
                   const std::vector<std::vector<float>>& corpus, size_t k, bool use_l2) {
    const size_t numQueries = queries.size();
    const size_t numThreads = std::min<size_t>(std::thread::hardware_concurrency(), numQueries);

    std::vector<std::unordered_set<size_t>> gtSets(numQueries);
    std::vector<std::thread> threads;

    auto worker = [&](size_t start, size_t end) {
        CosineMetric<float> cosineMetric;
        L2Metric<float> l2Metric;
        for (size_t q = start; q < end; ++q) {
            std::vector<std::pair<size_t, float>> dists;
            dists.reserve(corpus.size());
            std::span<const float> qspan{queries[q]};
            for (size_t i = 0; i < corpus.size(); ++i) {
                std::span<const float> cspan{corpus[i]};
                float d = use_l2 ? l2Metric(qspan, cspan) : cosineMetric(qspan, cspan);
                dists.emplace_back(i, d);
            }
            std::partial_sort(dists.begin(), dists.begin() + static_cast<ptrdiff_t>(k), dists.end(),
                              [](auto& a, auto& b) { return a.second < b.second; });
            for (size_t i = 0; i < k; ++i) {
                gtSets[q].insert(dists[i].first);
            }
        }
    };

    size_t chunk = (numQueries + numThreads - 1) / numThreads;
    for (size_t t = 0; t < numThreads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, numQueries);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto& t : threads) {
        t.join();
    }
    return gtSets;
}

// ============================================================================
// Result
// ============================================================================

struct SweepResult {
    size_t M;
    size_t ef_search;
    double buildTimeMs;
    double searchLatencyUs;
    double qps;
    double recall;
    size_t totalEdges;
};

static void printCSVHeader() {
    printf("M,ef_search,Build(ms),Latency(us),QPS,Recall@10,TotalEdges,EdgesPerNode\n");
}

static void printCSV(const SweepResult& r, size_t numNodes) {
    printf("%zu,%zu,%.1f,%.1f,%.0f,%.4f,%zu,%.1f\n", r.M, r.ef_search, r.buildTimeMs,
           r.searchLatencyUs, r.qps, r.recall, r.totalEdges,
           numNodes > 0 ? static_cast<double>(r.totalEdges) / static_cast<double>(numNodes) : 0.0);
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char* argv[]) {
    SweepConfig cfg = parseArgs(argc, argv);

    printf("# HNSW M-parameter Sweep Benchmark\n");
    printf("# Corpus: %zu vectors, %zud, k=%zu\n", cfg.corpus_size, cfg.dim, cfg.k);
    printf("# ef_construction: %zu, metric: %s\n", cfg.ef_construction,
           cfg.use_l2 ? "L2" : "cosine");
    printf("# M values: ");
    for (size_t i = 0; i < cfg.m_values.size(); ++i) {
        printf("%zu%s", cfg.m_values[i], i + 1 < cfg.m_values.size() ? ", " : "");
    }
    printf("\n\n");

    // Generate deterministic data
    std::mt19937 rng(cfg.seed);
    auto corpus = generateDataset(cfg.corpus_size, cfg.dim, rng);
    auto queries = generateDataset(cfg.num_queries, cfg.dim, rng);

    printf("# Computing ground truth (brute-force k-NN, %zu queries)...\n", cfg.num_queries);
    fflush(stdout);
    auto gtStart = std::chrono::high_resolution_clock::now();
    auto gtSets = computeGroundTruth(queries, corpus, cfg.k, cfg.use_l2);
    auto gtEnd = std::chrono::high_resolution_clock::now();
    double gtMs = std::chrono::duration<double, std::milli>(gtEnd - gtStart).count();
    printf("# Ground truth computed in %.1f ms\n\n", gtMs);

    printCSVHeader();

    for (size_t M : cfg.m_values) {
        printf("# Building HNSW with M=%zu...\n", M);
        fflush(stdout);

        // Build index
        HNSWIndex<float, CosineMetric<float>>::Config hnswCfg;
        hnswCfg.M = M;
        hnswCfg.M_max = M * 2;
        hnswCfg.M_max_0 = M * 4;
        hnswCfg.ef_construction = cfg.ef_construction;
        hnswCfg.normalize_vectors = !cfg.use_l2; // normalize for cosine
        hnswCfg.random_seed = cfg.seed;

        HNSWIndex<float, CosineMetric<float>> idx(hnswCfg);

        auto buildStart = std::chrono::high_resolution_clock::now();
        idx.reserve(corpus.size());
        for (size_t i = 0; i < corpus.size(); ++i) {
            idx.insert_single_threaded(i, std::span<const float>{corpus[i]});
        }
        auto buildEnd = std::chrono::high_resolution_clock::now();
        double buildMs =
            std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();

        // Graph stats
        auto stats = idx.compute_graph_stats();
        size_t totalEdges = stats.total_edges;

        // Search at each ef value
        for (size_t ef : cfg.ef_search_values) {
            size_t totalHits = 0;
            auto searchStart = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < queries.size(); ++q) {
                auto res = idx.search_read_mostly(std::span<const float>{queries[q]}, cfg.k, ef);
                for (const auto& [id, _] : res) {
                    if (gtSets[q].count(id)) {
                        ++totalHits;
                    }
                }
            }
            auto searchEnd = std::chrono::high_resolution_clock::now();
            double totalUs =
                std::chrono::duration<double, std::micro>(searchEnd - searchStart).count();
            double latencyUs = totalUs / static_cast<double>(queries.size());
            double qps = 1e6 / latencyUs;
            double recall = 100.0 * static_cast<double>(totalHits) /
                            static_cast<double>(queries.size() * cfg.k);

            SweepResult result{M, ef, buildMs, latencyUs, qps, recall, totalEdges};
            printCSV(result, idx.size());
        }
        printf("\n");
        fflush(stdout);
    }

    printf("# Done.\n");
    return 0;
}
