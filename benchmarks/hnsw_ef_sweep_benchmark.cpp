// SPDX-License-Identifier: Apache-2.0 OR MIT
// Benchmark: HNSW ef_construction sweep for validating for_corpus() thresholds
//
// Tests ef_construction values from 50 to 400 across different corpus sizes
// to determine the minimum ef_construction needed for acceptable recall.
//
// This directly validates whether ef_construction=100 (current for_corpus()
// default for <100K corpora) is sufficient, or whether 200 should be the floor.
//
// Usage:
//   ./hnsw_ef_sweep_benchmark [--corpus=N] [--queries=N] [--dim=N] [--M=N]

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

struct EFSweepConfig {
    size_t corpus_size = 10000;
    size_t num_queries = 100;
    size_t dim = 768;
    size_t k = 10;
    size_t M = 16; // fixed M for this sweep
    // ef_construction values to sweep
    std::vector<size_t> ef_construction_values = {50, 100, 150, 200, 250, 300, 400, 500};
    // ef_search values to test per ef_construction
    std::vector<size_t> ef_search_values = {50, 100, 200, 400};
    uint32_t seed = 42;
    bool use_l2 = false;
};

static EFSweepConfig parseArgs(int argc, char* argv[]) {
    EFSweepConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--corpus=", 9) == 0)
            cfg.corpus_size = std::stoul(argv[i] + 9);
        else if (strncmp(argv[i], "--queries=", 10) == 0)
            cfg.num_queries = std::stoul(argv[i] + 10);
        else if (strncmp(argv[i], "--dim=", 6) == 0)
            cfg.dim = std::stoul(argv[i] + 6);
        else if (strncmp(argv[i], "--k=", 4) == 0)
            cfg.k = std::stoul(argv[i] + 4);
        else if (strncmp(argv[i], "--M=", 4) == 0)
            cfg.M = std::stoul(argv[i] + 4);
        else if (strncmp(argv[i], "--seed=", 7) == 0)
            cfg.seed = std::stoul(argv[i] + 7);
        else if (strcmp(argv[i], "--l2") == 0)
            cfg.use_l2 = true;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --corpus=N    Corpus size (default: 10000)\n");
            printf("  --queries=N   Number of queries (default: 100)\n");
            printf("  --dim=N       Vector dimension (default: 768)\n");
            printf("  --k=N         Top-K results (default: 10)\n");
            printf("  --M=N         HNSW M parameter (default: 16)\n");
            printf("  --seed=N      Random seed (default: 42)\n");
            printf("  --l2          Use L2 distance (default: cosine)\n");
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
// Ground Truth
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

struct EFSweepResult {
    size_t ef_construction;
    size_t ef_search;
    double buildTimeMs;
    double searchLatencyUs;
    double qps;
    double recall;
};

static void printHeader() {
    printf("%-16s | %-12s | %-12s | %-12s | %-12s | %-10s\n", "ef_construction", "ef_search",
           "Build(ms)", "Latency(us)", "QPS", "Recall@10");
    printf("-----------------+--------------+--------------+-"
           "-------------+--------------+-----------\n");
}

static void printRow(const EFSweepResult& r) {
    printf("%-16zu | %-12zu | %10.1f | %10.1f | %10.0f | %8.2f%%\n", r.ef_construction,
           r.ef_search, r.buildTimeMs, r.searchLatencyUs, r.qps, r.recall);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    EFSweepConfig cfg = parseArgs(argc, argv);

    printf("# HNSW ef_construction Sweep Benchmark\n");
    printf("# Corpus: %zu vectors, %zud, M=%zu, k=%zu\n", cfg.corpus_size, cfg.dim, cfg.M, cfg.k);
    printf("# Metric: %s\n\n", cfg.use_l2 ? "L2" : "cosine");

    // Generate data
    std::mt19937 rng(cfg.seed);
    auto corpus = generateDataset(cfg.corpus_size, cfg.dim, rng);
    auto queries = generateDataset(cfg.num_queries, cfg.dim, rng);

    printf("# Computing ground truth...\n");
    fflush(stdout);
    auto gtSets = computeGroundTruth(queries, corpus, cfg.k, cfg.use_l2);
    printf("# Ground truth computed.\n\n");

    printHeader();

    for (size_t efc : cfg.ef_construction_values) {
        printf("# Building HNSW with ef_construction=%zu...\n", efc);
        fflush(stdout);

        HNSWIndex<float, CosineMetric<float>>::Config hnswCfg;
        hnswCfg.M = cfg.M;
        hnswCfg.M_max = cfg.M * 2;
        hnswCfg.M_max_0 = cfg.M * 4;
        hnswCfg.ef_construction = efc;
        hnswCfg.normalize_vectors = !cfg.use_l2;
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

        for (size_t ef : cfg.ef_search_values) {
            size_t totalHits = 0;
            auto searchStart = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < queries.size(); ++q) {
                auto res = idx.search_read_mostly(std::span<const float>{queries[q]}, cfg.k, ef);
                for (const auto& [id, _] : res) {
                    if (gtSets[q].count(id))
                        ++totalHits;
                }
            }
            auto searchEnd = std::chrono::high_resolution_clock::now();
            double totalUs =
                std::chrono::duration<double, std::micro>(searchEnd - searchStart).count();
            double latencyUs = totalUs / static_cast<double>(queries.size());
            double qps = 1e6 / latencyUs;
            double recall = 100.0 * static_cast<double>(totalHits) /
                            static_cast<double>(queries.size() * cfg.k);

            printRow(EFSweepResult{efc, ef, buildMs, latencyUs, qps, recall});
        }
        printf("\n");
        fflush(stdout);
    }

    printf("# Done.\n");
    return 0;
}
