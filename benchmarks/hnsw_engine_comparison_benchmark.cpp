// SPDX-License-Identifier: Apache-2.0 OR MIT
// Benchmark: YAMS HNSW (sqlite-vec-cpp) vs zvec (Alibaba Proxima) comparison
//
// Compares build time, search QPS, search latency, and recall@K for both
// implementations at 768-dimensional embeddings (Cohere-style workload).
//
// When zvec is available (YAMS_HAS_ZVEC=1), both engines are benchmarked
// side-by-side. Otherwise, only the YAMS HNSW baseline is measured, and
// reference zvec numbers from published benchmarks are printed for context.
//
// Usage:
//   ./hnsw_engine_comparison_benchmark [--corpus=N] [--queries=N] [--dim=N]
//
// See third_party/sqlite-vec-cpp/BENCHMARKS.md for methodology and results.

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

#if YAMS_HAS_ZVEC
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/index_param_builders.h>
#endif

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

// ============================================================================
// Configuration
// ============================================================================

struct BenchConfig {
    size_t corpus_size = 10000;
    size_t num_queries = 100;
    size_t dim = 768;
    size_t k = 10;
    size_t ef_construction = 200;
    uint32_t seed = 42;
    bool run_zvec = false;
};

static BenchConfig parseArgs(int argc, char* argv[]) {
    BenchConfig cfg;
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
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --corpus=N          Corpus size (default: 10000)\n");
            printf("  --queries=N         Number of queries (default: 100)\n");
            printf("  --dim=N             Vector dimension (default: 768)\n");
            printf("  --k=N               Top-K results (default: 10)\n");
            printf("  --ef-construction=N HNSW ef_construction (default: 200)\n");
            printf("  --seed=N            Random seed (default: 42)\n");
            exit(0);
        }
    }
#if YAMS_HAS_ZVEC
    cfg.run_zvec = true;
#endif
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
        for (auto& v : vec) {
            v /= norm;
        }
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
                   const std::vector<std::vector<float>>& corpus, size_t k) {
    const size_t numQueries = queries.size();
    const size_t numThreads = std::min<size_t>(std::thread::hardware_concurrency(), numQueries);

    std::vector<std::unordered_set<size_t>> gtSets(numQueries);
    std::vector<std::thread> threads;

    auto worker = [&](size_t start, size_t end) {
        CosineMetric<float> metric;
        for (size_t q = start; q < end; ++q) {
            std::vector<std::pair<size_t, float>> dists;
            dists.reserve(corpus.size());
            for (size_t i = 0; i < corpus.size(); ++i) {
                float d =
                    metric(std::span<const float>{queries[q]}, std::span<const float>{corpus[i]});
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
// Benchmark Result
// ============================================================================

struct EngineResult {
    const char* engine;
    int M;
    size_t efSearch;
    double buildTimeMs;
    double searchLatencyUs; // per query
    double qps;
    double recall;
    size_t memoryBytes; // approximate
};

static void printHeader() {
    printf("%-16s | %-4s | %-10s | %-12s | %-12s | %-12s | %-10s\n", "Engine", "M", "ef_search",
           "Build(ms)", "Latency(us)", "QPS", "Recall@K");
    printf("-----------------+------+------------+--------------+-"
           "-------------+--------------+-----------\n");
}

static void printResult(const EngineResult& r) {
    printf("%-16s | %-4d | %-10zu | %10.1f | %10.1f | %10.0f | %8.1f%%\n", r.engine, r.M,
           r.efSearch, r.buildTimeMs, r.searchLatencyUs, r.qps, r.recall);
}

// ============================================================================
// YAMS HNSW Benchmark (sqlite-vec-cpp)
// ============================================================================

static std::vector<EngineResult>
benchmarkYamsHNSW(const BenchConfig& cfg, const std::vector<std::vector<float>>& corpus,
                  const std::vector<std::vector<float>>& queries,
                  const std::vector<std::unordered_set<size_t>>& gtSets) {
    std::vector<EngineResult> results;

    for (int M : {16, 24, 32}) {
        // Build index
        HNSWIndex<float, CosineMetric<float>>::Config hnswCfg;
        hnswCfg.M = static_cast<size_t>(M);
        hnswCfg.M_max = static_cast<size_t>(M * 2);
        hnswCfg.M_max_0 = static_cast<size_t>(M * 4);
        hnswCfg.ef_construction = cfg.ef_construction;
        hnswCfg.normalize_vectors = true; // Pre-normalize for ~3x faster distance computation
        HNSWIndex<float, CosineMetric<float>> idx(hnswCfg);

        auto buildStart = std::chrono::high_resolution_clock::now();
        idx.reserve(corpus.size());
        for (size_t i = 0; i < corpus.size(); ++i) {
            idx.insert_single_threaded(i, std::span<const float>{corpus[i]});
        }
        auto buildEnd = std::chrono::high_resolution_clock::now();
        double buildMs = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();

        // Search at various ef values
        for (size_t ef : {50UL, 100UL, 200UL}) {
            size_t totalHits = 0;
            auto searchStart = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < queries.size(); ++q) {
                auto res = idx.search(std::span<const float>{queries[q]}, cfg.k, ef);
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
            double recall =
                100.0 * static_cast<double>(totalHits) /
                static_cast<double>(queries.size() * cfg.k);

            results.push_back(EngineResult{"yams-hnsw", M, ef, buildMs, latencyUs, qps, recall, 0});
        }
    }

    return results;
}

// ============================================================================
// Zvec Benchmark (when available)
// ============================================================================

#if YAMS_HAS_ZVEC
static std::vector<EngineResult>
benchmarkZvec(const BenchConfig& cfg, const std::vector<std::vector<float>>& corpus,
              const std::vector<std::vector<float>>& queries,
              const std::vector<std::unordered_set<size_t>>& gtSets) {
    using namespace zvec::core_interface;
    std::vector<EngineResult> results;

    for (int M : {16, 24, 32}) {
        // Create zvec HNSW index
        auto param = HNSWIndexParamBuilder()
                         .WithMetricType(MetricType::kInnerProduct)
                         .WithDataType(DataType::DT_FP32)
                         .WithDimension(static_cast<int>(cfg.dim))
                         .WithIsSparse(false)
                         .WithM(M)
                         .WithEFConstruction(static_cast<int>(cfg.ef_construction))
                         .Build();

        auto index = IndexFactory::CreateAndInitIndex(*param);
        if (!index) {
            fprintf(stderr, "zvec: Failed to create HNSW index (M=%d)\n", M);
            continue;
        }

        // Use a temporary file for the zvec index
        std::string indexPath = "/tmp/zvec_bench_" + std::to_string(M) + ".index";
        std::remove(indexPath.c_str());

        int ret = index->Open(indexPath,
                              StorageOptions{StorageOptions::StorageType::kMMAP, true});
        if (ret != 0) {
            fprintf(stderr, "zvec: Failed to open index at %s\n", indexPath.c_str());
            continue;
        }

        // Build index
        auto buildStart = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < corpus.size(); ++i) {
            VectorData vd;
            vd.vector = DenseVector{corpus[i].data()};
            index->Add(vd, static_cast<uint32_t>(i));
        }
        index->Train();
        auto buildEnd = std::chrono::high_resolution_clock::now();
        double buildMs = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();

        // Search at various ef_search values
        for (size_t ef : {50UL, 100UL, 200UL}) {
            auto queryParam = HNSWQueryParamBuilder()
                                  .with_topk(static_cast<int>(cfg.k))
                                  .with_ef_search(static_cast<int>(ef))
                                  .build();

            size_t totalHits = 0;
            auto searchStart = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < queries.size(); ++q) {
                SearchResult result;
                VectorData qd;
                qd.vector = DenseVector{queries[q].data()};
                index->Search(qd, queryParam, &result);

                for (const auto& doc : result.doc_list_) {
                    if (gtSets[q].count(static_cast<size_t>(doc.key()))) {
                        ++totalHits;
                    }
                }
            }
            auto searchEnd = std::chrono::high_resolution_clock::now();
            double totalUs =
                std::chrono::duration<double, std::micro>(searchEnd - searchStart).count();
            double latencyUs = totalUs / static_cast<double>(queries.size());
            double qps = 1e6 / latencyUs;
            double recall =
                100.0 * static_cast<double>(totalHits) /
                static_cast<double>(queries.size() * cfg.k);

            results.push_back(EngineResult{"zvec-proxima", M, ef, buildMs, latencyUs, qps, recall, 0});
        }

        // Clean up
        std::remove(indexPath.c_str());
    }

    return results;
}
#endif

// ============================================================================
// Published Reference Numbers
// ============================================================================

static void printPublishedZvecNumbers() {
    printf("\n");
    printf("=== Published zvec Benchmark Numbers (from zvec.org/en/docs/benchmarks/) ===\n");
    printf("Dataset: Cohere 1M (768d), Hardware: 16 vCPU / 64 GiB ECS\n");
    printf("  - zvec (int8 quant, M=15, ef_search=180): ~16K QPS at 95%%+ recall\n");
    printf("Dataset: Cohere 10M (768d), Hardware: 16 vCPU / 64 GiB ECS\n");
    printf("  - zvec (int8 quant, M=50, ef_search=118, refiner): ~8K QPS at 95%%+ recall\n");
    printf("\n");
    printf("Note: zvec uses int8 quantization + refiner by default in published benchmarks.\n");
    printf("      YAMS HNSW above uses FP32 (no quantization). For a fair comparison,\n");
    printf("      build zvec from source and rerun with YAMS_HAS_ZVEC=1.\n");
    printf("      See third_party/sqlite-vec-cpp/BENCHMARKS.md for methodology.\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    auto cfg = parseArgs(argc, argv);

    printf("================================================================\n");
    printf("HNSW Engine Comparison Benchmark\n");
    printf("================================================================\n");
    printf("Corpus: %zu vectors, Dim: %zu, k: %zu, Queries: %zu\n", cfg.corpus_size, cfg.dim,
           cfg.k, cfg.num_queries);
    printf("ef_construction: %zu, Seed: %u\n", cfg.ef_construction, cfg.seed);
    printf("Threads: %u\n", std::thread::hardware_concurrency());
#if YAMS_HAS_ZVEC
    printf("zvec: ENABLED (linked against libzvec_core)\n");
#else
    printf("zvec: NOT AVAILABLE (using published reference numbers)\n");
    printf("      To enable: build zvec, set -Dzvec-root=/path/to/zvec, then reconfigure\n");
#endif
    printf("================================================================\n\n");

    std::mt19937 rng(cfg.seed);

    // Generate dataset
    printf("Generating %zu-dim corpus (%zu vectors)...\n", cfg.dim, cfg.corpus_size);
    fflush(stdout);
    auto corpus = generateDataset(cfg.corpus_size, cfg.dim, rng);

    printf("Generating %zu queries...\n", cfg.num_queries);
    fflush(stdout);
    auto queries = generateDataset(cfg.num_queries, cfg.dim, rng);

    // Compute ground truth
    printf("Computing ground truth (parallel brute-force)...\n");
    fflush(stdout);
    auto gtStart = std::chrono::high_resolution_clock::now();
    auto gtSets = computeGroundTruth(queries, corpus, cfg.k);
    auto gtEnd = std::chrono::high_resolution_clock::now();
    printf("Ground truth: %.1fs\n\n",
           std::chrono::duration<double>(gtEnd - gtStart).count());
    fflush(stdout);

    // === YAMS HNSW ===
    printf("--- YAMS HNSW (sqlite-vec-cpp) ---\n");
    fflush(stdout);
    printHeader();
    auto yamsResults = benchmarkYamsHNSW(cfg, corpus, queries, gtSets);
    for (const auto& r : yamsResults) {
        printResult(r);
    }
    printf("\n");

    // === ef_construction sweep (M=24 fixed) ===
    printf("--- ef_construction sweep (M=24, normalized) ---\n");
    printf("%-16s | %-4s | %-6s | %-10s | %-12s | %-10s\n",
           "Engine", "M", "ef_c", "ef_search", "Build(ms)", "Recall@K");
    printf("-----------------+------+--------+------------+--------------+-----------\n");
    fflush(stdout);
    for (size_t ef_c : {50UL, 100UL, 200UL, 400UL}) {
        HNSWIndex<float, CosineMetric<float>>::Config hnswCfg;
        hnswCfg.M = 24;
        hnswCfg.M_max = 48;
        hnswCfg.M_max_0 = 96;
        hnswCfg.ef_construction = ef_c;
        hnswCfg.normalize_vectors = true;
        HNSWIndex<float, CosineMetric<float>> idx(hnswCfg);

        auto t0 = std::chrono::high_resolution_clock::now();
        idx.reserve(corpus.size());
        for (size_t i = 0; i < corpus.size(); ++i) {
            idx.insert_single_threaded(i, std::span<const float>{corpus[i]});
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Measure recall@10 at ef_search=100
        size_t totalHits = 0;
        for (size_t q = 0; q < queries.size(); ++q) {
            auto res = idx.search(std::span<const float>{queries[q]}, cfg.k, 100);
            for (const auto& [id, _] : res) {
                if (gtSets[q].count(id))
                    ++totalHits;
            }
        }
        double recall = 100.0 * static_cast<double>(totalHits)
                       / static_cast<double>(queries.size() * cfg.k);

        printf("%-16s | %-4d | %-6zu | %-10d | %10.1f | %8.1f%%\n",
               "yams-hnsw", 24, ef_c, 100, buildMs, recall);
        fflush(stdout);
    }
    printf("\n");

    // === Parallel build comparison (M=24, ef_c=200) ===
    printf("--- Parallel build (M=24, ef_c=200, normalized) ---\n");
    printf("%-24s | %-12s | %-10s\n", "Build method", "Build(ms)", "Recall@K");
    printf("-------------------------+--------------+-----------\n");
    fflush(stdout);
    {
        // Sequential (insert_single_threaded)
        HNSWIndex<float, CosineMetric<float>>::Config hnswCfg;
        hnswCfg.M = 24; hnswCfg.M_max = 48; hnswCfg.M_max_0 = 96;
        hnswCfg.ef_construction = 200; hnswCfg.normalize_vectors = true;

        // Prepare spans
        std::vector<std::span<const float>> spans;
        std::vector<size_t> ids;
        spans.reserve(corpus.size());
        ids.reserve(corpus.size());
        for (size_t i = 0; i < corpus.size(); ++i) {
            spans.emplace_back(corpus[i]);
            ids.push_back(i);
        }

        HNSWIndex<float, CosineMetric<float>> idx_seq(hnswCfg);
        auto t0 = std::chrono::high_resolution_clock::now();
        idx_seq.reserve(corpus.size());
        for (size_t i = 0; i < corpus.size(); ++i) {
            idx_seq.insert_single_threaded(i, std::span<const float>{corpus[i]});
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double seqMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        size_t seqHits = 0;
        for (size_t q = 0; q < queries.size(); ++q) {
            auto res = idx_seq.search(std::span<const float>{queries[q]}, cfg.k, 100);
            for (const auto& [id, _] : res)
                if (gtSets[q].count(id)) ++seqHits;
        }
        double seqRecall = 100.0 * static_cast<double>(seqHits)
                          / static_cast<double>(queries.size() * cfg.k);
        printf("%-24s | %10.1f | %8.1f%%\n", "sequential", seqMs, seqRecall);

        // Parallel (build_parallel with default threads)
        HNSWIndex<float, CosineMetric<float>> idx_par(hnswCfg);
        auto t2 = std::chrono::high_resolution_clock::now();
        idx_par.build_parallel(std::span{ids}, std::span{spans}, 0);
        auto t3 = std::chrono::high_resolution_clock::now();
        double parMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

        size_t parHits = 0;
        for (size_t q = 0; q < queries.size(); ++q) {
            auto res = idx_par.search(std::span<const float>{queries[q]}, cfg.k, 100);
            for (const auto& [id, _] : res)
                if (gtSets[q].count(id)) ++parHits;
        }
        double parRecall = 100.0 * static_cast<double>(parHits)
                          / static_cast<double>(queries.size() * cfg.k);
        printf("%-24s | %10.1f | %8.1f%%\n", "parallel (auto threads)", parMs, parRecall);

        double speedup = seqMs / parMs;
        printf("\n  Parallel speedup: %.1fx (%d threads available)\n",
               speedup, static_cast<int>(std::thread::hardware_concurrency()));
    }
    printf("\n");

    // === zvec ===
#if YAMS_HAS_ZVEC
    printf("--- zvec (Alibaba Proxima) ---\n");
    fflush(stdout);
    printHeader();
    auto zvecResults = benchmarkZvec(cfg, corpus, queries, gtSets);
    for (const auto& r : zvecResults) {
        printResult(r);
    }
    printf("\n");

    // === Side-by-side comparison ===
    printf("=== Side-by-Side Comparison (M=24, ef_search=100) ===\n");
    printf("%-16s | %-12s | %-12s | %-10s\n", "Engine", "Latency(us)", "QPS", "Recall@K");
    printf("-----------------+--------------+--------------+-----------\n");
    for (const auto& r : yamsResults) {
        if (r.M == 24 && r.efSearch == 100) {
            printf("%-16s | %10.1f | %10.0f | %8.1f%%\n", r.engine, r.searchLatencyUs, r.qps,
                   r.recall);
        }
    }
    for (const auto& r : zvecResults) {
        if (r.M == 24 && r.efSearch == 100) {
            printf("%-16s | %10.1f | %10.0f | %8.1f%%\n", r.engine, r.searchLatencyUs, r.qps,
                   r.recall);
        }
    }
    printf("\n");
#else
    printPublishedZvecNumbers();
#endif

    // Summary
    printf("=== YAMS HNSW Summary ===\n");
    for (const auto& r : yamsResults) {
        if (r.efSearch == 100) {
            printf("  M=%-2d: %.0f QPS, %.1f%% recall@%zu, build=%.0fms\n", r.M, r.qps, r.recall,
                   cfg.k, r.buildTimeMs);
        }
    }
    printf("\n");

    return 0;
}
