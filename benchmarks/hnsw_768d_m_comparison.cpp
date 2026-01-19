// Benchmark: Compare M=24 vs M=32 for 768d embeddings
// Parallelized ground truth computation

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>
#include <sqlite-vec-cpp/distances/cosine.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

std::vector<float> gen_vec(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    float norm = 0.0f;
    for (auto& v : vec) { v = dist(rng); norm += v * v; }
    norm = std::sqrt(norm);
    for (auto& v : vec) { v /= norm; }
    return vec;
}

// Compute ground truth top-k for a single query (brute force)
std::unordered_set<size_t> compute_gt(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& corpus,
    size_t k, const CosineMetric<float>& metric
) {
    std::vector<std::pair<size_t, float>> dists;
    dists.reserve(corpus.size());
    for (size_t i = 0; i < corpus.size(); ++i) {
        float d = metric(std::span<const float>{query}, std::span<const float>{corpus[i]});
        dists.emplace_back(i, d);
    }
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end(),
                      [](auto& a, auto& b) { return a.second < b.second; });
    std::unordered_set<size_t> result;
    for (size_t i = 0; i < k; ++i) result.insert(dists[i].first);
    return result;
}

int main() {
    constexpr size_t corpus_size = 10000, dim = 768, k = 10, num_queries = 100;
    const size_t num_threads = std::thread::hardware_concurrency();
    std::mt19937 rng(42);

    printf("=================================================================\n");
    printf("768d Recall Benchmark: M=24 vs M=32\n");
    printf("Corpus: %zu, Dim: %zu, k: %zu, Queries: %zu, Threads: %zu\n",
           corpus_size, dim, k, num_queries, num_threads);
    printf("=================================================================\n\n");
    fflush(stdout);

    // Generate corpus
    printf("Generating corpus...\n"); fflush(stdout);
    std::vector<std::vector<float>> vecs;
    vecs.reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) vecs.push_back(gen_vec(dim, rng));

    // Generate queries
    printf("Generating queries...\n"); fflush(stdout);
    std::vector<std::vector<float>> qvecs;
    for (size_t i = 0; i < num_queries; ++i) qvecs.push_back(gen_vec(dim, rng));

    // Parallel ground truth computation
    printf("Computing ground truth (parallel, %zu threads)...\n", num_threads); fflush(stdout);
    auto gt_start = std::chrono::high_resolution_clock::now();

    std::vector<std::unordered_set<size_t>> gt_sets(num_queries);
    std::vector<std::thread> threads;

    auto worker = [&](size_t start, size_t end) {
        CosineMetric<float> metric;
        for (size_t q = start; q < end; ++q) {
            gt_sets[q] = compute_gt(qvecs[q], vecs, k, metric);
        }
    };

    size_t chunk = (num_queries + num_threads - 1) / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, num_queries);
        if (start < end) threads.emplace_back(worker, start, end);
    }
    for (auto& t : threads) t.join();

    auto gt_end = std::chrono::high_resolution_clock::now();
    printf("Ground truth: %.1fs\n\n",
           std::chrono::duration<double>(gt_end - gt_start).count());
    fflush(stdout);

    printf("%-6s | %-10s | %-12s | %-12s\n", "M", "ef_search", "Recall@10", "Latency(us)");
    printf("-------|------------|--------------|-------------\n");
    fflush(stdout);

    for (auto [M, M_max, M_max_0] : {std::tuple{24,48,96}, std::tuple{32,64,128}}) {
        // Build index
        HNSWIndex<float, CosineMetric<float>>::Config cfg;
        cfg.M = M; cfg.M_max = M_max; cfg.M_max_0 = M_max_0; cfg.ef_construction = 200;
        HNSWIndex<float, CosineMetric<float>> idx(cfg);

        auto build_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < corpus_size; ++i)
            idx.insert(i, std::span<const float>{vecs[i]});
        auto build_end = std::chrono::high_resolution_clock::now();
        double build_s = std::chrono::duration<double>(build_end - build_start).count();
        printf("M=%d built in %.1fs\n", M, build_s); fflush(stdout);

        for (size_t ef : {50UL, 100UL, 200UL}) {
            size_t hits = 0;
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < num_queries; ++q) {
                auto results = idx.search(std::span<const float>{qvecs[q]}, k, ef);
                for (const auto& [id, _] : results)
                    if (gt_sets[q].count(id)) ++hits;
            }
            auto end = std::chrono::high_resolution_clock::now();
            double lat_us = std::chrono::duration<double, std::micro>(end - start).count() / num_queries;
            double recall = 100.0 * hits / (num_queries * k);
            printf("%-6d | %-10zu | %10.1f%% | %10.1f\n", M, ef, recall, lat_us);
            fflush(stdout);
        }
        printf("-------|------------|--------------|-------------\n");
        fflush(stdout);
    }

    printf("\nConclusion: If M=32 recall > M=24 by >3%%, consider extending Config::for_corpus()\n");
    return 0;
}
