// Benchmark: quantized HNSW search vs FP32 baseline
//
// Measures latency, QPS, and recall@K for:
//   - FP32 baseline (standard HNSW search)
//   - LVQ-8 two-stage search
//   - LVQ-4 two-stage search
//   - RaBitQ two-stage search
//
// Usage:
//   ./quantized_search_benchmark [--corpus N] [--dim D] [--queries Q]

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>
#include <sqlite-vec-cpp/index/hnsw_quantized.hpp>

using namespace sqlite_vec_cpp;
using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

// ========== Helpers ==========

std::vector<float> generate_vector(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& v : vec)
        v = dist(rng);
    return vec;
}

std::vector<std::pair<size_t, float>> brute_force_knn(std::span<const float> query,
                                                      const std::vector<std::vector<float>>& corpus,
                                                      size_t k) {
    L2Metric<float> metric;
    std::vector<std::pair<size_t, float>> distances;
    distances.reserve(corpus.size());
    for (size_t i = 0; i < corpus.size(); ++i) {
        distances.emplace_back(i, metric(query, std::span<const float>(corpus[i])));
    }
    std::partial_sort(distances.begin(), distances.begin() + std::min(k, distances.size()),
                      distances.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
    distances.resize(std::min(k, distances.size()));
    return distances;
}

float compute_recall(const std::vector<std::pair<size_t, float>>& predicted,
                     const std::vector<std::pair<size_t, float>>& ground_truth, size_t k) {
    size_t hits = 0;
    for (size_t i = 0; i < std::min(k, predicted.size()); ++i) {
        for (size_t j = 0; j < std::min(k, ground_truth.size()); ++j) {
            if (predicted[i].first == ground_truth[j].first) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<float>(hits) / static_cast<float>(std::min(k, ground_truth.size()));
}

struct BenchmarkResult {
    std::string name;
    double build_ms;
    double avg_latency_us;
    double qps;
    float recall;
    size_t memory_bytes;
};

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << "| Method | Build (ms) | Latency (us) | QPS | Recall@10 | Quant Memory |"
              << std::endl;
    std::cout << "|--------|------------|--------------|-----|-----------|--------------|"
              << std::endl;

    for (const auto& r : results) {
        std::cout << "| " << std::setw(16) << std::left << r.name << " | " << std::setw(10)
                  << std::right << std::fixed << std::setprecision(1) << r.build_ms << " | "
                  << std::setw(12) << std::setprecision(1) << r.avg_latency_us << " | "
                  << std::setw(5) << std::setprecision(0) << r.qps << " | " << std::setw(8)
                  << std::setprecision(1) << (r.recall * 100.0f) << "% | " << std::setw(10)
                  << r.memory_bytes << " B |" << std::endl;
    }
    std::cout << std::endl;
}

// ========== Main ==========

int main(int argc, char* argv[]) {
    size_t corpus_size = 10000;
    size_t dim = 384;
    size_t num_queries = 100;
    size_t k = 10;

    // Parse CLI args
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            corpus_size = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            dim = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--queries") == 0 && i + 1 < argc) {
            num_queries = std::stoul(argv[++i]);
        }
    }

    std::cout << "=== Quantized HNSW Search Benchmark ===" << std::endl;
    std::cout << "Corpus: " << corpus_size << " vectors, " << dim << "d, " << num_queries
              << " queries, k=" << k << std::endl;

    std::mt19937 rng(42);

    // Generate corpus
    std::cout << "Generating corpus..." << std::flush;
    std::vector<std::vector<float>> corpus;
    corpus.reserve(corpus_size);
    for (size_t i = 0; i < corpus_size; ++i) {
        corpus.push_back(generate_vector(dim, rng));
    }
    std::cout << " done." << std::endl;

    // Generate queries
    std::vector<std::vector<float>> queries;
    queries.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        queries.push_back(generate_vector(dim, rng));
    }

    // Compute ground truth
    std::cout << "Computing ground truth..." << std::flush;
    std::vector<std::vector<std::pair<size_t, float>>> ground_truth;
    ground_truth.reserve(num_queries);
    for (const auto& q : queries) {
        ground_truth.push_back(brute_force_knn(std::span<const float>(q), corpus, k));
    }
    std::cout << " done." << std::endl;

    // Build HNSW index
    std::cout << "Building HNSW index..." << std::flush;
    typename HNSWIndex<float, L2Metric<float>>::Config hnsw_config;
    hnsw_config.M = 16;
    hnsw_config.M_max = 32;
    hnsw_config.M_max_0 = 64;
    hnsw_config.ef_construction = 200;
    HNSWIndex<float, L2Metric<float>> index(hnsw_config);

    auto build_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert_single_threaded(i, std::span<const float>(corpus[i]));
    }
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << build_ms << " ms)."
              << std::endl;

    std::vector<BenchmarkResult> results;

    // Benchmark configurations
    struct BenchConfig {
        std::string name;
        QuantizationType qtype;
        size_t rerank_factor;
    };

    std::vector<BenchConfig> configs = {
        {"FP32 baseline", QuantizationType::None, 1},
        {"LVQ-8 (2x rerank)", QuantizationType::LVQ8, 2},
        {"LVQ-8 (3x rerank)", QuantizationType::LVQ8, 3},
        {"LVQ-4 (3x rerank)", QuantizationType::LVQ4, 3},
        {"RaBitQ (3x rerank)", QuantizationType::RaBitQ, 3},
        {"RaBitQ (5x rerank)", QuantizationType::RaBitQ, 5},
    };

    for (size_t ef_search : {50, 100, 200}) {
        std::cout << std::endl << "--- ef_search = " << ef_search << " ---" << std::endl;
        results.clear();

        for (const auto& cfg : configs) {
            HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
            qconfig.quantization = cfg.qtype;
            qconfig.rerank_factor = cfg.rerank_factor;

            HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);

            // Time quantization build
            auto qbuild_start = std::chrono::high_resolution_clock::now();
            qsearch.build_quantization();
            auto qbuild_end = std::chrono::high_resolution_clock::now();
            double qbuild_ms =
                std::chrono::duration<double, std::milli>(qbuild_end - qbuild_start).count();

            // Warm up
            for (size_t w = 0; w < 3; ++w) {
                qsearch.search(std::span<const float>(queries[0]), k, ef_search);
            }

            // Benchmark search
            float total_recall = 0.0f;
            auto search_start = std::chrono::high_resolution_clock::now();

            for (size_t q = 0; q < num_queries; ++q) {
                auto results_q = qsearch.search(std::span<const float>(queries[q]), k, ef_search);
                total_recall += compute_recall(results_q, ground_truth[q], k);
            }

            auto search_end = std::chrono::high_resolution_clock::now();
            double total_us =
                std::chrono::duration<double, std::micro>(search_end - search_start).count();

            BenchmarkResult br;
            br.name = cfg.name;
            br.build_ms = qbuild_ms;
            br.avg_latency_us = total_us / static_cast<double>(num_queries);
            br.qps = 1e6 / br.avg_latency_us;
            br.recall = total_recall / static_cast<float>(num_queries);
            br.memory_bytes = qsearch.quantized_memory_bytes();
            results.push_back(br);
        }

        print_results(results);
    }

    // Memory summary
    size_t fp32_bytes = corpus_size * dim * sizeof(float);
    std::cout << "FP32 vector memory: " << fp32_bytes << " bytes (" << (fp32_bytes / 1024 / 1024)
              << " MB)" << std::endl;

    return 0;
}
