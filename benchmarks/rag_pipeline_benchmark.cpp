// SPDX-License-Identifier: Apache-2.0 OR MIT
// Benchmark: RAG Pipeline - Top-K search over document corpus
//
// Simulates a Retrieval-Augmented Generation (RAG) pipeline:
// - Query: single embedding vector (e.g., user question)
// - Corpus: N document embeddings
// - Task: Find K most similar documents
//
// Metrics:
// - Latency: p50, p95, p99 (via Google Benchmark statistics)
// - Throughput: queries/second

#include <algorithm>
#include <random>
#include <vector>
#include "../include/sqlite-vec-cpp/distances/batch.hpp"
#include "../include/sqlite-vec-cpp/distances/l2.hpp"
#include <benchmark/benchmark.h>

using namespace sqlite_vec_cpp::distances;

// ============================================================================
// Document Corpus Generator
// ============================================================================

/// Generates random normalized embeddings (simulating real document vectors)
template <typename T> class DocumentCorpus {
public:
    DocumentCorpus(size_t num_docs, size_t dim, uint32_t seed = 42) : dim_(dim), gen_(seed) {
        documents_.reserve(num_docs);
        for (size_t i = 0; i < num_docs; ++i) {
            documents_.push_back(generate_normalized_vector(dim));
        }

        // Create spans for batch operations
        doc_spans_.reserve(num_docs);
        for (const auto& doc : documents_) {
            doc_spans_.emplace_back(doc);
        }
    }

    std::span<const std::span<const T>> documents() const { return std::span{doc_spans_}; }

    std::vector<T> generate_query() { return generate_normalized_vector(dim_); }

    size_t size() const { return documents_.size(); }
    size_t dimension() const { return dim_; }

private:
    std::vector<T> generate_normalized_vector(size_t dim) {
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(-1.0, 1.0);
            std::vector<T> vec(dim);
            std::generate(vec.begin(), vec.end(), [&]() { return dist(gen_); });

            // L2 normalization (unit vector)
            T norm = 0;
            for (T val : vec) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            if (norm > 1e-6) {
                for (T& val : vec) {
                    val /= norm;
                }
            }
            return vec;
        } else {
            // int8 quantized: [-128, 127]
            std::uniform_int_distribution<int> dist(-128, 127);
            std::vector<T> vec(dim);
            std::generate(vec.begin(), vec.end(), [&]() { return static_cast<T>(dist(gen_)); });
            return vec;
        }
    }

    size_t dim_;
    std::mt19937 gen_;
    std::vector<std::vector<T>> documents_;
    std::vector<std::span<const T>> doc_spans_;
};

// ============================================================================
// RAG Pipeline: Top-K Search (Sequential Baseline)
// ============================================================================

static void BM_RAG_Sequential_1K_384d_K5(benchmark::State& state) {
    DocumentCorpus<float> corpus(1000, 384);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();

        // Sequential: compute all distances, then find top-K
        std::vector<float> distances;
        distances.reserve(corpus.size());
        for (const auto& doc : corpus.documents()) {
            distances.push_back(metric(std::span<const float>{query}, doc));
        }

        // Find top-5
        std::vector<size_t> indices(distances.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                          [&distances](size_t a, size_t b) { return distances[a] < distances[b]; });

        benchmark::DoNotOptimize(indices);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Sequential_1K_384d_K5)->Unit(benchmark::kMicrosecond);

// ============================================================================
// RAG Pipeline: Top-K Search (Batch Optimized)
// ============================================================================

static void BM_RAG_Batch_1K_384d_K5(benchmark::State& state) {
    DocumentCorpus<float> corpus(1000, 384);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();

        auto top_k =
            batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 5, metric);

        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_1K_384d_K5)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Corpus Size Scaling (1K, 10K, 100K documents)
// ============================================================================

static void BM_RAG_Batch_10K_384d_K5(benchmark::State& state) {
    DocumentCorpus<float> corpus(10000, 384);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();
        auto top_k =
            batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 5, metric);
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_10K_384d_K5)->Unit(benchmark::kMicrosecond);

static void BM_RAG_Batch_100K_384d_K5(benchmark::State& state) {
    DocumentCorpus<float> corpus(100000, 384);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();
        auto top_k =
            batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 5, metric);
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_100K_384d_K5)->Unit(benchmark::kMillisecond);

// ============================================================================
// K-value Scaling (K=1, 5, 10, 50)
// ============================================================================

static void BM_RAG_Batch_10K_384d_K1(benchmark::State& state) {
    DocumentCorpus<float> corpus(10000, 384);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();
        auto top_k =
            batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 1, metric);
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_10K_384d_K1)->Unit(benchmark::kMicrosecond);

static void BM_RAG_Batch_10K_384d_K10(benchmark::State& state) {
    DocumentCorpus<float> corpus(10000, 384);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();
        auto top_k =
            batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 10, metric);
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_10K_384d_K10)->Unit(benchmark::kMicrosecond);

static void BM_RAG_Batch_10K_384d_K50(benchmark::State& state) {
    DocumentCorpus<float> corpus(10000, 384);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();
        auto top_k =
            batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 50, metric);
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_10K_384d_K50)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Embedding Dimension Scaling (384d, 768d, 1536d)
// ============================================================================

static void BM_RAG_Batch_10K_768d_K5(benchmark::State& state) {
    DocumentCorpus<float> corpus(10000, 768);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();
        auto top_k =
            batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 5, metric);
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_10K_768d_K5)->Unit(benchmark::kMicrosecond);

static void BM_RAG_Batch_10K_1536d_K5(benchmark::State& state) {
    DocumentCorpus<float> corpus(10000, 1536);
    L2Metric<float> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();
        auto top_k =
            batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 5, metric);
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_10K_1536d_K5)->Unit(benchmark::kMicrosecond);

// ============================================================================
// int8 Quantized Embeddings (4x storage reduction)
// ============================================================================

static void BM_RAG_Batch_10K_384d_K5_int8(benchmark::State& state) {
    DocumentCorpus<int8_t> corpus(10000, 384);
    L2Metric<int8_t> metric;

    for (auto _ : state) {
        auto query = corpus.generate_query();
        auto top_k =
            batch::batch_top_k(std::span<const int8_t>{query}, corpus.documents(), 5, metric);
        benchmark::DoNotOptimize(top_k);
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_RAG_Batch_10K_384d_K5_int8)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Multi-Query Throughput (10 queries in rapid succession)
// ============================================================================

static void BM_RAG_Throughput_10K_384d_10Q(benchmark::State& state) {
    DocumentCorpus<float> corpus(10000, 384);
    L2Metric<float> metric;

    // Pre-generate 10 queries
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 10; ++i) {
        queries.push_back(corpus.generate_query());
    }

    for (auto _ : state) {
        for (const auto& query : queries) {
            auto top_k =
                batch::batch_top_k(std::span<const float>{query}, corpus.documents(), 5, metric);
            benchmark::DoNotOptimize(top_k);
        }
    }
    state.SetItemsProcessed(state.iterations() * corpus.size() * 10);
}
BENCHMARK(BM_RAG_Throughput_10K_384d_10Q)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
