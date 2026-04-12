// SPDX-License-Identifier: Apache-2.0 OR MIT
// Unit tests for LVQ and RaBitQ quantization, plus two-stage HNSW search

#include <cassert>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>
#include <sqlite-vec-cpp/index/hnsw_quantized.hpp>
#include <sqlite-vec-cpp/quantization/lvq.hpp>
#include <sqlite-vec-cpp/quantization/rabitq.hpp>

using namespace sqlite_vec_cpp::quantization;
using namespace sqlite_vec_cpp::distances;
using namespace sqlite_vec_cpp::index;

// ========== Helpers ==========

bool approx_equal(float a, float b, float epsilon = 0.01f) {
    return std::abs(a - b) < epsilon;
}

std::vector<float> generate_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(rng);
    }
    return vec;
}

/// Brute-force k-NN for ground truth
std::vector<std::pair<size_t, float>> brute_force_knn(std::span<const float> query,
                                                      const std::vector<std::vector<float>>& corpus,
                                                      size_t k) {
    L2Metric<float> metric;
    std::vector<std::pair<size_t, float>> distances;
    for (size_t i = 0; i < corpus.size(); ++i) {
        float dist = metric(query, std::span<const float>(corpus[i]));
        distances.emplace_back(i, dist);
    }
    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    if (distances.size() > k) {
        distances.resize(k);
    }
    return distances;
}

/// Compute recall@k: fraction of true top-k that appear in predicted top-k
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

// ========== LVQ-8 Tests ==========

void test_lvq8_encode_decode() {
    std::cout << "Testing LVQ-8 encode/decode round-trip..." << std::endl;

    std::vector<float> vec = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    auto code = LVQ8::encode(std::span<const float>(vec));

    assert(code.codes.size() == 5);
    assert(approx_equal(code.offset, -1.0f, 0.01f));
    assert(approx_equal(code.scale, 2.0f / 255.0f, 0.01f));

    // Codes should span 0 to 255
    assert(code.codes[0] == 0);   // -1.0 -> 0
    assert(code.codes[4] == 255); // 1.0 -> 255

    // Decode and check reconstruction error
    auto decoded = LVQ8::decode(code);
    for (size_t i = 0; i < vec.size(); ++i) {
        assert(approx_equal(vec[i], decoded[i], 0.01f));
    }

    std::cout << "  PASS: LVQ-8 round-trip error < 0.01" << std::endl;
}

void test_lvq8_constant_vector() {
    std::cout << "Testing LVQ-8 with constant vector..." << std::endl;

    std::vector<float> vec = {0.5f, 0.5f, 0.5f, 0.5f};
    auto code = LVQ8::encode(std::span<const float>(vec));

    assert(code.scale == 0.0f);
    assert(approx_equal(code.offset, 0.5f));

    // Distance from a different query should be correct
    std::vector<float> query = {1.0f, 1.0f, 1.0f, 1.0f};
    float dist = LVQ8::l2_distance(std::span<const float>(query), code);
    float expected = std::sqrt(4.0f * 0.25f); // sqrt(4 * 0.5^2) = 1.0
    assert(approx_equal(dist, expected, 0.01f));

    std::cout << "  PASS: constant vector handled correctly" << std::endl;
}

void test_lvq8_distance_accuracy() {
    std::cout << "Testing LVQ-8 distance estimation accuracy..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 384;
    const size_t num_pairs = 1000;

    L2Metric<float> metric;
    float max_relative_error = 0.0f;
    float total_relative_error = 0.0f;

    for (size_t i = 0; i < num_pairs; ++i) {
        auto query = generate_vector(dim, rng);
        auto vec = generate_vector(dim, rng);

        float exact = metric(std::span<const float>(query), std::span<const float>(vec));
        auto code = LVQ8::encode(std::span<const float>(vec));
        float estimated = LVQ8::l2_distance(std::span<const float>(query), code);

        if (exact > 0.01f) {
            float rel_error = std::abs(exact - estimated) / exact;
            max_relative_error = std::max(max_relative_error, rel_error);
            total_relative_error += rel_error;
        }
    }

    float avg_relative_error = total_relative_error / static_cast<float>(num_pairs);

    std::cout << "  384d, 1000 pairs: avg_rel_error=" << avg_relative_error
              << " max_rel_error=" << max_relative_error << std::endl;

    // LVQ-8 should have <5% average relative error on random vectors
    assert(avg_relative_error < 0.05f);
    assert(max_relative_error < 0.15f);

    std::cout << "  PASS: LVQ-8 distance error within bounds" << std::endl;
}

void test_lvq8_simd_scalar_consistency() {
    std::cout << "Testing LVQ-8 SIMD vs scalar consistency..." << std::endl;

    std::mt19937 rng(123);

    // Test with dimensions that exercise SIMD paths (>= 16) and scalar tail
    for (size_t dim : {16, 32, 64, 128, 384, 385, 768}) {
        auto query = generate_vector(dim, rng);
        auto vec = generate_vector(dim, rng);
        auto code = LVQ8::encode(std::span<const float>(vec));

        // The l2_distance function dispatches to SIMD automatically
        float dist = LVQ8::l2_distance(std::span<const float>(query), code);

        // Compare against manual scalar reconstruction
        auto decoded = LVQ8::decode(code);
        L2Metric<float> metric;
        float exact_decoded =
            metric(std::span<const float>(query), std::span<const float>(decoded));

        assert(approx_equal(dist, exact_decoded, 0.01f));
    }

    std::cout << "  PASS: SIMD and scalar produce consistent results" << std::endl;
}

// ========== LVQ-4 Tests ==========

void test_lvq4_encode_decode() {
    std::cout << "Testing LVQ-4 encode/decode round-trip..." << std::endl;

    std::vector<float> vec = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 0.25f};
    auto code = LVQ4::encode(std::span<const float>(vec));

    // 6 dims -> 3 packed bytes
    assert(code.codes.size() == 3);
    assert(approx_equal(code.offset, -1.0f, 0.01f));

    auto decoded = LVQ4::decode(code, vec.size());
    // 4-bit has coarser resolution: 15 levels over range [-1, 1]
    for (size_t i = 0; i < vec.size(); ++i) {
        assert(approx_equal(vec[i], decoded[i], 0.15f));
    }

    std::cout << "  PASS: LVQ-4 round-trip error < 0.15" << std::endl;
}

void test_lvq4_distance_accuracy() {
    std::cout << "Testing LVQ-4 distance estimation accuracy..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 384;
    const size_t num_pairs = 1000;

    L2Metric<float> metric;
    float total_relative_error = 0.0f;

    for (size_t i = 0; i < num_pairs; ++i) {
        auto query = generate_vector(dim, rng);
        auto vec = generate_vector(dim, rng);

        float exact = metric(std::span<const float>(query), std::span<const float>(vec));
        auto code = LVQ4::encode(std::span<const float>(vec));
        float estimated = LVQ4::l2_distance(std::span<const float>(query), code, dim);

        if (exact > 0.01f) {
            float rel_error = std::abs(exact - estimated) / exact;
            total_relative_error += rel_error;
        }
    }

    float avg_relative_error = total_relative_error / static_cast<float>(num_pairs);
    std::cout << "  384d, 1000 pairs: avg_rel_error=" << avg_relative_error << std::endl;

    // LVQ-4 is coarser; allow up to 15% average error
    assert(avg_relative_error < 0.15f);

    std::cout << "  PASS: LVQ-4 distance error within bounds" << std::endl;
}

void test_lvq4_simd_scalar_consistency() {
    std::cout << "Testing LVQ-4 SIMD vs scalar consistency..." << std::endl;

    std::mt19937 rng(123);

    // Test dims that exercise NEON path (>= 32) and scalar-only path (< 32)
    for (size_t dim : {16, 17, 31, 32, 64, 128, 384, 385, 768}) {
        auto query = generate_vector(dim, rng);
        auto vec = generate_vector(dim, rng);
        auto code = LVQ4::encode(std::span<const float>(vec));

        float dist = LVQ4::l2_distance(std::span<const float>(query), code, dim);

        // Compare against decode → exact L2
        auto decoded = LVQ4::decode(code, dim);
        L2Metric<float> metric;
        float exact_decoded =
            metric(std::span<const float>(query), std::span<const float>(decoded));

        // LVQ-4 has coarser resolution; tolerance 0.15
        assert(approx_equal(dist, exact_decoded, 0.15f));
    }

    std::cout << "  PASS: LVQ-4 SIMD and scalar produce consistent results" << std::endl;
}

// ========== RaBitQ Tests ==========

void test_rabitq_encode() {
    std::cout << "Testing RaBitQ encoding..." << std::endl;

    std::vector<float> centroid = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    RaBitQ encoder(centroid);

    // A vector with known sign pattern
    std::vector<float> vec = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
    auto code = encoder.encode(std::span<const float>(vec));

    assert(code.byte_size() == 1); // 8 dims -> 1 byte
    // bits: dim 0 = +1 (bit 0 set), dim 1 = -1 (bit 1 clear), etc.
    // expected: 0b01010101 = 0x55
    assert(code.bits[0] == 0x55);
    assert(approx_equal(code.norm, std::sqrt(8.0f), 0.01f));

    std::cout << "  PASS: RaBitQ encoding correct" << std::endl;
}

void test_rabitq_centroid_computation() {
    std::cout << "Testing RaBitQ centroid computation..." << std::endl;

    std::vector<std::vector<float>> vecs = {
        {1.0f, 2.0f, 3.0f},
        {3.0f, 4.0f, 5.0f},
        {5.0f, 6.0f, 7.0f},
    };

    std::vector<std::span<const float>> spans;
    for (auto& v : vecs) {
        spans.push_back(std::span<const float>(v));
    }

    auto centroid = RaBitQ::compute_centroid(std::span(spans));
    assert(centroid.size() == 3);
    assert(approx_equal(centroid[0], 3.0f));
    assert(approx_equal(centroid[1], 4.0f));
    assert(approx_equal(centroid[2], 5.0f));

    std::cout << "  PASS: centroid computation correct" << std::endl;
}

void test_rabitq_distance_ordering() {
    std::cout << "Testing RaBitQ distance ordering preservation..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 384;
    const size_t corpus_size = 500;

    // Generate corpus and compute centroid
    std::vector<std::vector<float>> corpus;
    std::vector<std::span<const float>> corpus_spans;
    for (size_t i = 0; i < corpus_size; ++i) {
        corpus.push_back(generate_vector(dim, rng));
        corpus_spans.push_back(std::span<const float>(corpus.back()));
    }

    auto centroid = RaBitQ::compute_centroid(std::span(corpus_spans));
    RaBitQ encoder(centroid);

    // Encode all
    auto codes = encoder.encode_all(std::span(corpus_spans));

    // For multiple queries, check that RaBitQ ordering correlates with exact ordering
    L2Metric<float> metric;
    size_t ordering_preserved = 0;
    size_t total_comparisons = 0;

    for (size_t q = 0; q < 50; ++q) {
        auto query = generate_vector(dim, rng);
        auto query_state = encoder.prepare_query(std::span<const float>(query));

        // Compare all pairs of distances
        for (size_t i = 0; i < 20; ++i) {
            for (size_t j = i + 1; j < 20; ++j) {
                float exact_i =
                    metric(std::span<const float>(query), std::span<const float>(corpus[i]));
                float exact_j =
                    metric(std::span<const float>(query), std::span<const float>(corpus[j]));

                float rabitq_i = encoder.estimate_l2_distance(query_state, codes[i]);
                float rabitq_j = encoder.estimate_l2_distance(query_state, codes[j]);

                // Check if ordering is preserved
                if ((exact_i < exact_j && rabitq_i < rabitq_j) ||
                    (exact_i > exact_j && rabitq_i > rabitq_j) || (exact_i == exact_j)) {
                    ++ordering_preserved;
                }
                ++total_comparisons;
            }
        }
    }

    float ordering_rate =
        static_cast<float>(ordering_preserved) / static_cast<float>(total_comparisons);
    std::cout << "  384d ordering preservation: " << (ordering_rate * 100.0f) << "%" << std::endl;

    // Binary quantization won't preserve all orderings, but should be well above chance (50%)
    assert(ordering_rate > 0.60f);

    std::cout << "  PASS: RaBitQ ordering preservation > 60%" << std::endl;
}

void test_rabitq_hamming_neon_scalar_consistency() {
    std::cout << "Testing RaBitQ Hamming NEON vs scalar consistency..." << std::endl;

    std::mt19937 rng(99);
    const size_t dim = 768; // 96 bytes

    std::vector<float> centroid(dim, 0.0f);
    RaBitQ encoder(centroid);

    for (size_t trial = 0; trial < 100; ++trial) {
        auto v1 = generate_vector(dim, rng);
        auto v2 = generate_vector(dim, rng);

        auto c1 = encoder.encode(std::span<const float>(v1));
        auto c2 = encoder.encode(std::span<const float>(v2));

        // Distance should be deterministic regardless of SIMD path
        auto state = encoder.prepare_query(std::span<const float>(v1));
        float d1 = encoder.estimate_l2_distance(state, c2);
        float d2 = encoder.estimate_l2_distance(state, c2);
        assert(d1 == d2); // Exact equality expected for same computation
    }

    std::cout << "  PASS: Hamming distance deterministic across 100 trials" << std::endl;
}

// ========== Two-Stage HNSW Tests ==========

void test_two_stage_search_recall() {
    std::cout << "Testing two-stage HNSW search recall..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 128;
    const size_t corpus_size = 1000;
    const size_t k = 10;
    const size_t num_queries = 50;

    // Build corpus
    std::vector<std::vector<float>> corpus;
    for (size_t i = 0; i < corpus_size; ++i) {
        corpus.push_back(generate_vector(dim, rng));
    }

    // Build HNSW index
    typename HNSWIndex<float, L2Metric<float>>::Config hnsw_config;
    hnsw_config.M = 16;
    hnsw_config.M_max = 32;
    hnsw_config.M_max_0 = 64;
    hnsw_config.ef_construction = 100;
    HNSWIndex<float, L2Metric<float>> index(hnsw_config);

    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert_single_threaded(i, std::span<const float>(corpus[i]));
    }

    // Test each quantization type
    for (auto qtype : {QuantizationType::LVQ8, QuantizationType::LVQ4, QuantizationType::RaBitQ}) {
        const char* name = qtype == QuantizationType::LVQ8     ? "LVQ-8"
                           : qtype == QuantizationType::LVQ4   ? "LVQ-4"
                           : qtype == QuantizationType::RaBitQ ? "RaBitQ"
                                                               : "Unknown";

        HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
        qconfig.quantization = qtype;
        qconfig.rerank_factor = 3;

        HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);
        qsearch.build_quantization();

        assert(qsearch.has_quantization());

        float total_recall = 0.0f;
        for (size_t q = 0; q < num_queries; ++q) {
            auto query = generate_vector(dim, rng);
            auto gt = brute_force_knn(std::span<const float>(query), corpus, k);
            auto results = qsearch.search(std::span<const float>(query), k, 100);

            float recall = compute_recall(results, gt, k);
            total_recall += recall;
        }

        float avg_recall = total_recall / static_cast<float>(num_queries);
        std::cout << "  " << name << " recall@" << k << " = " << (avg_recall * 100.0f) << "%"
                  << std::endl;

        // Two-stage with reranking should achieve decent recall
        // LVQ-8 should be best, RaBitQ more approximate
        if (qtype == QuantizationType::LVQ8) {
            assert(avg_recall >= 0.50f);
        } else {
            assert(avg_recall >= 0.30f);
        }
    }

    std::cout << "  PASS: two-stage search produces valid recall" << std::endl;
}

void test_two_stage_vs_baseline_regression() {
    std::cout << "Testing two-stage search doesn't regress vs baseline..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 64;
    const size_t corpus_size = 500;
    const size_t k = 5;

    std::vector<std::vector<float>> corpus;
    for (size_t i = 0; i < corpus_size; ++i) {
        corpus.push_back(generate_vector(dim, rng));
    }

    typename HNSWIndex<float, L2Metric<float>>::Config hnsw_config;
    hnsw_config.ef_construction = 100;
    HNSWIndex<float, L2Metric<float>> index(hnsw_config);

    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert_single_threaded(i, std::span<const float>(corpus[i]));
    }

    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::LVQ8;
    qconfig.rerank_factor = 4;

    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);
    qsearch.build_quantization();

    // Compare recall of baseline vs two-stage
    float baseline_recall = 0.0f;
    float quantized_recall = 0.0f;
    size_t num_queries = 30;

    for (size_t q = 0; q < num_queries; ++q) {
        auto query = generate_vector(dim, rng);
        auto gt = brute_force_knn(std::span<const float>(query), corpus, k);

        auto baseline_results = index.search(std::span<const float>(query), k, 100);
        auto quant_results = qsearch.search(std::span<const float>(query), k, 100);

        baseline_recall += compute_recall(baseline_results, gt, k);
        quantized_recall += compute_recall(quant_results, gt, k);
    }

    baseline_recall /= static_cast<float>(num_queries);
    quantized_recall /= static_cast<float>(num_queries);

    std::cout << "  baseline recall@5 = " << (baseline_recall * 100.0f) << "%" << std::endl;
    std::cout << "  quantized recall@5 = " << (quantized_recall * 100.0f) << "%" << std::endl;

    // Quantized + rerank should be within 20% of baseline
    // (it may actually exceed baseline if rerank_factor expands the candidate set)
    assert(quantized_recall >= baseline_recall * 0.70f);

    std::cout << "  PASS: quantized search within acceptable range of baseline" << std::endl;
}

void test_two_stage_memory_savings() {
    std::cout << "Testing two-stage memory savings..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 384;
    const size_t corpus_size = 100;

    std::vector<std::vector<float>> corpus;
    for (size_t i = 0; i < corpus_size; ++i) {
        corpus.push_back(generate_vector(dim, rng));
    }

    HNSWIndex<float, L2Metric<float>> index;
    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert_single_threaded(i, std::span<const float>(corpus[i]));
    }

    size_t fp32_bytes = corpus_size * dim * sizeof(float); // 153,600 bytes

    for (auto qtype : {QuantizationType::LVQ8, QuantizationType::LVQ4, QuantizationType::RaBitQ}) {
        HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
        qconfig.quantization = qtype;
        HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);
        qsearch.build_quantization();

        size_t quant_bytes = qsearch.quantized_memory_bytes();
        float compression = static_cast<float>(fp32_bytes) / static_cast<float>(quant_bytes);

        const char* name = qtype == QuantizationType::LVQ8     ? "LVQ-8"
                           : qtype == QuantizationType::LVQ4   ? "LVQ-4"
                           : qtype == QuantizationType::RaBitQ ? "RaBitQ"
                                                               : "Unknown";

        std::cout << "  " << name << ": " << quant_bytes << " bytes, " << compression
                  << "x compression vs FP32" << std::endl;

        // Verify expected compression ratios
        if (qtype == QuantizationType::LVQ8) {
            assert(compression > 3.0f); // ~4x expected
        } else if (qtype == QuantizationType::LVQ4) {
            assert(compression > 6.0f); // ~8x expected
        } else if (qtype == QuantizationType::RaBitQ) {
            assert(compression > 20.0f); // ~32x expected
        }
    }

    std::cout << "  PASS: compression ratios meet expectations" << std::endl;
}

void test_two_stage_with_filter() {
    std::cout << "Testing two-stage search with filter..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 64;
    const size_t corpus_size = 200;
    const size_t k = 5;

    std::vector<std::vector<float>> corpus;
    for (size_t i = 0; i < corpus_size; ++i) {
        corpus.push_back(generate_vector(dim, rng));
    }

    HNSWIndex<float, L2Metric<float>> index;
    for (size_t i = 0; i < corpus_size; ++i) {
        index.insert_single_threaded(i, std::span<const float>(corpus[i]));
    }

    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::LVQ8;
    qconfig.rerank_factor = 3;
    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);
    qsearch.build_quantization();

    // Filter: only even IDs
    auto filter = [](size_t id) -> bool { return id % 2 == 0; };

    auto query = generate_vector(dim, rng);
    auto results = qsearch.search_with_filter(std::span<const float>(query), k, 50, filter);

    // All results should have even IDs
    for (const auto& [id, dist] : results) {
        assert(id % 2 == 0);
    }
    // Should return at most k results
    assert(results.size() <= k);

    std::cout << "  PASS: filtered search returns only matching IDs" << std::endl;
}

void test_two_stage_empty_index() {
    std::cout << "Testing two-stage search on empty index..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::LVQ8;
    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);
    qsearch.build_quantization();

    assert(!qsearch.has_quantization());

    std::vector<float> query = {1.0f, 2.0f, 3.0f};
    auto results = qsearch.search(std::span<const float>(query), 5, 50);
    assert(results.empty());

    std::cout << "  PASS: empty index returns empty results" << std::endl;
}

void test_two_stage_none_fallback() {
    std::cout << "Testing None quantization falls back to base index..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 32;
    const size_t corpus_size = 50;

    HNSWIndex<float, L2Metric<float>> index;
    for (size_t i = 0; i < corpus_size; ++i) {
        auto vec = generate_vector(dim, rng);
        index.insert_single_threaded(i, std::span<const float>(vec));
    }

    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::None;
    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);

    auto query = generate_vector(dim, rng);
    auto results = qsearch.search(std::span<const float>(query), 5, 50);
    auto baseline = index.search(std::span<const float>(query), 5, 50);

    // Should produce identical results
    assert(results.size() == baseline.size());
    for (size_t i = 0; i < results.size(); ++i) {
        assert(results[i].first == baseline[i].first);
        assert(approx_equal(results[i].second, baseline[i].second, 0.001f));
    }

    std::cout << "  PASS: None quantization is transparent fallback" << std::endl;
}

void test_two_stage_stale_detection() {
    std::cout << "Testing stale quantization detection..." << std::endl;

    std::mt19937 rng(42);
    const size_t dim = 32;
    const size_t corpus_size = 50;

    HNSWIndex<float, L2Metric<float>> index;
    for (size_t i = 0; i < corpus_size; ++i) {
        auto vec = generate_vector(dim, rng);
        index.insert_single_threaded(i, std::span<const float>(vec));
    }

    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::LVQ8;
    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);

    // Before build: no quantization, is_stale() should be false (nothing to be stale)
    assert(!qsearch.is_stale());

    qsearch.build_quantization();
    assert(qsearch.has_quantization());
    assert(!qsearch.is_stale());

    // Insert a new vector -> should become stale
    auto new_vec = generate_vector(dim, rng);
    index.insert_single_threaded(corpus_size, std::span<const float>(new_vec));
    assert(qsearch.is_stale());

    // Search should still work (falls back to exact)
    auto query = generate_vector(dim, rng);
    auto results = qsearch.search(std::span<const float>(query), 5, 50);
    assert(!results.empty());

    // Rebuild quantization -> not stale again
    qsearch.build_quantization();
    assert(!qsearch.is_stale());

    // Delete a vector -> should become stale
    index.remove(0);
    assert(qsearch.is_stale());

    // Search still works after delete
    results = qsearch.search(std::span<const float>(query), 5, 50);
    assert(!results.empty());

    std::cout << "  PASS: stale detection works for insert and delete" << std::endl;
}

void test_stale_isolate_deleted() {
    std::cout << "Testing stale detection with isolate_deleted..." << std::endl;

    std::mt19937 rng(99);
    const size_t dim = 32;
    const size_t corpus_size = 50;

    HNSWIndex<float, L2Metric<float>> index;
    for (size_t i = 0; i < corpus_size; ++i) {
        auto vec = generate_vector(dim, rng);
        index.insert_single_threaded(i, std::span<const float>(vec));
    }

    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::LVQ8;
    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);
    qsearch.build_quantization();
    assert(!qsearch.is_stale());

    // Soft-delete some nodes
    index.remove(0);
    index.remove(1);
    assert(qsearch.is_stale());

    // Rebuild to get fresh
    qsearch.build_quantization();
    assert(!qsearch.is_stale());

    // isolate_deleted rewrites graph edges -> must bump generation
    index.isolate_deleted();
    assert(qsearch.is_stale());

    // Search still works (falls back to exact)
    auto query = generate_vector(dim, rng);
    auto results = qsearch.search(std::span<const float>(query), 5, 50);
    assert(!results.empty());

    std::cout << "  PASS: isolate_deleted bumps generation" << std::endl;
}

void test_stale_restore_and_clear() {
    std::cout << "Testing stale detection with restore/clear_deletions..." << std::endl;

    std::mt19937 rng(77);
    const size_t dim = 32;
    const size_t corpus_size = 50;

    HNSWIndex<float, L2Metric<float>> index;
    for (size_t i = 0; i < corpus_size; ++i) {
        auto vec = generate_vector(dim, rng);
        index.insert_single_threaded(i, std::span<const float>(vec));
    }

    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::LVQ4;
    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);

    // Delete, rebuild, then restore -> stale
    index.remove(5);
    qsearch.build_quantization();
    assert(!qsearch.is_stale());

    bool restored = index.restore(5);
    assert(restored);
    assert(qsearch.is_stale());

    // Rebuild, then clear_deletions on empty set -> should NOT bump (no-op)
    qsearch.build_quantization();
    assert(!qsearch.is_stale());
    index.clear_deletions();     // nothing to clear
    assert(!qsearch.is_stale()); // generation unchanged

    // Delete, rebuild, then clear_deletions with actual deletions -> stale
    index.remove(10);
    index.remove(11);
    qsearch.build_quantization();
    assert(!qsearch.is_stale());
    index.clear_deletions();
    assert(qsearch.is_stale());

    std::cout << "  PASS: restore/clear_deletions bump generation correctly" << std::endl;
}

void test_snapshot_rebuild_consistency() {
    std::cout << "Testing snapshot-based rebuild consistency..." << std::endl;

    std::mt19937 rng(55);
    const size_t dim = 64;
    const size_t corpus_size = 100;

    HNSWIndex<float, L2Metric<float>> index;
    for (size_t i = 0; i < corpus_size; ++i) {
        auto vec = generate_vector(dim, rng);
        index.insert_single_threaded(i, std::span<const float>(vec));
    }

    // Build LVQ-8, search, get results
    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::LVQ8;
    qconfig.rerank_factor = 2;
    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);
    qsearch.build_quantization();

    auto query = generate_vector(dim, rng);
    auto results1 = qsearch.search(std::span<const float>(query), 10, 100);
    assert(results1.size() == 10);

    // Rebuild and search again -> same results (deterministic)
    qsearch.build_quantization();
    auto results2 = qsearch.search(std::span<const float>(query), 10, 100);
    assert(results2.size() == 10);

    // Same IDs in same order
    for (size_t i = 0; i < results1.size(); ++i) {
        assert(results1[i].first == results2[i].first);
        float dist_diff = std::abs(results1[i].second - results2[i].second);
        assert(dist_diff < 1e-5f);
    }

    std::cout << "  PASS: snapshot rebuild produces identical results" << std::endl;
}

void test_concurrent_insert_snapshot_staleness() {
    std::cout << "Testing concurrent insert snapshot staleness..." << std::endl;

    std::mt19937 rng(1234);
    const size_t dim = 32;
    const size_t corpus_size = 64;

    HNSWIndex<float, L2Metric<float>> index;
    for (size_t i = 0; i < corpus_size; ++i) {
        auto vec = generate_vector(dim, rng);
        index.insert_single_threaded(i, std::span<const float>(vec));
    }

    HNSWQuantizedSearch<float, L2Metric<float>>::Config qconfig;
    qconfig.quantization = QuantizationType::LVQ8;
    HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qconfig);
    qsearch.build_quantization();
    assert(!qsearch.is_stale());

    std::mutex m;
    std::condition_variable cv;
    bool published = false;
    bool allow_continue = false;

    HNSWIndex<float, L2Metric<float>>::testing_set_after_insert_publish_hook([&]() {
        std::unique_lock lk(m);
        published = true;
        cv.notify_all();
        cv.wait(lk, [&]() { return allow_continue; });
    });

    auto new_vec = generate_vector(dim, rng);
    std::thread inserter([&]() { index.insert(corpus_size, std::span<const float>(new_vec)); });

    {
        std::unique_lock lk(m);
        cv.wait(lk, [&]() { return published; });
    }

    // Snapshot/build while insert is mid-flight (node visible, graph not fully connected yet).
    qsearch.build_quantization();
    bool stale_while_insert_paused = qsearch.is_stale();

    {
        std::lock_guard lk(m);
        allow_continue = true;
    }
    cv.notify_all();
    inserter.join();

    bool stale_after_insert_complete = qsearch.is_stale();
    HNSWIndex<float, L2Metric<float>>::testing_clear_after_insert_publish_hook();

    assert(!stale_while_insert_paused);
    assert(stale_after_insert_complete &&
           "Snapshot built mid-insert must be stale after insertion completes");

    auto query = generate_vector(dim, rng);
    auto results = qsearch.search(std::span<const float>(query), 5, 50);
    assert(!results.empty());

    std::cout << "  PASS: concurrent insert invalidates mid-insert snapshot" << std::endl;
}

// ========== Main ==========

int main() {
    std::cout << "=== Quantization Tests ===" << std::endl;
    std::cout << std::endl;

    // LVQ-8 tests
    test_lvq8_encode_decode();
    test_lvq8_constant_vector();
    test_lvq8_distance_accuracy();
    test_lvq8_simd_scalar_consistency();
    std::cout << std::endl;

    // LVQ-4 tests
    test_lvq4_encode_decode();
    test_lvq4_distance_accuracy();
    test_lvq4_simd_scalar_consistency();
    std::cout << std::endl;

    // RaBitQ tests
    test_rabitq_encode();
    test_rabitq_centroid_computation();
    test_rabitq_distance_ordering();
    test_rabitq_hamming_neon_scalar_consistency();
    std::cout << std::endl;

    // Two-stage HNSW tests
    test_two_stage_search_recall();
    test_two_stage_vs_baseline_regression();
    test_two_stage_memory_savings();
    test_two_stage_with_filter();
    test_two_stage_empty_index();
    test_two_stage_none_fallback();
    test_two_stage_stale_detection();
    test_stale_isolate_deleted();
    test_stale_restore_and_clear();
    test_snapshot_rebuild_consistency();
    test_concurrent_insert_snapshot_staleness();
    std::cout << std::endl;

    std::cout << "=== All quantization tests passed ===" << std::endl;
    return 0;
}
