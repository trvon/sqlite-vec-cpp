// SPDX-License-Identifier: Apache-2.0 OR MIT
// Unit tests for batch distance calculations

#include <cassert>
#include <cmath>
#include <iostream>
#include <span>
#include <vector>
#include <sqlite-vec-cpp/distances/batch.hpp>
#include <sqlite-vec-cpp/distances/l2.hpp>

using namespace sqlite_vec_cpp::distances;

// Helper to compare floats with tolerance
bool approx_equal(float a, float b, float epsilon = 0.0001f) {
    return std::abs(a - b) < epsilon;
}

void test_batch_basic() {
    std::cout << "Testing batch distance - basic functionality..." << std::endl;

    // Query vector
    std::vector<float> query = {1.0f, 2.0f, 3.0f};

    // Database vectors
    std::vector<std::vector<float>> db_vecs = {
        {1.0f, 2.0f, 3.0f}, // distance = 0 (identical)
        {2.0f, 3.0f, 4.0f}, // distance = sqrt(3)
        {0.0f, 0.0f, 0.0f}, // distance = sqrt(14)
        {4.0f, 5.0f, 6.0f}  // distance = sqrt(27)
    };

    // Convert to spans
    std::vector<std::span<const float>> db_spans;
    for (const auto& vec : db_vecs) {
        db_spans.emplace_back(vec);
    }

    // Test batch distance
    auto distances =
        batch::batch_distance(std::span<const float>(query),
                              std::span<const std::span<const float>>(db_spans), L2Metric<float>{});

    assert(distances.size() == 4);
    assert(approx_equal(distances[0], 0.0f));
    assert(approx_equal(distances[1], std::sqrt(3.0f)));
    assert(approx_equal(distances[2], std::sqrt(14.0f)));
    assert(approx_equal(distances[3], std::sqrt(27.0f)));

    std::cout << "  ✓ Basic batch distance passed" << std::endl;
}

void test_batch_contiguous() {
    std::cout << "Testing contiguous batch distance..." << std::endl;

    std::vector<float> query = {1.0f, 2.0f, 3.0f};

    // Flatten database into contiguous array
    std::vector<float> db_flat = {
        1.0f, 2.0f, 3.0f, // vec 0
        2.0f, 3.0f, 4.0f, // vec 1
        0.0f, 0.0f, 0.0f, // vec 2
        4.0f, 5.0f, 6.0f  // vec 3
    };

    auto distances = batch::batch_distance_contiguous(std::span<const float>(query),
                                                      std::span<const float>(db_flat),
                                                      4, // num_vectors
                                                      3, // dim
                                                      L2Metric<float>{});

    assert(distances.size() == 4);
    assert(approx_equal(distances[0], 0.0f));
    assert(approx_equal(distances[1], std::sqrt(3.0f)));

    std::cout << "  ✓ Contiguous batch distance passed" << std::endl;
}

void test_batch_top_k() {
    std::cout << "Testing top-K neighbors..." << std::endl;

    std::vector<float> query = {1.0f, 2.0f, 3.0f};

    std::vector<std::vector<float>> db_vecs = {
        {1.0f, 2.0f, 3.0f}, // distance = 0
        {2.0f, 3.0f, 4.0f}, // distance = sqrt(3)
        {0.0f, 0.0f, 0.0f}, // distance = sqrt(14)
        {4.0f, 5.0f, 6.0f}  // distance = sqrt(27)
    };

    std::vector<std::span<const float>> db_spans;
    for (const auto& vec : db_vecs) {
        db_spans.emplace_back(vec);
    }

    auto top_2 = batch::batch_top_k(std::span<const float>(query),
                                    std::span<const std::span<const float>>(db_spans),
                                    2, // k=2
                                    L2Metric<float>{});

    assert(top_2.size() == 2);
    assert(top_2[0] == 0); // Closest: identical vector
    assert(top_2[1] == 1); // Second: distance sqrt(3)

    std::cout << "  ✓ Top-K neighbors passed" << std::endl;
}

void test_batch_filtered() {
    std::cout << "Testing filtered distance..." << std::endl;

    std::vector<float> query = {1.0f, 2.0f, 3.0f};

    std::vector<std::vector<float>> db_vecs = {
        {1.0f, 2.0f, 3.0f}, // distance = 0
        {2.0f, 3.0f, 4.0f}, // distance = sqrt(3) ~ 1.73
        {0.0f, 0.0f, 0.0f}, // distance = sqrt(14) ~ 3.74
        {4.0f, 5.0f, 6.0f}  // distance = sqrt(27) ~ 5.20
    };

    std::vector<std::span<const float>> db_spans;
    for (const auto& vec : db_vecs) {
        db_spans.emplace_back(vec);
    }

    auto filtered = batch::batch_distance_filtered(
        std::span<const float>(query), std::span<const std::span<const float>>(db_spans),
        2.0f, // threshold
        L2Metric<float>{});

    // Only vectors 0 and 1 should pass (dist < 2.0)
    assert(filtered.size() == 2);
    assert(filtered[0].first == 0); // Index 0
    assert(approx_equal(filtered[0].second, 0.0f));
    assert(filtered[1].first == 1); // Index 1
    assert(approx_equal(filtered[1].second, std::sqrt(3.0f)));

    std::cout << "  ✓ Filtered distance passed" << std::endl;
}

void test_batch_int8() {
    std::cout << "Testing batch distance with int8 quantization..." << std::endl;

    std::vector<int8_t> query = {10, 20, 30};

    std::vector<std::vector<int8_t>> db_vecs = {
        {10, 20, 30}, // identical
        {11, 21, 31}, // close
        {0, 0, 0}     // far
    };

    std::vector<std::span<const int8_t>> db_spans;
    for (const auto& vec : db_vecs) {
        db_spans.emplace_back(vec);
    }

    auto distances = batch::batch_distance(std::span<const int8_t>(query),
                                           std::span<const std::span<const int8_t>>(db_spans),
                                           L2Metric<int8_t>{});

    assert(distances.size() == 3);
    assert(approx_equal(distances[0], 0.0f));
    assert(approx_equal(distances[1], std::sqrt(3.0f)));

    std::cout << "  ✓ int8 quantized batch distance passed" << std::endl;
}

void test_batch_consistency() {
    std::cout << "Testing consistency with single distance..." << std::endl;

    std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> db_vec = {2.0f, 3.0f, 4.0f, 5.0f};

    // Single distance computation
    float single_dist = l2_distance(std::span<const float>(query), std::span<const float>(db_vec));

    // Batch distance computation
    std::vector<std::span<const float>> db_spans = {std::span<const float>(db_vec)};
    auto batch_dists =
        batch::batch_distance(std::span<const float>(query),
                              std::span<const std::span<const float>>(db_spans), L2Metric<float>{});

    assert(batch_dists.size() == 1);
    assert(approx_equal(batch_dists[0], single_dist));

    std::cout << "  ✓ Consistency check passed" << std::endl;
}

int main() {
    std::cout << "Running batch distance tests...\n" << std::endl;

    test_batch_basic();
    test_batch_contiguous();
    test_batch_top_k();
    test_batch_filtered();
    test_batch_int8();
    test_batch_consistency();

    std::cout << "\nAll batch distance tests passed!" << std::endl;
    return 0;
}
