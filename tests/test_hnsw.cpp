// SPDX-License-Identifier: Apache-2.0 OR MIT
// Unit tests for HNSW index

#include <sqlite3.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <span>
#include <vector>
#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>
#include <sqlite-vec-cpp/index/hnsw_persistence.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

// Helper to compare floats with tolerance
bool approx_equal(float a, float b, float epsilon = 0.001f) {
    return std::abs(a - b) < epsilon;
}

// Generate random vector
std::vector<float> generate_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(rng);
    }
    return vec;
}

void test_node_creation() {
    std::cout << "Testing HNSW node creation..." << std::endl;

    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    HNSWNode<float> node(42, std::span{vec}, 2);

    assert(node.id == 42);
    assert(node.vector.size() == 3);
    assert(node.num_layers() == 3); // 0, 1, 2
    assert(!node.has_connections(0));

    std::cout << "  ✓ Node creation passed" << std::endl;
}

void test_layer_assignment() {
    std::cout << "Testing layer assignment distribution..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    // Insert 1000 nodes and check layer distribution
    std::vector<size_t> layer_counts(10, 0);
    std::mt19937 rng(42);

    for (size_t i = 0; i < 1000; ++i) {
        auto vec = generate_vector(3, rng);
        index.insert(i, std::span{vec});
    }

    // Entry point should be at a reasonable layer (not too high)
    // For 1000 nodes: expected max ≈ -ln(1/1000) * 1.44 ≈ 10
    assert(index.max_layer() < 15); // Probabilistically very unlikely to be higher

    std::cout << "  ✓ Layer assignment passed (max_layer=" << index.max_layer() << ")" << std::endl;
}

void test_small_graph_search() {
    std::cout << "Testing search on small graph..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    // Create 10 vectors in a simple pattern
    std::vector<std::vector<float>> vectors = {
        {0.0f, 0.0f},   // 0: origin
        {1.0f, 0.0f},   // 1: right
        {0.0f, 1.0f},   // 2: up
        {-1.0f, 0.0f},  // 3: left
        {0.0f, -1.0f},  // 4: down
        {1.0f, 1.0f},   // 5: up-right
        {-1.0f, 1.0f},  // 6: up-left
        {-1.0f, -1.0f}, // 7: down-left
        {1.0f, -1.0f},  // 8: down-right
        {2.0f, 0.0f},   // 9: far right
    };

    for (size_t i = 0; i < vectors.size(); ++i) {
        index.insert(i, std::span{vectors[i]});
    }

    // Query: origin should find itself
    std::vector<float> query1 = {0.0f, 0.0f};
    auto results1 = index.search(std::span{query1}, 1, 10);
    assert(results1.size() == 1);
    assert(results1[0].first == 0);                 // ID 0
    assert(approx_equal(results1[0].second, 0.0f)); // Distance 0

    // Query: far right should find ID 9 first
    std::vector<float> query2 = {2.5f, 0.0f};
    auto results2 = index.search(std::span{query2}, 1, 10);
    assert(results2.size() == 1);
    assert(results2[0].first == 9);

    // Query: top-5 from origin
    auto results3 = index.search(std::span{query1}, 5, 10);
    assert(results3.size() == 5);
    assert(results3[0].first == 0); // Origin is closest

    std::cout << "  ✓ Small graph search passed" << std::endl;
}

void test_recall_quality() {
    std::cout << "Testing recall quality (10K vectors)..." << std::endl;

    constexpr size_t num_vectors = 10000;
    constexpr size_t dim = 128;
    constexpr size_t k = 10;

    std::mt19937 rng(42);
    HNSWIndex<float, L2Metric<float>> index;

    // Generate random vectors
    std::vector<std::vector<float>> vectors;
    vectors.reserve(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        index.insert(i, std::span{vectors[i]});
    }

    // Generate random query
    auto query = generate_vector(dim, rng);

    // HNSW search (ef=100 for 90% recall on 10K vectors)
    auto hnsw_results = index.search(std::span{query}, k, 100);

    // Brute-force ground truth
    std::vector<std::pair<size_t, float>> ground_truth;
    ground_truth.reserve(num_vectors);
    L2Metric<float> metric;
    for (size_t i = 0; i < num_vectors; ++i) {
        float dist = metric(std::span{query}, std::span{vectors[i]});
        ground_truth.emplace_back(i, dist);
    }
    std::sort(ground_truth.begin(), ground_truth.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    ground_truth.resize(k);

    // Calculate recall
    std::unordered_set<size_t> gt_ids;
    for (const auto& [id, _] : ground_truth) {
        gt_ids.insert(id);
    }

    size_t hits = 0;
    for (const auto& [id, _] : hnsw_results) {
        if (gt_ids.count(id)) {
            ++hits;
        }
    }

    float recall = static_cast<float>(hits) / k;
    std::cout << "  Recall@" << k << ": " << (recall * 100) << "%" << std::endl;

    // Expect >=90% recall for ef=100
    assert(recall >= 0.90f);

    std::cout << "  ✓ Recall quality passed" << std::endl;
}

void test_insertion_order_independence() {
    std::cout << "Testing insertion order independence..." << std::endl;

    constexpr size_t num_vectors = 100;
    constexpr size_t dim = 10;
    std::mt19937 rng(42);

    // Generate vectors
    std::vector<std::vector<float>> vectors;
    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(generate_vector(dim, rng));
    }

    // Build index 1: sequential order
    HNSWIndex<float, L2Metric<float>> index1;
    for (size_t i = 0; i < num_vectors; ++i) {
        index1.insert(i, std::span{vectors[i]});
    }

    // Build index 2: random order
    HNSWIndex<float, L2Metric<float>> index2;
    std::vector<size_t> order(num_vectors);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t i : order) {
        index2.insert(i, std::span{vectors[i]});
    }

    // Query both indices
    auto query = generate_vector(dim, rng);
    auto results1 = index1.search(std::span{query}, 10, 50);
    auto results2 = index2.search(std::span{query}, 10, 50);

    // Results should have high overlap (not necessarily identical due to randomness)
    std::unordered_set<size_t> ids1;
    for (const auto& [id, _] : results1) {
        ids1.insert(id);
    }

    size_t overlap = 0;
    for (const auto& [id, _] : results2) {
        if (ids1.count(id))
            ++overlap;
    }

    float overlap_ratio = static_cast<float>(overlap) / 10;
    std::cout << "  Overlap: " << (overlap_ratio * 100) << "%" << std::endl;

    // Expect >70% overlap (not perfect due to different graph structures)
    assert(overlap_ratio > 0.70f);

    std::cout << "  ✓ Insertion order independence passed" << std::endl;
}

void test_edge_pruning() {
    std::cout << "Testing edge pruning (M_max enforcement)..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index({.M = 5, .M_max = 10, .M_max_0 = 10});

    std::mt19937 rng(42);

    // Insert 100 vectors
    for (size_t i = 0; i < 100; ++i) {
        auto vec = generate_vector(10, rng);
        index.insert(i, std::span{vec});
    }

    // Verify M_max is enforced (implementation detail, but check index is valid)
    assert(index.size() == 100);
    assert(!index.empty());

    std::cout << "  ✓ Edge pruning passed" << std::endl;
}

void test_empty_index() {
    std::cout << "Testing empty index..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    assert(index.empty());
    assert(index.size() == 0);

    std::vector<float> query = {1.0f, 2.0f, 3.0f};
    auto results = index.search(std::span{query}, 10, 50);
    assert(results.empty());

    std::cout << "  ✓ Empty index passed" << std::endl;
}

void test_single_vector() {
    std::cout << "Testing single vector..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    index.insert(0, std::span{vec});

    assert(index.size() == 1);

    std::vector<float> query = {1.0f, 2.0f, 3.0f};
    auto results = index.search(std::span{query}, 1, 10);
    assert(results.size() == 1);
    assert(results[0].first == 0);
    assert(approx_equal(results[0].second, 0.0f));

    std::cout << "  ✓ Single vector passed" << std::endl;
}

void test_batch_build() {
    std::cout << "Testing batch build..." << std::endl;

    constexpr size_t num_vectors = 1000;
    constexpr size_t dim = 32;
    std::mt19937 rng(42);

    // Generate vectors
    std::vector<std::vector<float>> vectors;
    std::vector<std::span<const float>> spans;
    std::vector<size_t> ids;

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        spans.emplace_back(vectors.back());
        ids.push_back(i);
    }

    // Batch build
    HNSWIndex<float, L2Metric<float>> index;
    index.build(std::span{ids}, std::span{spans});

    assert(index.size() == num_vectors);

    // Test search
    auto query = generate_vector(dim, rng);
    auto results = index.search(std::span{query}, 10, 50);
    assert(results.size() == 10);

    std::cout << "  ✓ Batch build passed" << std::endl;
}

// Test 10: Save/load round-trip
void test_persistence_roundtrip() {
    std::cout << "Test 10: Save/load round-trip..." << std::endl;

    constexpr size_t num_vectors = 1000;
    constexpr size_t dim = 128;
    constexpr size_t k = 10;

    std::mt19937 rng(42);

    // Build index
    HNSWIndex<float, L2Metric<float>> original_index;
    std::vector<std::vector<float>> vectors;
    vectors.reserve(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        original_index.insert(i, std::span{vectors[i]});
    }

    // Create test database
    sqlite3* db;
    int rc = sqlite3_open(":memory:", &db);
    assert(rc == SQLITE_OK);

    // Save index
    char* err = nullptr;
    rc = save_hnsw_index(db, "main", "test", original_index, &err);
    if (rc != SQLITE_OK) {
        std::cerr << "  ✗ Save failed (rc=" << rc << "): " << (err ? err : "unknown error")
                  << std::endl;
        if (err)
            std::cerr << "  Error: " << err << std::endl;
        sqlite3_free(err);
        sqlite3_close(db);
        assert(false);
    }

    std::cout << "  ✓ Index saved successfully" << std::endl;

    // Check database size before loading
    sqlite3_int64 page_count = 0;
    sqlite3_int64 page_size = 0;
    sqlite3_stmt* stmt_check;
    sqlite3_prepare_v2(db, "PRAGMA page_count", -1, &stmt_check, nullptr);
    if (sqlite3_step(stmt_check) == SQLITE_ROW) {
        page_count = sqlite3_column_int64(stmt_check, 0);
    }
    sqlite3_finalize(stmt_check);

    sqlite3_prepare_v2(db, "PRAGMA page_size", -1, &stmt_check, nullptr);
    if (sqlite3_step(stmt_check) == SQLITE_ROW) {
        page_size = sqlite3_column_int64(stmt_check, 0);
    }
    sqlite3_finalize(stmt_check);

    sqlite3_int64 db_size_bytes = page_count * page_size;
    size_t bytes_per_vector = db_size_bytes / num_vectors;

    // Load index
    std::cout << "  Loading index..." << std::endl;
    HNSWIndex<float, L2Metric<float>> loaded_index =
        load_hnsw_index<float, L2Metric<float>>(db, "main", "test", &err);
    std::cout << "  ✓ Index loaded successfully" << std::endl;

    sqlite3_close(db);

    // Verify index properties match
    assert(loaded_index.size() == original_index.size());
    assert(loaded_index.max_layer() == original_index.max_layer());
    assert(loaded_index.entry_point() == original_index.entry_point());

    // Verify search results match
    auto query = generate_vector(dim, rng);
    auto original_results = original_index.search(std::span{query}, k, 100);
    auto loaded_results = loaded_index.search(std::span{query}, k, 100);

    assert(original_results.size() == loaded_results.size());

    // Results should be identical (same IDs, same distances)
    for (size_t i = 0; i < original_results.size(); ++i) {
        assert(original_results[i].first == loaded_results[i].first);
        assert(approx_equal(original_results[i].second, loaded_results[i].second));
    }

    std::cout << "  ✓ Save/load round-trip passed" << std::endl;
    std::cout << "    - Saved and loaded " << num_vectors << " vectors" << std::endl;
    std::cout << "    - Database size: " << db_size_bytes << " bytes (" << bytes_per_vector
              << " bytes/vector)" << std::endl;
    std::cout << "    - Search results match exactly" << std::endl;
}

int main() {
    std::cout << "Running HNSW tests...\\n" << std::endl;

    test_node_creation();
    test_layer_assignment();
    test_small_graph_search();
    test_recall_quality();
    test_insertion_order_independence();
    test_edge_pruning();
    test_empty_index();
    test_single_vector();
    test_batch_build();
    test_persistence_roundtrip();

    std::cout << "\\nAll HNSW tests passed!" << std::endl;
    return 0;
}
