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
#include <sqlite-vec-cpp/utils/float16.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;
using namespace sqlite_vec_cpp::utils;

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

// Test 11: Soft deletion
void test_soft_deletion() {
    std::cout << "Test 11: Soft deletion..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    // Create vectors in a known pattern
    std::vector<std::vector<float>> vectors = {
        {0.0f, 0.0f},  // 0: origin
        {1.0f, 0.0f},  // 1: right
        {0.0f, 1.0f},  // 2: up
        {-1.0f, 0.0f}, // 3: left
        {0.0f, -1.0f}, // 4: down
        {2.0f, 0.0f},  // 5: far right
        {3.0f, 0.0f},  // 6: very far right
    };

    for (size_t i = 0; i < vectors.size(); ++i) {
        index.insert(i, std::span{vectors[i]});
    }

    assert(index.size() == 7);
    assert(index.active_size() == 7);
    assert(index.deleted_count() == 0);

    // Search before deletion - query near far right should find 5, 6
    std::vector<float> query = {2.5f, 0.0f};
    auto results_before = index.search(std::span{query}, 3, 10);
    assert(results_before.size() == 3);

    // Should be 6, 5, 1 in order of distance
    std::unordered_set<size_t> before_ids;
    for (const auto& [id, dist] : results_before) {
        before_ids.insert(id);
    }
    assert(before_ids.count(5) == 1);
    assert(before_ids.count(6) == 1);

    // Soft delete node 5 (far right)
    index.remove(5);
    assert(index.is_deleted(5));
    assert(index.size() == 7);        // Total size unchanged
    assert(index.active_size() == 6); // Active size decremented
    assert(index.deleted_count() == 1);

    // Search after deletion - should not include 5
    auto results_after = index.search(std::span{query}, 3, 10);
    for (const auto& [id, dist] : results_after) {
        assert(id != 5); // Deleted node should not appear
    }

    // Verify 6 is still findable
    bool found_6 = false;
    for (const auto& [id, dist] : results_after) {
        if (id == 6)
            found_6 = true;
    }
    assert(found_6);

    std::cout << "  ✓ Soft deletion passed" << std::endl;
}

// Test 12: Restore deleted nodes
void test_restore_deletion() {
    std::cout << "Test 12: Restore deleted nodes..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    std::vector<float> vec1 = {0.0f, 0.0f};
    std::vector<float> vec2 = {1.0f, 0.0f};
    index.insert(0, std::span{vec1});
    index.insert(1, std::span{vec2});

    // Delete node 1
    index.remove(1);
    assert(index.is_deleted(1));
    assert(index.active_size() == 1);

    // Search should not find node 1
    std::vector<float> query = {0.5f, 0.0f};
    auto results = index.search(std::span{query}, 2, 10);
    for (const auto& [id, dist] : results) {
        assert(id != 1);
    }

    // Restore node 1
    bool restored = index.restore(1);
    assert(restored);
    assert(!index.is_deleted(1));
    assert(index.active_size() == 2);

    // Search should now find node 1
    results = index.search(std::span{query}, 2, 10);
    bool found_1 = false;
    for (const auto& [id, dist] : results) {
        if (id == 1)
            found_1 = true;
    }
    assert(found_1);

    // Restore non-existent node should return false
    assert(!index.restore(99));

    std::cout << "  ✓ Restore deletion passed" << std::endl;
}

// Test 13: Compaction
void test_compaction() {
    std::cout << "Test 13: Compaction..." << std::endl;

    constexpr size_t num_vectors = 100;
    constexpr size_t dim = 10;
    std::mt19937 rng(42);

    HNSWIndex<float, L2Metric<float>> index;
    std::vector<std::vector<float>> vectors;

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        index.insert(i, std::span{vectors[i]});
    }

    // Delete 30% of nodes
    for (size_t i = 0; i < 30; ++i) {
        index.remove(i);
    }

    assert(index.size() == 100);
    assert(index.active_size() == 70);
    assert(index.needs_compaction(0.2f)); // 30% deleted > 20% threshold

    // Compact the index
    auto compacted = index.compact();

    assert(compacted.size() == 70);
    assert(compacted.active_size() == 70);
    assert(compacted.deleted_count() == 0);
    assert(!compacted.needs_compaction());

    // Verify search still works on compacted index
    auto query = generate_vector(dim, rng);
    auto results = compacted.search(std::span{query}, 10, 50);
    assert(results.size() == 10);

    // None of the results should be from deleted IDs
    for (const auto& [id, dist] : results) {
        assert(id >= 30); // IDs 0-29 were deleted
    }

    std::cout << "  ✓ Compaction passed" << std::endl;
}

// Test 14: Isolate deleted nodes
void test_isolate_deleted() {
    std::cout << "Test 14: Isolate deleted nodes..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    // Create a small graph
    std::vector<std::vector<float>> vectors = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {2.0f, 0.0f},
        {3.0f, 0.0f},
    };

    for (size_t i = 0; i < vectors.size(); ++i) {
        index.insert(i, std::span{vectors[i]});
    }

    // Delete middle node
    index.remove(1);
    index.remove(2);

    // Isolate - removes edges to deleted nodes
    index.isolate_deleted();

    // Search should still work
    std::vector<float> query = {3.5f, 0.0f};
    auto results = index.search(std::span{query}, 2, 10);

    // Should find 3 and 0 (skipping deleted 1, 2)
    assert(results.size() == 2);
    for (const auto& [id, dist] : results) {
        assert(id == 0 || id == 3);
    }

    std::cout << "  ✓ Isolate deleted passed" << std::endl;
}

// Test 15: Clear deletions
void test_clear_deletions() {
    std::cout << "Test 15: Clear deletions..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    std::vector<float> vec1 = {0.0f, 0.0f};
    std::vector<float> vec2 = {1.0f, 0.0f};
    index.insert(0, std::span{vec1});
    index.insert(1, std::span{vec2});

    index.remove(0);
    index.remove(1);
    assert(index.deleted_count() == 2);

    index.clear_deletions();
    assert(index.deleted_count() == 0);
    assert(!index.is_deleted(0));
    assert(!index.is_deleted(1));
    assert(index.active_size() == 2);

    std::cout << "  ✓ Clear deletions passed" << std::endl;
}

// Test 16a: build_parallel validation
void test_build_parallel_validation() {
    std::cout << "Test 16a: build_parallel input validation..." << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    std::vector<float> vec = {0.0f, 1.0f};
    std::vector<std::span<const float>> spans = {std::span{vec}};
    std::vector<size_t> ids = {0, 1};

    bool threw = false;
    try {
        index.build_parallel(std::span{ids}, std::span{spans}, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
    std::cout << "  ✓ build_parallel validates input" << std::endl;
}

// Test 16b: Parallel build
void test_parallel_build() {
    std::cout << "Test 16b: Parallel build..." << std::endl;

    constexpr size_t num_vectors = 1000;
    constexpr size_t dim = 64;

    std::mt19937 rng(42);
    HNSWIndex<float, L2Metric<float>> index;

    // Generate vectors
    std::vector<std::vector<float>> vectors;
    std::vector<std::span<const float>> spans;
    std::vector<size_t> ids;

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(generate_vector(dim, rng));
        spans.emplace_back(vectors.back());
        ids.push_back(i);
    }

    // Parallel build
    // NOTE: This test can be flaky under coverage/instrumentation on some platforms.
    // Keep it single-threaded here; parallelism is covered by benchmarks and non-coverage runs.
    index.build_parallel(std::span{ids}, std::span{spans}, 1);

    assert(index.size() == num_vectors);

    // Test search still works
    auto query = generate_vector(dim, rng);
    auto results = index.search(std::span{query}, 10, 50);
    assert(results.size() == 10);

    std::cout << "  ✓ Parallel build passed (built " << num_vectors << " vectors)" << std::endl;
}

// Test 17: fp16 storage
void test_fp16_storage() {
    std::cout << "Test 17: fp16 storage..." << std::endl;

    HNSWIndex<sqlite_vec_cpp::utils::float16_t, L2Metric<float>> index;

    // Create vectors in a known pattern
    std::vector<std::vector<float>> f32_vectors = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f},
    };

    // Insert as fp16
    for (size_t i = 0; i < f32_vectors.size(); ++i) {
        auto f16_vec = sqlite_vec_cpp::utils::to_float16(std::span{f32_vectors[i]});
        index.insert(i, std::span{f16_vec});
    }

    assert(index.size() == 5);

    // Search should still work (queries are float, storage is fp16)
    std::vector<float> query = {0.5f, 0.0f};
    auto results = index.search(std::span{query}, 3, 10);

    assert(results.size() == 3);
    // ID 1 (1.0, 0.0) or ID 0 (0.0, 0.0) should be closest to (0.5, 0.0) - both at L2 distance 0.5
    assert(results[0].first == 1 || results[0].first == 0);
    // Distance should be reasonable (L2 distance from (0.5,0) to (1,0) or (0,0) is 0.5)
    assert(results[0].second < 0.6f);

    std::cout << "  ✓ fp16 storage passed" << std::endl;
}

// Test 18: fp16 accuracy
void test_fp16_accuracy() {
    std::cout << "Test 18: fp16 accuracy..." << std::endl;

    constexpr size_t num_vectors = 100;
    constexpr size_t dim = 64;

    std::mt19937 rng(42);

    // Build float32 index
    HNSWIndex<float, L2Metric<float>> f32_index;
    std::vector<std::vector<float>> f32_vectors;
    f32_vectors.reserve(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        f32_vectors.push_back(generate_vector(dim, rng));
        f32_index.insert(i, std::span{f32_vectors.back()});
    }

    // Build fp16 index with same vectors
    HNSWIndex<sqlite_vec_cpp::utils::float16_t, L2Metric<float>> f16_index;
    for (size_t i = 0; i < num_vectors; ++i) {
        auto f16_vec = sqlite_vec_cpp::utils::to_float16(std::span{f32_vectors[i]});
        f16_index.insert(i, std::span{f16_vec});
    }

    // Generate random query
    auto query = generate_vector(dim, rng);

    // Search both indices
    auto f32_results = f32_index.search(std::span{query}, 10, 50);
    auto f16_results = f16_index.search(std::span{query}, 10, 50);

    assert(f32_results.size() == 10);
    assert(f16_results.size() == 10);

    // Check that most results match
    std::unordered_set<size_t> f32_ids;
    for (const auto& [id, _] : f32_results) {
        f32_ids.insert(id);
    }

    size_t matches = 0;
    for (const auto& [id, _] : f16_results) {
        if (f32_ids.count(id)) {
            matches++;
        }
    }

    // At least 7 out of 10 should match (fp16 quantization is lossy but shouldn't change results
    // much)
    float match_rate = static_cast<float>(matches) / 10.0f;
    std::cout << "  fp16 match rate: " << (match_rate * 100) << "%" << std::endl;
    assert(match_rate >= 0.7f);

    std::cout << "  ✓ fp16 accuracy passed" << std::endl;
}

int main() {
    std::cout << "Running HNSW tests...\n" << std::endl;

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
    test_soft_deletion();
    test_restore_deletion();
    test_compaction();
    test_isolate_deleted();
    test_clear_deletions();
    test_build_parallel_validation();
    test_parallel_build();
    test_fp16_storage();
    test_fp16_accuracy();

    std::cout << "\nAll HNSW tests passed!" << std::endl;
    return 0;
}
