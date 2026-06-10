// SPDX-License-Identifier: Apache-2.0 OR MIT
// Deterministic fuzz tests for HNSW persistence deserialization.
// Property under test: corrupt shadow-table blobs must fail cleanly
// (throw std::runtime_error) and never crash, OOM, or return invalid nodes.

#include <sqlite3.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>
#include <sqlite-vec-cpp/index/hnsw_persistence.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

namespace {

std::vector<float> generate_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(rng);
    }
    return vec;
}

void patch_u64(std::vector<uint8_t>& blob, size_t offset, uint64_t value) {
    assert(offset + 8 <= blob.size());
    for (int i = 0; i < 8; ++i) {
        blob[offset + i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
    }
}

template <typename Fn> bool throws_runtime_error(Fn&& fn) {
    try {
        fn();
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

constexpr size_t kDim = 16;

std::vector<uint8_t> make_valid_node_blob() {
    std::mt19937 rng(7);
    auto vec = generate_vector(kDim, rng);
    HNSWNode<float> node(42, std::span<const float>(vec), 2);
    node.edges[0] = {1, 2, 3};
    node.edges[1] = {4, 5};
    node.edges[2] = {6};
    return serialize_hnsw_node(node);
}

void test_node_truncation_sweep() {
    std::cout << "Fuzz 1: node blob truncated at every offset..." << std::endl;

    auto blob = make_valid_node_blob();
    auto full = deserialize_hnsw_node<float>(blob.data(), blob.size());
    assert(full.id == 42);
    assert(full.vector.size() == kDim);
    assert(full.edges.size() == 3);

    size_t threw = 0;
    for (size_t len = 0; len < blob.size(); ++len) {
        try {
            auto node = deserialize_hnsw_node<float>(blob.data(), len);
            assert(node.vector.size() <= kDim);
            assert(node.edges.size() <= kMaxHnswLayers);
        } catch (const std::runtime_error&) {
            ++threw;
        }
    }
    assert(threw > 0);
    std::cout << "  ✓ " << blob.size() << " truncations survived (" << threw << " threw)"
              << std::endl;
}

void test_node_byte_flips() {
    std::cout << "Fuzz 2: single byte flips across node blob..." << std::endl;

    auto blob = make_valid_node_blob();
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    for (size_t pos = 0; pos < blob.size(); ++pos) {
        for (int round = 0; round < 4; ++round) {
            auto mutated = blob;
            mutated[pos] = static_cast<uint8_t>(byte_dist(rng));
            try {
                auto node = deserialize_hnsw_node<float>(mutated.data(), mutated.size());
                assert(node.edges.size() <= kMaxHnswLayers);
                assert(node.vector.size() * sizeof(float) <= mutated.size());
            } catch (const std::runtime_error&) {
            }
        }
    }
    std::cout << "  ✓ " << blob.size() * 4 << " byte flips survived" << std::endl;
}

void test_node_hostile_splices() {
    std::cout << "Fuzz 3: hostile num_layers / num_neighbors / dim splices..." << std::endl;

    auto blob = make_valid_node_blob();
    const size_t dim_offset = 8;
    const size_t num_layers_offset = 8 + 8 + kDim * sizeof(float);
    const size_t first_neighbor_count_offset = num_layers_offset + 8;

    for (uint64_t hostile :
         {uint64_t{kMaxHnswLayers + 1}, uint64_t{1} << 32, ~uint64_t{0}, ~uint64_t{0} / 8}) {
        auto mutated = blob;
        patch_u64(mutated, num_layers_offset, hostile);
        assert(throws_runtime_error(
            [&] { deserialize_hnsw_node<float>(mutated.data(), mutated.size()); }));
    }

    for (uint64_t hostile : {uint64_t{1} << 32, ~uint64_t{0}, ~uint64_t{0} / 8}) {
        auto mutated = blob;
        patch_u64(mutated, first_neighbor_count_offset, hostile);
        assert(throws_runtime_error(
            [&] { deserialize_hnsw_node<float>(mutated.data(), mutated.size()); }));
    }

    for (uint64_t hostile : {uint64_t{1} << 32, ~uint64_t{0}, ~uint64_t{0} / sizeof(float)}) {
        auto mutated = blob;
        patch_u64(mutated, dim_offset, hostile);
        assert(throws_runtime_error(
            [&] { deserialize_hnsw_node<float>(mutated.data(), mutated.size()); }));
    }

    std::cout << "  ✓ hostile splices all rejected" << std::endl;
}

void test_config_blob_corruption() {
    std::cout << "Fuzz 4: config blob truncation and bad version..." << std::endl;

    using Index = HNSWIndex<float, L2Metric<float>>;
    Index::Config config;
    auto blob = serialize_hnsw_config<float, L2Metric<float>>(config);

    for (size_t len = 0; len < 40; ++len) {
        assert(throws_runtime_error(
            [&] { (void)deserialize_hnsw_config<float, L2Metric<float>>(blob.data(), len); }));
    }

    auto bad_version = blob;
    bad_version[0] = 0xEE;
    bad_version[1] = 0xEE;
    assert(throws_runtime_error([&] {
        (void)deserialize_hnsw_config<float, L2Metric<float>>(bad_version.data(),
                                                              bad_version.size());
    }));

    auto roundtrip = deserialize_hnsw_config<float, L2Metric<float>>(blob.data(), blob.size());
    assert(roundtrip.M == config.M);
    std::cout << "  ✓ config corruption rejected, valid blob round-trips" << std::endl;
}

void test_deleted_ids_hostile_count() {
    std::cout << "Fuzz 5: deleted-ids blob with hostile count..." << std::endl;

    std::unordered_set<size_t> ids = {1, 2, 3};
    auto blob = serialize_deleted_ids(ids);

    for (uint64_t hostile : {uint64_t{4}, uint64_t{1} << 32, ~uint64_t{0}, ~uint64_t{0} / 8}) {
        auto mutated = blob;
        patch_u64(mutated, 0, hostile);
        assert(throws_runtime_error(
            [&] { (void)deserialize_deleted_ids(mutated.data(), mutated.size()); }));
    }

    auto roundtrip = deserialize_deleted_ids(blob.data(), blob.size());
    assert(roundtrip == ids);
    std::cout << "  ✓ hostile counts rejected, valid blob round-trips" << std::endl;
}

void test_load_index_corrupt_shadow_rows() {
    std::cout << "Fuzz 6: load_hnsw_index over corrupted shadow tables..." << std::endl;

    using Index = HNSWIndex<float, L2Metric<float>>;
    std::mt19937 rng(99);

    Index index;
    std::vector<std::vector<float>> vectors;
    for (size_t i = 0; i < 64; ++i) {
        vectors.push_back(generate_vector(kDim, rng));
        index.insert(i, std::span<const float>(vectors[i]));
    }

    sqlite3* db = nullptr;
    int rc = sqlite3_open(":memory:", &db);
    assert(rc == SQLITE_OK);
    char* err = nullptr;
    rc = save_hnsw_index(db, "main", "fuzz", index, &err);
    assert(rc == SQLITE_OK);

    {
        auto loaded = load_hnsw_index<float, L2Metric<float>>(db, "main", "fuzz", &err);
        assert(loaded.size() == index.size());
    }

    rc = sqlite3_exec(db, "UPDATE fuzz_hnsw_nodes SET data = x'DEADBEEF' WHERE node_id = 5",
                      nullptr, nullptr, nullptr);
    assert(rc == SQLITE_OK);
    assert(throws_runtime_error([&] {
        char* e = nullptr;
        (void)load_hnsw_index<float, L2Metric<float>>(db, "main", "fuzz", &e);
        sqlite3_free(e);
    }));

    sqlite3_close(db);
    std::cout << "  ✓ corrupt node row rejected cleanly" << std::endl;
}

void test_load_index_dangling_edges() {
    std::cout << "Fuzz 7: dangling neighbor ids pruned on load..." << std::endl;

    using Index = HNSWIndex<float, L2Metric<float>>;
    std::mt19937 rng(123);

    Index index;
    std::vector<std::vector<float>> vectors;
    for (size_t i = 0; i < 64; ++i) {
        vectors.push_back(generate_vector(kDim, rng));
        index.insert(i, std::span<const float>(vectors[i]));
    }

    sqlite3* db = nullptr;
    int rc = sqlite3_open(":memory:", &db);
    assert(rc == SQLITE_OK);
    char* err = nullptr;
    rc = save_hnsw_index(db, "main", "fuzz2", index, &err);
    assert(rc == SQLITE_OK);

    const size_t victim = (index.entry_point() + 1) % 64;
    std::string del =
        "DELETE FROM fuzz2_hnsw_nodes WHERE node_id = " + std::to_string(victim);
    rc = sqlite3_exec(db, del.c_str(), nullptr, nullptr, nullptr);
    assert(rc == SQLITE_OK);

    auto loaded = load_hnsw_index<float, L2Metric<float>>(db, "main", "fuzz2", &err);
    assert(loaded.size() == index.size() - 1);

    auto query = generate_vector(kDim, rng);
    auto results = loaded.search_read_mostly(std::span<const float>(query), 10, 64);
    assert(!results.empty());
    for (const auto& [id, dist] : results) {
        assert(id != victim);
        assert(id < 64);
    }

    std::string del_entry = "DELETE FROM fuzz2_hnsw_nodes WHERE node_id = " +
                            std::to_string(index.entry_point());
    rc = sqlite3_exec(db, del_entry.c_str(), nullptr, nullptr, nullptr);
    assert(rc == SQLITE_OK);
    assert(throws_runtime_error([&] {
        char* e = nullptr;
        (void)load_hnsw_index<float, L2Metric<float>>(db, "main", "fuzz2", &e);
        sqlite3_free(e);
    }));

    sqlite3_close(db);
    std::cout << "  ✓ dangling edges pruned, missing entry point rejected" << std::endl;
}

void test_entry_point_blob_corruption() {
    std::cout << "Fuzz 8: undersized entry_point meta blob..." << std::endl;

    using Index = HNSWIndex<float, L2Metric<float>>;
    std::mt19937 rng(321);

    Index index;
    std::vector<std::vector<float>> vectors;
    for (size_t i = 0; i < 8; ++i) {
        vectors.push_back(generate_vector(kDim, rng));
        index.insert(i, std::span<const float>(vectors[i]));
    }

    sqlite3* db = nullptr;
    int rc = sqlite3_open(":memory:", &db);
    assert(rc == SQLITE_OK);
    char* err = nullptr;
    rc = save_hnsw_index(db, "main", "fuzz3", index, &err);
    assert(rc == SQLITE_OK);

    rc = sqlite3_exec(db, "UPDATE fuzz3_hnsw_meta SET value = x'00' WHERE key = 'entry_point'",
                      nullptr, nullptr, nullptr);
    assert(rc == SQLITE_OK);
    assert(throws_runtime_error([&] {
        char* e = nullptr;
        (void)load_hnsw_index<float, L2Metric<float>>(db, "main", "fuzz3", &e);
        sqlite3_free(e);
    }));

    auto [ep_id, ep_layer, max_id] =
        get_hnsw_checkpoint_info<float, L2Metric<float>>(db, "main", "fuzz3");
    assert(ep_id == 0);
    assert(ep_layer == 0);
    assert(max_id == 7);

    sqlite3_close(db);
    std::cout << "  ✓ undersized entry point rejected; checkpoint info degrades safely"
              << std::endl;
}

} // namespace

int main() {
    std::cout << "=== HNSW Persistence Fuzz Tests ===" << std::endl;

    test_node_truncation_sweep();
    test_node_byte_flips();
    test_node_hostile_splices();
    test_config_blob_corruption();
    test_deleted_ids_hostile_count();
    test_load_index_corrupt_shadow_rows();
    test_load_index_dangling_edges();
    test_entry_point_blob_corruption();

    std::cout << "=== All persistence fuzz tests passed ===" << std::endl;
    return 0;
}
