// SPDX-License-Identifier: Apache-2.0 OR MIT
// Invariant, SIMD-vs-scalar parity, SearchStats telemetry, and incremental vec0 tests

#include <sqlite3.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

#include <sqlite-vec-cpp/distances/hamming.hpp>
#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>
#include <sqlite-vec-cpp/index/hnsw_quantized.hpp>
#include <sqlite-vec-cpp/quantization/lvq.hpp>
#include <sqlite-vec-cpp/quantization/rabitq.hpp>
#include <sqlite-vec-cpp/quantization/store.hpp>
#include <sqlite-vec-cpp/sqlite/vec0_module.hpp>

#define INV_CHECK(cond)                                                                            \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond           \
                      << std::endl;                                                                \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

using namespace sqlite_vec_cpp::distances;
using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::quantization;
using namespace sqlite_vec_cpp::sqlite;
namespace qdetail = sqlite_vec_cpp::quantization::detail;

namespace {

constexpr size_t kOddDims[] = {1, 7, 15, 17, 31, 33, 385, 769};

bool rel_approx_equal(float a, float b, float rel_tol = 1e-4f, float abs_tol = 1e-4f) {
    const float diff = std::abs(a - b);
    return diff <= abs_tol + rel_tol * std::max(std::abs(a), std::abs(b));
}

std::vector<float> make_random_f32(size_t dim, std::mt19937& rng, float lo = -5.0f,
                                   float hi = 5.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(dim);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

std::vector<int8_t> make_random_i8(size_t dim, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> v(dim);
    for (auto& x : v) {
        x = static_cast<int8_t>(dist(rng));
    }
    return v;
}

std::vector<uint8_t> make_random_bytes(size_t n, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> v(n);
    for (auto& x : v) {
        x = static_cast<uint8_t>(dist(rng));
    }
    return v;
}

// 1. Triangle inequality & symmetry for L2 and Hamming
void test_metric_invariants() {
    std::cout << "Testing L2 and Hamming metric invariants (symmetry + triangle inequality)..."
              << std::endl;
    std::mt19937 rng(2026);

    for (size_t dim : kOddDims) {
        auto a = make_random_f32(dim, rng);
        auto b = make_random_f32(dim, rng);
        auto c = make_random_f32(dim, rng);

        const float d_aa = l2_distance(std::span<const float>(a), std::span<const float>(a));
        const float d_ab = l2_distance(std::span<const float>(a), std::span<const float>(b));
        const float d_ba = l2_distance(std::span<const float>(b), std::span<const float>(a));
        const float d_bc = l2_distance(std::span<const float>(b), std::span<const float>(c));
        const float d_ac = l2_distance(std::span<const float>(a), std::span<const float>(c));

        INV_CHECK(d_aa == 0.0f);
        INV_CHECK(d_ab >= 0.0f);
        INV_CHECK(rel_approx_equal(d_ab, d_ba, 1e-5f, 1e-5f));
        INV_CHECK(d_ac <= d_ab + d_bc + 1e-3f);

        // int8 L2
        auto ia = make_random_i8(dim, rng);
        auto ib = make_random_i8(dim, rng);
        auto ic = make_random_i8(dim, rng);
        const float id_aa = l2_distance(std::span<const int8_t>(ia), std::span<const int8_t>(ia));
        const float id_ab = l2_distance(std::span<const int8_t>(ia), std::span<const int8_t>(ib));
        const float id_ba = l2_distance(std::span<const int8_t>(ib), std::span<const int8_t>(ia));
        const float id_bc = l2_distance(std::span<const int8_t>(ib), std::span<const int8_t>(ic));
        const float id_ac = l2_distance(std::span<const int8_t>(ia), std::span<const int8_t>(ic));
        INV_CHECK(id_aa == 0.0f);
        INV_CHECK(id_ab >= 0.0f);
        INV_CHECK(rel_approx_equal(id_ab, id_ba, 1e-5f, 1e-5f));
        INV_CHECK(id_ac <= id_ab + id_bc + 1e-2f);
    }

    // Hamming across both 64-bit-aligned and odd bit lengths
    for (size_t bits : {size_t{7}, size_t{15}, size_t{33}, size_t{64}, size_t{128}, size_t{256},
                        size_t{385}, size_t{512}}) {
        const size_t bytes = (bits + 7) / 8;
        auto a = make_random_bytes(bytes, rng);
        auto b = make_random_bytes(bytes, rng);
        auto c = make_random_bytes(bytes, rng);

        HammingMetric metric(bits);
        const float d_aa = metric(std::span<const uint8_t>(a), std::span<const uint8_t>(a));
        const float d_ab = metric(std::span<const uint8_t>(a), std::span<const uint8_t>(b));
        const float d_ba = metric(std::span<const uint8_t>(b), std::span<const uint8_t>(a));
        const float d_bc = metric(std::span<const uint8_t>(b), std::span<const uint8_t>(c));
        const float d_ac = metric(std::span<const uint8_t>(a), std::span<const uint8_t>(c));

        INV_CHECK(d_aa == 0.0f);
        INV_CHECK(d_ab >= 0.0f);
        INV_CHECK(d_ab == d_ba);
        INV_CHECK(d_ac <= d_ab + d_bc);
    }

    std::cout << "  PASS: metric invariants" << std::endl;
}

// 2. Unaligned float* and int8_t* spans + odd dimensions SIMD-vs-scalar parity
void test_unaligned_and_simd_scalar_parity() {
    std::cout << "Testing unaligned spans and SIMD-vs-scalar L2 parity across odd dims..."
              << std::endl;
    std::mt19937 rng(777);

    for (size_t dim : kOddDims) {
        // Allocate with extra padding so we can test non-16/32-byte aligned float offsets
        std::vector<float> a_buf = make_random_f32(dim + 7, rng);
        std::vector<float> b_buf = make_random_f32(dim + 7, rng);

        for (size_t offset : {size_t{0}, size_t{1}, size_t{2}, size_t{3}}) {
            std::span<const float> a(a_buf.data() + offset, dim);
            std::span<const float> b(b_buf.data() + ((offset + 1) % 4), dim);

            const float sq_simd = l2_squared_distance(a, b);
            const float sq_scalar = l2_squared_distance_float(a, b);
            INV_CHECK(rel_approx_equal(sq_simd, sq_scalar, 1e-4f, 1e-4f));

            const float d_simd = l2_distance(a, b);
            const float d_scalar = l2_distance_float(a, b);
            INV_CHECK(rel_approx_equal(d_simd, d_scalar, 1e-4f, 1e-4f));
            INV_CHECK(rel_approx_equal(d_simd * d_simd, sq_simd, 1e-3f, 1e-3f));
        }

        // Unaligned int8_t spans at odd byte offsets
        std::vector<int8_t> ia_buf = make_random_i8(dim + 15, rng);
        std::vector<int8_t> ib_buf = make_random_i8(dim + 15, rng);
        for (size_t offset : {size_t{0}, size_t{1}, size_t{3}, size_t{5}, size_t{7}}) {
            std::span<const int8_t> ia(ia_buf.data() + offset, dim);
            std::span<const int8_t> ib(ib_buf.data() + ((offset + 2) % 7), dim);

            const float sq_dispatch = l2_squared_distance(ia, ib);
            const float sq_scalar = l2_squared_distance_int(ia, ib);
            INV_CHECK(rel_approx_equal(sq_dispatch, sq_scalar, 1e-5f, 1e-5f));

            const float d_dispatch = l2_distance(ia, ib);
            const float d_scalar = l2_distance_int(ia, ib);
            INV_CHECK(rel_approx_equal(d_dispatch, d_scalar, 1e-5f, 1e-5f));
        }
    }

    std::cout << "  PASS: unaligned spans & SIMD-vs-scalar L2 parity" << std::endl;
}

// 3. Quantized store parity across odd dimensions (LVQ8, LVQ4, RaBitQ) + fused 4-plane popcount
void test_quantized_stores_odd_dims_parity() {
    std::cout << "Testing LVQ8Store, LVQ4Store, RaBitQStore parity across odd dims..." << std::endl;
    std::mt19937 rng(999);

    for (size_t dim : kOddDims) {
        constexpr size_t kN = 16;
        QuantizationSnapshot snap;
        snap.dim = dim;
        snap.generation = 1;
        snap.entries.reserve(kN);
        std::vector<std::vector<float>> corpus;
        std::vector<std::span<const float>> corpus_spans;
        corpus.reserve(kN);
        corpus_spans.reserve(kN);

        for (size_t i = 0; i < kN; ++i) {
            corpus.push_back(make_random_f32(dim, rng, -2.0f, 2.0f));
            corpus_spans.emplace_back(corpus.back());
            snap.entries.push_back({i, corpus.back()});
        }

        // Unaligned query slice
        std::vector<float> q_buf = make_random_f32(dim + 5, rng, -2.0f, 2.0f);
        std::span<const float> query(q_buf.data() + 1, dim);

        // LVQ8Store
        LVQ8Store lvq8;
        lvq8.build(snap);
        INV_CHECK(lvq8.inv_scales.size() == kN);
        for (size_t i = 0; i < kN; ++i) {
            auto ref_code = LVQ8::encode(std::span<const float>(corpus[i]));
            const float ref_dist = LVQ8::l2_distance(query, ref_code);
            const float store_dist = lvq8.l2_distance(query, i);
            const float store_sq = lvq8.l2_distance_squared(query, i);
            INV_CHECK(rel_approx_equal(store_dist, ref_dist, 1e-4f, 1e-4f));
            INV_CHECK(rel_approx_equal(std::sqrt(store_sq), store_dist, 1e-5f, 1e-5f));
        }

        // LVQ4Store
        LVQ4Store lvq4;
        lvq4.build(snap);
        INV_CHECK(lvq4.inv_scales.size() == kN);
        for (size_t i = 0; i < kN; ++i) {
            auto ref_code = LVQ4::encode(std::span<const float>(corpus[i]));
            const float ref_dist = LVQ4::l2_distance(query, ref_code, dim);
            const float store_dist = lvq4.l2_distance(query, i);
            const float store_sq = lvq4.l2_distance_squared(query, i);
            INV_CHECK(rel_approx_equal(store_dist, ref_dist, 1e-4f, 1e-4f));
            INV_CHECK(rel_approx_equal(std::sqrt(store_sq), store_dist, 1e-5f, 1e-5f));
        }

        // RaBitQStore vs RaBitQ standalone + popcount & 4-plane kernel parity
        RaBitQStore rabitq_store;
        rabitq_store.build(snap);
        INV_CHECK(rabitq_store.code_popcount.size() == kN);

        auto centroid = RaBitQ::compute_centroid(corpus_spans);
        RaBitQ rabitq_ref(centroid, kRaBitQDefaultSeed);
        auto ref_qs = rabitq_ref.prepare_query(query);
        auto store_qs = rabitq_store.prepare_query(query);

        for (size_t i = 0; i < kN; ++i) {
            auto ref_code = rabitq_ref.encode(corpus_spans[i]);
            const uint8_t* code_ptr = rabitq_store.bits.data() + i * rabitq_store.bytes_per_vec;
            const size_t nb = rabitq_store.bytes_per_vec;

            // Precomputed popcount matches full popcount
            INV_CHECK(rabitq_store.code_popcount[i] == qdetail::popcount_bytes(code_ptr, nb));
            INV_CHECK(ref_code.code_popcount == qdetail::popcount_bytes(ref_code.bits.data(), nb));

            // Fused 4-plane popcount matches naive 4-pass AND-popcount
            uint64_t naive_su = 0;
            for (size_t j = 0; j < 4; ++j) {
                naive_su += static_cast<uint64_t>(1u << j) *
                            qdetail::and_popcount_bytes(code_ptr, store_qs.planes.data() + j * nb,
                                                       nb);
            }
            const uint64_t fused_su =
                qdetail::and_popcount_4planes(code_ptr, store_qs.planes.data(), nb);
            INV_CHECK(fused_su == naive_su);

            const float ref_dist = rabitq_ref.estimate_l2_distance(ref_qs, ref_code);
            const float store_dist = rabitq_store.l2_distance(store_qs, i);
            const float store_sq = rabitq_store.l2_distance_squared(store_qs, i);
            INV_CHECK(rel_approx_equal(store_dist, ref_dist, 1e-4f, 1e-4f));
            INV_CHECK(rel_approx_equal(std::sqrt(store_sq), store_dist, 1e-5f, 1e-5f));
        }
    }

    std::cout << "  PASS: quantized stores odd-dims & popcount parity" << std::endl;
}

// 4. HNSW SearchStats telemetry tests
void test_hnsw_search_stats() {
    std::cout << "Testing HNSW SearchStats telemetry..." << std::endl;
    std::mt19937 rng(12345);
    constexpr size_t kDim = 32;
    constexpr size_t kN = 200;

    HNSWIndex<float, L2Metric<float>>::Config cfg;
    cfg.M = 12;
    cfg.M_max = 12;
    cfg.M_max_0 = 24;
    cfg.ef_construction = 64;
    cfg.random_seed = 42;
    HNSWIndex<float, L2Metric<float>> index(cfg);

    std::vector<std::vector<float>> corpus;
    corpus.reserve(kN);
    for (size_t i = 0; i < kN; ++i) {
        corpus.push_back(make_random_f32(kDim, rng, -1.0f, 1.0f));
        index.insert_single_threaded(i, std::span<const float>(corpus.back()));
    }

    auto query = make_random_f32(kDim, rng, -1.0f, 1.0f);

    // Unfiltered search with and without stats must match identically
    auto res_no_stats = index.search(query, 10, 32);
    SearchStats stats{};
    auto res_with_stats = index.search(query, 10, 32, &stats);
    INV_CHECK(res_no_stats == res_with_stats);
    INV_CHECK(stats.hops > 0);
    INV_CHECK(stats.distance_evals >= stats.hops);
    INV_CHECK(stats.rerank_evals == 0);
    INV_CHECK(stats.filter_rejections == 0);

    // Filtered search should record filter_rejections
    SearchStats filtered_stats{};
    auto res_filtered = index.search_with_filter(
        query, 10, 32, [](size_t id) { return (id % 2) == 0; }, &filtered_stats);
    INV_CHECK(!res_filtered.empty());
    for (const auto& [id, d] : res_filtered) {
        INV_CHECK((id % 2) == 0);
    }
    INV_CHECK(filtered_stats.hops > 0);
    INV_CHECK(filtered_stats.distance_evals > 0);
    INV_CHECK(filtered_stats.filter_rejections > 0);

    // Quantized two-stage search should record rerank_evals
    for (auto qtype : {QuantizationType::LVQ8, QuantizationType::LVQ4, QuantizationType::RaBitQ}) {
        HNSWQuantizedSearch<float, L2Metric<float>>::Config qcfg;
        qcfg.quantization = qtype;
        qcfg.rerank_factor = 2;
        HNSWQuantizedSearch<float, L2Metric<float>> qsearch(index, qcfg);
        qsearch.build_quantization();

        SearchStats qstats{};
        auto qres_no_stats = qsearch.search(query, 10, 24);
        auto qres_stats = qsearch.search(query, 10, 24, &qstats);
        INV_CHECK(qres_no_stats == qres_stats);
        INV_CHECK(qstats.hops > 0);
        INV_CHECK(qstats.distance_evals > 0);
        INV_CHECK(qstats.rerank_evals >= qres_stats.size());
    }

    // PHSS rerank should record rerank_evals
    HNSWIndex<float, L2Metric<float>>::PhssRerankConfig phss_cfg;
    phss_cfg.enabled = true;
    phss_cfg.candidates = 20;
    SearchStats phss_stats{};
    auto phss_res = index.search_phss_rerank(query, 5, 32, phss_cfg, &phss_stats);
    INV_CHECK(phss_res.size() == 5);
    INV_CHECK(phss_stats.hops > 0);
    INV_CHECK(phss_stats.distance_evals > 0);
    INV_CHECK(phss_stats.rerank_evals > 0);

    std::cout << "  PASS: HNSW SearchStats telemetry" << std::endl;
}

// 5. Incremental vec0 insert/update/delete recall parity & cached statements
void test_vec0_incremental_maintenance() {
    std::cout << "Testing incremental vec0 ANN insert/update/delete recall parity..." << std::endl;

    sqlite3* db = nullptr;
    INV_CHECK(sqlite3_open(":memory:", &db) == SQLITE_OK);
    auto reg = register_vec0_module(db);
    INV_CHECK(reg.has_value());

    constexpr size_t kDim = 16;
    char* err = nullptr;
    int rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE vecs USING vec0(embedding float[16])", nullptr,
                          nullptr, &err);
    INV_CHECK(rc == SQLITE_OK);

    auto bind_and_insert = [&](int64_t rowid, const std::vector<float>& v) {
        sqlite3_stmt* stmt = nullptr;
        int s_rc = sqlite3_prepare_v2(db, "INSERT INTO vecs(rowid, embedding) VALUES (?, ?)", -1,
                                      &stmt, nullptr);
        INV_CHECK(s_rc == SQLITE_OK);
        sqlite3_bind_int64(stmt, 1, rowid);
        sqlite3_bind_blob(stmt, 2, v.data(), static_cast<int>(v.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        s_rc = sqlite3_step(stmt);
        INV_CHECK(s_rc == SQLITE_DONE);
        sqlite3_finalize(stmt);
    };

    auto update_row = [&](int64_t rowid, const std::vector<float>& v) {
        sqlite3_stmt* stmt = nullptr;
        int s_rc = sqlite3_prepare_v2(db, "UPDATE vecs SET embedding = ? WHERE rowid = ?", -1,
                                      &stmt, nullptr);
        INV_CHECK(s_rc == SQLITE_OK);
        sqlite3_bind_blob(stmt, 1, v.data(), static_cast<int>(v.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, rowid);
        s_rc = sqlite3_step(stmt);
        INV_CHECK(s_rc == SQLITE_DONE);
        sqlite3_finalize(stmt);
    };

    auto delete_row = [&](int64_t rowid) {
        sqlite3_stmt* stmt = nullptr;
        int s_rc = sqlite3_prepare_v2(db, "DELETE FROM vecs WHERE rowid = ?", -1, &stmt, nullptr);
        INV_CHECK(s_rc == SQLITE_OK);
        sqlite3_bind_int64(stmt, 1, rowid);
        s_rc = sqlite3_step(stmt);
        INV_CHECK(s_rc == SQLITE_DONE);
        sqlite3_finalize(stmt);
    };

    auto query_ann = [&](const std::vector<float>& q, int k, int ef = 64) {
        sqlite3_stmt* stmt = nullptr;
        int s_rc = sqlite3_prepare_v2(
            db,
            "SELECT rowid, distance FROM vecs WHERE embedding MATCH ? AND k = ? AND ef_search = ?",
            -1, &stmt, nullptr);
        INV_CHECK(s_rc == SQLITE_OK);
        sqlite3_bind_blob(stmt, 1, q.data(), static_cast<int>(q.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 2, k);
        sqlite3_bind_int(stmt, 3, ef);
        std::vector<std::pair<int64_t, float>> out;
        while ((s_rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            out.emplace_back(sqlite3_column_int64(stmt, 0),
                             static_cast<float>(sqlite3_column_double(stmt, 1)));
        }
        INV_CHECK(s_rc == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return out;
    };

    std::mt19937 rng(31415);
    std::unordered_map<int64_t, std::vector<float>> truth;

    // Phase 1: Initial population (60 vectors) + warm ANN index
    for (int64_t id = 1; id <= 60; ++id) {
        auto v = make_random_f32(kDim, rng, -1.0f, 1.0f);
        bind_and_insert(id, v);
        truth[id] = std::move(v);
    }

    auto q0 = make_random_f32(kDim, rng, -1.0f, 1.0f);
    auto init_res = query_ann(q0, 5);
    INV_CHECK(init_res.size() == 5);

    // Grab the built ann_index pointer to verify incremental inserts do NOT invalidate it
    Vec0AnnIndex* initial_ptr = nullptr;
    vec0_with_table(db, "main", "vecs", [&](Vec0Table* t) {
        INV_CHECK(t != nullptr);
        INV_CHECK(t->ann_ready);
        INV_CHECK(t->ann_index != nullptr);
        initial_ptr = t->ann_index.get();
    });

    // Phase 2: Incremental INSERTs (40 more vectors) — ann_index must stay ready and same instance
    for (int64_t id = 61; id <= 100; ++id) {
        auto v = make_random_f32(kDim, rng, -1.0f, 1.0f);
        bind_and_insert(id, v);
        truth[id] = std::move(v);
    }

    vec0_with_table(db, "main", "vecs", [&](Vec0Table* t) {
        INV_CHECK(t != nullptr);
        INV_CHECK(t->ann_ready);
        INV_CHECK(t->ann_index.get() == initial_ptr);
        INV_CHECK(t->ann_index->active_size() == 100);
    });

    // Verify a newly inserted vector is found as exact top-1 match
    auto res_new = query_ann(truth[95], 1);
    INV_CHECK(res_new.size() == 1);
    INV_CHECK(res_new[0].first == 95);
    INV_CHECK(res_new[0].second < 1e-4f);

    // Phase 3: Incremental UPDATE (in-place same rowid)
    std::vector<float> updated_vec(kDim, 9.5f);
    update_row(10, updated_vec);
    truth[10] = updated_vec;

    vec0_with_table(db, "main", "vecs", [&](Vec0Table* t) {
        INV_CHECK(t != nullptr);
        INV_CHECK(t->ann_ready);
        INV_CHECK(t->ann_index != nullptr);
        INV_CHECK(t->ann_index->active_size() == 100);
    });

    auto res_upd = query_ann(updated_vec, 1);
    INV_CHECK(res_upd.size() == 1);
    INV_CHECK(res_upd[0].first == 10);
    INV_CHECK(res_upd[0].second < 1e-4f);

    // Phase 4: Incremental DELETEs (30% of rows -> triggers compaction cleanly)
    std::unordered_set<int64_t> deleted;
    for (int64_t id = 1; id <= 30; ++id) {
        if (id == 10)
            continue;
        delete_row(id);
        truth.erase(id);
        deleted.insert(id);
    }

    vec0_with_table(db, "main", "vecs", [&](Vec0Table* t) {
        INV_CHECK(t != nullptr);
        INV_CHECK(t->ann_ready);
        INV_CHECK(t->ann_index != nullptr);
        INV_CHECK(t->ann_index->active_size() == truth.size());
    });

    // Phase 5: Re-insert a previously deleted rowid with a distinct vector
    std::vector<float> reinserted_vec(kDim, -8.25f);
    bind_and_insert(5, reinserted_vec);
    truth[5] = reinserted_vec;
    deleted.erase(5);

    auto res_reins = query_ann(reinserted_vec, 1);
    INV_CHECK(res_reins.size() == 1);
    INV_CHECK(res_reins[0].first == 5);
    INV_CHECK(res_reins[0].second < 1e-4f);

    // Phase 6: Recall parity vs brute-force truth over 20 random queries
    double total_recall = 0.0;
    constexpr int kQueries = 20;
    constexpr int kTopK = 10;
    for (int qi = 0; qi < kQueries; ++qi) {
        auto q = make_random_f32(kDim, rng, -1.0f, 1.0f);
        auto ann_res = query_ann(q, kTopK, 64);
        INV_CHECK(ann_res.size() == static_cast<size_t>(kTopK));

        // Ensure no deleted rowid ever leaks into ANN results
        for (const auto& [rid, dist] : ann_res) {
            INV_CHECK(!deleted.contains(rid));
            INV_CHECK(truth.contains(rid));
        }

        // Brute-force top-K over live truth
        std::vector<std::pair<int64_t, float>> exact;
        exact.reserve(truth.size());
        for (const auto& [rid, vec] : truth) {
            exact.emplace_back(rid, l2_distance(std::span<const float>(q),
                                                std::span<const float>(vec)));
        }
        std::sort(exact.begin(), exact.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
        exact.resize(kTopK);

        std::unordered_set<int64_t> exact_ids;
        for (const auto& [rid, _] : exact) {
            exact_ids.insert(rid);
        }
        int hits = 0;
        for (const auto& [rid, _] : ann_res) {
            if (exact_ids.contains(rid)) {
                ++hits;
            }
        }
        total_recall += static_cast<double>(hits) / static_cast<double>(kTopK);
    }

    const double mean_recall = total_recall / static_cast<double>(kQueries);
    std::cout << "  vec0 incremental mean recall@" << kTopK << ": " << mean_recall << std::endl;
    INV_CHECK(mean_recall >= 0.90);

    // Cleanly drop table to exercise finalize_cached_stmts() before shadow table drop
    rc = sqlite3_exec(db, "DROP TABLE vecs", nullptr, nullptr, &err);
    INV_CHECK(rc == SQLITE_OK);
    sqlite3_close(db);

    std::cout << "  PASS: incremental vec0 ANN maintenance & recall parity" << std::endl;
}

} // namespace

int main() {
    test_metric_invariants();
    test_unaligned_and_simd_scalar_parity();
    test_quantized_stores_odd_dims_parity();
    test_hnsw_search_stats();
    test_vec0_incremental_maintenance();
    std::cout << "All invariant tests passed!" << std::endl;
    return 0;
}
