// SPDX-License-Identifier: Apache-2.0 OR MIT

#include <sqlite3.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <sqlite-vec-cpp/sqlite_vec.hpp>

struct BenchConfig {
    size_t rows = 5000;
    size_t queries = 100;
    size_t dim = 128;
    size_t k = 10;
    unsigned int seed = 42;
};

static BenchConfig parseArgs(int argc, char* argv[]) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--rows=", 7) == 0) {
            cfg.rows = std::stoul(argv[i] + 7);
        } else if (std::strncmp(argv[i], "--queries=", 10) == 0) {
            cfg.queries = std::stoul(argv[i] + 10);
        } else if (std::strncmp(argv[i], "--dim=", 6) == 0) {
            cfg.dim = std::stoul(argv[i] + 6);
        } else if (std::strncmp(argv[i], "--k=", 4) == 0) {
            cfg.k = std::stoul(argv[i] + 4);
        } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
            cfg.seed = static_cast<unsigned int>(std::stoul(argv[i] + 7));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--rows=N] [--queries=N] [--dim=N] [--k=N] [--seed=N]\n",
                        argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

static std::vector<float> generateVector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& value : vec) {
        value = dist(rng);
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

static void checkRc(sqlite3* db, int rc, const char* context) {
    if (rc != SQLITE_OK) {
        std::fprintf(stderr, "%s failed: %s (rc=%d)\n", context, sqlite3_errmsg(db), rc);
        std::exit(1);
    }
}

static std::vector<std::int64_t> collectIds(sqlite3_stmt* stmt) {
    std::vector<std::int64_t> ids;
    int rc = SQLITE_OK;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        ids.push_back(sqlite3_column_int64(stmt, 0));
    }
    if (rc != SQLITE_DONE) {
        std::fprintf(stderr, "Query step failed with rc=%d\n", rc);
        std::exit(1);
    }
    return ids;
}

static std::vector<std::vector<std::int64_t>>
runQueryBatch(sqlite3_stmt* stmt, const std::vector<std::vector<float>>& queries, size_t k,
              double& total_us) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::int64_t>> results;
    results.reserve(queries.size());

    for (const auto& query : queries) {
        sqlite3_bind_blob(stmt, 1, query.data(), static_cast<int>(query.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(k));
        results.push_back(collectIds(stmt));
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_us = std::chrono::duration<double, std::micro>(end - start).count();
    return results;
}

static double percent(size_t hits, size_t total) {
    return total == 0 ? 0.0 : (100.0 * static_cast<double>(hits) / static_cast<double>(total));
}

int main(int argc, char* argv[]) {
    BenchConfig cfg = parseArgs(argc, argv);

    std::printf("============================================================\n");
    std::printf("vec0 ANN Benchmark\n");
    std::printf("============================================================\n");
    std::printf("rows=%zu queries=%zu dim=%zu k=%zu seed=%u\n", cfg.rows, cfg.queries, cfg.dim,
                cfg.k, cfg.seed);

    sqlite3* db = nullptr;
    checkRc(db, sqlite3_open(":memory:", &db), "sqlite3_open");
    checkRc(db, sqlite3_vec_init(db, nullptr, nullptr), "sqlite3_vec_init");
    std::string create_sql =
        "CREATE VIRTUAL TABLE vecs USING vec0(embedding float[" + std::to_string(cfg.dim) + "])";
    checkRc(db, sqlite3_exec(db, create_sql.c_str(), nullptr, nullptr, nullptr),
            "CREATE VIRTUAL TABLE");

    std::mt19937 rng(cfg.seed);
    auto corpus = generateDataset(cfg.rows, cfg.dim, rng);
    auto queries = generateDataset(cfg.queries, cfg.dim, rng);

    sqlite3_stmt* insert_stmt = nullptr;
    checkRc(db,
            sqlite3_prepare_v2(db, "INSERT INTO vecs(rowid, embedding) VALUES (?, ?)", -1,
                               &insert_stmt, nullptr),
            "prepare insert");

    for (size_t i = 0; i < corpus.size(); ++i) {
        sqlite3_bind_int64(insert_stmt, 1, static_cast<sqlite3_int64>(i + 1));
        sqlite3_bind_blob(insert_stmt, 2, corpus[i].data(),
                          static_cast<int>(corpus[i].size() * sizeof(float)), SQLITE_TRANSIENT);
        int rc = sqlite3_step(insert_stmt);
        if (rc != SQLITE_DONE) {
            std::fprintf(stderr, "Insert failed for row %zu: %s\n", i + 1, sqlite3_errmsg(db));
            std::exit(1);
        }
        sqlite3_reset(insert_stmt);
        sqlite3_clear_bindings(insert_stmt);
    }
    sqlite3_finalize(insert_stmt);

    sqlite3_stmt* ann_stmt = nullptr;
    sqlite3_stmt* exact_stmt = nullptr;
    checkRc(
        db,
        sqlite3_prepare_v2(db,
                           "SELECT rowid FROM vecs WHERE embedding MATCH ?1 AND k = ?2 ORDER BY "
                           "distance",
                           -1, &ann_stmt, nullptr),
        "prepare ann query");
    checkRc(
        db,
        sqlite3_prepare_v2(db,
                           "SELECT rowid FROM vecs ORDER BY vec_distance_l2(embedding, ?1) LIMIT "
                           "?2",
                           -1, &exact_stmt, nullptr),
        "prepare exact query");

    auto warmup_start = std::chrono::high_resolution_clock::now();
    sqlite3_bind_blob(ann_stmt, 1, queries[0].data(),
                      static_cast<int>(queries[0].size() * sizeof(float)), SQLITE_TRANSIENT);
    sqlite3_bind_int64(ann_stmt, 2, static_cast<sqlite3_int64>(cfg.k));
    (void)collectIds(ann_stmt);
    sqlite3_reset(ann_stmt);
    sqlite3_clear_bindings(ann_stmt);
    auto warmup_end = std::chrono::high_resolution_clock::now();
    double warmup_ms = std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();

    double ann_total_us = 0.0;
    double exact_total_us = 0.0;
    auto ann_results = runQueryBatch(ann_stmt, queries, cfg.k, ann_total_us);
    auto exact_results = runQueryBatch(exact_stmt, queries, cfg.k, exact_total_us);

    size_t total_hits = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
        std::unordered_set<std::int64_t> exact_set(exact_results[i].begin(),
                                                   exact_results[i].end());
        for (std::int64_t id : ann_results[i]) {
            if (exact_set.contains(id)) {
                ++total_hits;
            }
        }
    }

    double ann_latency_us = ann_total_us / static_cast<double>(cfg.queries);
    double exact_latency_us = exact_total_us / static_cast<double>(cfg.queries);

    std::printf("Warm ANN first-query build: %.1f ms\n", warmup_ms);
    std::printf("ANN warm search:   %8.1f us/query | %8.0f QPS | recall@%zu %5.1f%%\n",
                ann_latency_us, 1e6 / ann_latency_us, cfg.k,
                percent(total_hits, cfg.queries * cfg.k));
    std::printf("Exact full scan:   %8.1f us/query | %8.0f QPS\n", exact_latency_us,
                1e6 / exact_latency_us);

    sqlite3_finalize(ann_stmt);
    sqlite3_finalize(exact_stmt);
    sqlite3_close(db);
    return 0;
}
