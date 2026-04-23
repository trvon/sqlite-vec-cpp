// SPDX-License-Identifier: Apache-2.0 OR MIT

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <sqlite-vec-cpp/sqlite_vec.hpp>

namespace {

using Clock = std::chrono::high_resolution_clock;

struct BenchConfig {
    size_t rows = 5000;
    size_t queries = 100;
    size_t dim = 128;
    size_t k = 10;
    size_t repeats = 3;
    bool phss = false;
    size_t phss_candidates = 64;
    unsigned int seed = 42;
    bool skip_exact = false;
};

struct Bucket {
    std::uint64_t calls = 0;
    double total_us = 0.0;
};

enum class Phase : std::uint8_t {
    Setup,
    ColdAnn,
    HotAnn,
    HotExact,
};

struct TraceState {
    Phase phase = Phase::Setup;
    std::unordered_map<std::string, Bucket> buckets;
};

const char* phase_name(Phase phase) {
    switch (phase) {
        case Phase::Setup:
            return "setup";
        case Phase::ColdAnn:
            return "cold_ann";
        case Phase::HotAnn:
            return "hot_ann";
        case Phase::HotExact:
            return "hot_exact";
    }
    return "unknown";
}

std::string classify_sql(const char* sql_cstr) {
    const std::string sql = sql_cstr ? sql_cstr : "";
    if (sql.find("embedding MATCH") != std::string::npos)
        return "ann_match";
    if (sql.find("vec_distance_l2(") != std::string::npos)
        return "exact_scan";
    if (sql.find("COUNT(*)") != std::string::npos)
        return "shadow_count";
    if (sql.find("_vectors") != std::string::npos &&
        sql.find("ORDER BY rowid") != std::string::npos)
        return "shadow_vector_scan";
    if (sql.find("INSERT INTO vecs") != std::string::npos)
        return "insert";
    if (sql.find("CREATE VIRTUAL TABLE") != std::string::npos)
        return "create_vtab";
    return "other";
}

int trace_callback(unsigned type, void* ctx, void* p_stmt, void* p_ns) {
    if (type != SQLITE_TRACE_PROFILE)
        return 0;
    auto* state = static_cast<TraceState*>(ctx);
    auto* stmt = static_cast<sqlite3_stmt*>(p_stmt);
    const auto* ns = static_cast<const sqlite3_uint64*>(p_ns);
    const std::string key =
        std::string(phase_name(state->phase)) + ":" + classify_sql(sqlite3_sql(stmt));
    auto& bucket = state->buckets[key];
    bucket.calls += 1;
    bucket.total_us += static_cast<double>(*ns) / 1000.0;
    return 0;
}

BenchConfig parse_args(int argc, char* argv[]) {
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
        } else if (std::strncmp(argv[i], "--repeats=", 10) == 0) {
            cfg.repeats = std::stoul(argv[i] + 10);
        } else if (std::strcmp(argv[i], "--phss") == 0) {
            cfg.phss = true;
        } else if (std::strncmp(argv[i], "--phss-candidates=", 18) == 0) {
            cfg.phss = true;
            cfg.phss_candidates = std::stoul(argv[i] + 18);
        } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
            cfg.seed = static_cast<unsigned int>(std::stoul(argv[i] + 7));
        } else if (std::strcmp(argv[i], "--skip-exact") == 0) {
            cfg.skip_exact = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--rows=N] [--queries=N] [--dim=N] [--k=N] [--repeats=N] "
                        "[--phss] [--phss-candidates=N] [--seed=N] [--skip-exact]\n",
                        argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

std::vector<float> generate_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& value : vec)
        value = dist(rng);
    return vec;
}

std::vector<std::vector<float>> generate_dataset(size_t count, size_t dim, std::mt19937& rng) {
    std::vector<std::vector<float>> data;
    data.reserve(count);
    for (size_t i = 0; i < count; ++i)
        data.push_back(generate_vector(dim, rng));
    return data;
}

void check_rc(sqlite3* db, int rc, const char* context) {
    if (rc != SQLITE_OK) {
        std::fprintf(stderr, "%s failed: %s (rc=%d)\n", context, sqlite3_errmsg(db), rc);
        std::exit(1);
    }
}

std::vector<std::int64_t> collect_ids(sqlite3_stmt* stmt) {
    std::vector<std::int64_t> ids;
    int rc = SQLITE_OK;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW)
        ids.push_back(sqlite3_column_int64(stmt, 0));
    if (rc != SQLITE_DONE) {
        std::fprintf(stderr, "Query step failed with rc=%d\n", rc);
        std::exit(1);
    }
    return ids;
}

void reset_stmt(sqlite3_stmt* stmt) {
    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
}

double elapsed_us(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

std::vector<std::int64_t> run_single_query(sqlite3_stmt* stmt, const std::vector<float>& query,
                                           size_t k, const BenchConfig& cfg) {
    sqlite3_bind_blob(stmt, 1, query.data(), static_cast<int>(query.size() * sizeof(float)),
                      SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(k));
    if (cfg.phss) {
        sqlite3_bind_int(stmt, 3, 1);
        sqlite3_bind_int64(stmt, 4, static_cast<sqlite3_int64>(std::max(k, cfg.phss_candidates)));
    }
    auto ids = collect_ids(stmt);
    reset_stmt(stmt);
    return ids;
}

double run_query_batch(sqlite3_stmt* stmt, const std::vector<std::vector<float>>& queries, size_t k,
                       size_t repeats, const BenchConfig& cfg) {
    const auto start = Clock::now();
    for (size_t rep = 0; rep < repeats; ++rep) {
        for (const auto& query : queries) {
            sqlite3_bind_blob(stmt, 1, query.data(), static_cast<int>(query.size() * sizeof(float)),
                              SQLITE_TRANSIENT);
            sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(k));
            if (cfg.phss) {
                sqlite3_bind_int(stmt, 3, 1);
                sqlite3_bind_int64(stmt, 4,
                                   static_cast<sqlite3_int64>(std::max(k, cfg.phss_candidates)));
            }
            (void)collect_ids(stmt);
            reset_stmt(stmt);
        }
    }
    return elapsed_us(start, Clock::now());
}

double mean_us(const Bucket& bucket) {
    return bucket.calls == 0 ? 0.0 : bucket.total_us / static_cast<double>(bucket.calls);
}

} // namespace

int main(int argc, char* argv[]) {
    const BenchConfig cfg = parse_args(argc, argv);

    sqlite3* db = nullptr;
    check_rc(db, sqlite3_open(":memory:", &db), "sqlite3_open");
    check_rc(db, sqlite3_vec_init(db, nullptr, nullptr), "sqlite3_vec_init");

    TraceState trace;
    check_rc(db, sqlite3_trace_v2(db, SQLITE_TRACE_PROFILE, trace_callback, &trace),
             "sqlite3_trace_v2");

    std::string create_sql =
        "CREATE VIRTUAL TABLE vecs USING vec0(embedding float[" + std::to_string(cfg.dim) + "] )";
    // Keep schema string exact and simple.
    create_sql =
        "CREATE VIRTUAL TABLE vecs USING vec0(embedding float[" + std::to_string(cfg.dim) + "])";
    check_rc(db, sqlite3_exec(db, create_sql.c_str(), nullptr, nullptr, nullptr),
             "CREATE VIRTUAL TABLE");

    std::mt19937 rng(cfg.seed);
    auto corpus = generate_dataset(cfg.rows, cfg.dim, rng);
    auto queries = generate_dataset(cfg.queries, cfg.dim, rng);

    sqlite3_stmt* insert_stmt = nullptr;
    check_rc(db,
             sqlite3_prepare_v2(db, "INSERT INTO vecs(rowid, embedding) VALUES (?, ?)", -1,
                                &insert_stmt, nullptr),
             "prepare insert");

    const auto insert_start = Clock::now();
    for (size_t i = 0; i < corpus.size(); ++i) {
        sqlite3_bind_int64(insert_stmt, 1, static_cast<sqlite3_int64>(i + 1));
        sqlite3_bind_blob(insert_stmt, 2, corpus[i].data(),
                          static_cast<int>(corpus[i].size() * sizeof(float)), SQLITE_TRANSIENT);
        const int rc = sqlite3_step(insert_stmt);
        if (rc != SQLITE_DONE) {
            std::fprintf(stderr, "Insert failed for row %zu: %s\n", i + 1, sqlite3_errmsg(db));
            std::exit(1);
        }
        reset_stmt(insert_stmt);
    }
    const auto insert_us = elapsed_us(insert_start, Clock::now());
    sqlite3_finalize(insert_stmt);

    sqlite3_stmt* ann_stmt = nullptr;
    sqlite3_stmt* exact_stmt = nullptr;
    std::string ann_sql = "SELECT rowid FROM vecs WHERE embedding MATCH ?1 AND k = ?2";
    if (cfg.phss) {
        ann_sql += " AND phss = ?3 AND phss_candidates = ?4";
    }
    ann_sql += " ORDER BY distance";
    check_rc(db, sqlite3_prepare_v2(db, ann_sql.c_str(), -1, &ann_stmt, nullptr),
             "prepare ann query");
    if (!cfg.skip_exact) {
        check_rc(db,
                 sqlite3_prepare_v2(
                     db, "SELECT rowid FROM vecs ORDER BY vec_distance_l2(embedding, ?1) LIMIT ?2",
                     -1, &exact_stmt, nullptr),
                 "prepare exact query");
    }

    trace.phase = Phase::ColdAnn;
    const auto cold_start = Clock::now();
    auto cold_ids = run_single_query(ann_stmt, queries.front(), cfg.k, cfg);
    const auto cold_us = elapsed_us(cold_start, Clock::now());

    trace.phase = Phase::HotAnn;
    const double hot_ann_total_us = run_query_batch(ann_stmt, queries, cfg.k, cfg.repeats, cfg);
    const double hot_ann_mean_us =
        hot_ann_total_us / static_cast<double>(queries.size() * cfg.repeats);

    double hot_exact_total_us = 0.0;
    double hot_exact_mean_us = 0.0;
    if (!cfg.skip_exact) {
        trace.phase = Phase::HotExact;
        hot_exact_total_us =
            run_query_batch(exact_stmt, queries, cfg.k, cfg.repeats, BenchConfig{});
        hot_exact_mean_us = hot_exact_total_us / static_cast<double>(queries.size() * cfg.repeats);
    }
    trace.phase = Phase::Setup;

    std::vector<std::pair<std::string, Bucket>> buckets(trace.buckets.begin(), trace.buckets.end());
    std::sort(buckets.begin(), buckets.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::printf("metric\tvalue\n");
    std::printf("rows\t%zu\n", cfg.rows);
    std::printf("queries\t%zu\n", cfg.queries);
    std::printf("dim\t%zu\n", cfg.dim);
    std::printf("k\t%zu\n", cfg.k);
    std::printf("repeats\t%zu\n", cfg.repeats);
    std::printf("phss\t%d\n", cfg.phss ? 1 : 0);
    std::printf("phss_candidates\t%zu\n", cfg.phss_candidates);
    std::printf("insert_total_us\t%.3f\n", insert_us);
    std::printf("cold_ann_first_query_us\t%.3f\n", cold_us);
    std::printf("cold_ann_first_query_results\t%zu\n", cold_ids.size());
    std::printf("hot_ann_total_us\t%.3f\n", hot_ann_total_us);
    std::printf("hot_ann_mean_us\t%.3f\n", hot_ann_mean_us);
    std::printf("hot_ann_qps\t%.3f\n", hot_ann_mean_us > 0.0 ? 1e6 / hot_ann_mean_us : 0.0);
    if (!cfg.skip_exact) {
        std::printf("hot_exact_total_us\t%.3f\n", hot_exact_total_us);
        std::printf("hot_exact_mean_us\t%.3f\n", hot_exact_mean_us);
        std::printf("hot_exact_qps\t%.3f\n",
                    hot_exact_mean_us > 0.0 ? 1e6 / hot_exact_mean_us : 0.0);
    }

    std::printf("\nstmt_phase\tsql_class\tcalls\ttotal_us\tmean_us\n");
    for (const auto& [key, bucket] : buckets) {
        const auto sep = key.find(':');
        const std::string phase = sep == std::string::npos ? "unknown" : key.substr(0, sep);
        const std::string sql_class = sep == std::string::npos ? key : key.substr(sep + 1);
        std::printf("%s\t%s\t%llu\t%.3f\t%.3f\n", phase.c_str(), sql_class.c_str(),
                    static_cast<unsigned long long>(bucket.calls), bucket.total_us,
                    mean_us(bucket));
    }

    sqlite3_finalize(ann_stmt);
    if (exact_stmt)
        sqlite3_finalize(exact_stmt);
    sqlite3_close(db);
    return 0;
}
