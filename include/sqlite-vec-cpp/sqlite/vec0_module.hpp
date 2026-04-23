#pragma once

#include <sqlite3.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "../distances/l2.hpp"
#include "../index/hnsw.hpp"
#include "../utils/error.hpp"
#include "parsers.hpp"
#include "value.hpp"
#include "vtab.hpp"

namespace sqlite_vec_cpp::sqlite {

// Vec0 Virtual Table Implementation
// Provides full CREATE VIRTUAL TABLE ... USING vec0(...) support
// Compatible with original sqlite-vec C implementation

using Vec0AnnIndex = index::HNSWIndex<float, distances::L2Metric<float>>;

constexpr int kVec0ColRowid = 0;
constexpr int kVec0ColEmbedding = 1;
constexpr int kVec0ColDistance = 2;
constexpr int kVec0ColK = 3;
constexpr int kVec0ColEfSearch = 4;
constexpr int kVec0ColQuery = 5;
constexpr int kVec0ColPhss = 6;
constexpr int kVec0ColPhssCandidates = 7;

constexpr int kVec0PlanFullScan = 0;
constexpr int kVec0PlanAnnMatch = 1 << 0;
constexpr int kVec0PlanAnnHiddenQuery = 1 << 1;
constexpr int kVec0PlanHasK = 1 << 2;
constexpr int kVec0PlanHasEfSearch = 1 << 3;
constexpr int kVec0PlanHasRowidEq = 1 << 4;
constexpr int kVec0PlanHasRowidIn = 1 << 5;
constexpr int kVec0PlanExactMatch = 1 << 6;
constexpr int kVec0PlanExactHiddenQuery = 1 << 7;
constexpr int kVec0PlanHasRowidGt = 1 << 8;
constexpr int kVec0PlanHasRowidGe = 1 << 9;
constexpr int kVec0PlanHasRowidLt = 1 << 10;
constexpr int kVec0PlanHasRowidLe = 1 << 11;
constexpr int kVec0PlanHasPhss = 1 << 12;
constexpr int kVec0PlanHasPhssCandidates = 1 << 13;

constexpr size_t kVec0DefaultK = 10;
constexpr size_t kVec0DefaultEfSearch = 64;
constexpr size_t kVec0DefaultPhssCandidates = 64;

struct Vec0PhssRerankOptions {
    bool enabled = false;
    size_t candidates = kVec0DefaultPhssCandidates;
};

struct Vec0Table {
    sqlite3_vtab base{};
    sqlite3* db = nullptr;
    std::string schema_name;
    std::string table_name;
    std::string embedding_column;
    size_t dimensions = 384;
    bool use_shadow_tables = true;
    std::shared_ptr<Vec0AnnIndex> ann_index;
    bool ann_ready = false;
    mutable std::mutex ann_mutex;

    Vec0Table() = default;
};

struct Vec0Cursor {
    sqlite3_vtab_cursor base{};
    Vec0Table* table = nullptr;
    sqlite3_stmt* stmt = nullptr; // For shadow table queries
    int64_t current_rowid = 0;
    bool eof = true;
    bool ann_mode = false;
    float current_distance = 0.0f;
    size_t ann_pos = 0;
    std::vector<std::pair<int64_t, float>> ann_results;
    std::shared_ptr<Vec0AnnIndex> ann_index;

    Vec0Cursor() = default;

    ~Vec0Cursor() {
        if (stmt) {
            sqlite3_finalize(stmt);
            stmt = nullptr;
        }
    }
};

struct Vec0RowidFilter {
    bool active = false;
    std::unordered_set<int64_t> allowed_rowids;
    std::optional<int64_t> min_rowid;
    std::optional<int64_t> max_rowid;
    bool include_min = true;
    bool include_max = true;

    [[nodiscard]] bool matches(int64_t rowid) const {
        if (!allowed_rowids.empty() && !allowed_rowids.contains(rowid)) {
            return false;
        }
        if (min_rowid) {
            if (include_min) {
                if (rowid < *min_rowid) {
                    return false;
                }
            } else if (rowid <= *min_rowid) {
                return false;
            }
        }
        if (max_rowid) {
            if (include_max) {
                if (rowid > *max_rowid) {
                    return false;
                }
            } else if (rowid >= *max_rowid) {
                return false;
            }
        }
        return !active || true;
    }
};

inline int vec0_set_error(Vec0Table* table, const std::string& message,
                          int rc = SQLITE_CONSTRAINT) {
    VTab(&table->base).set_error(message);
    return rc;
}

inline std::string vec0_vectors_table_name(const Vec0Table& table) {
    std::ostringstream sql;
    sql << "\"" << table.schema_name << "\".\"" << table.table_name << "_vectors\"";
    return sql.str();
}

inline void vec0_invalidate_ann_index(Vec0Table* table) {
    std::lock_guard<std::mutex> lock(table->ann_mutex);
    table->ann_index.reset();
    table->ann_ready = false;
}

inline Result<std::shared_ptr<Vec0AnnIndex>> vec0_ensure_ann_index(Vec0Table* table) {
    std::lock_guard<std::mutex> lock(table->ann_mutex);
    if (table->ann_ready && table->ann_index) {
        return Result<std::shared_ptr<Vec0AnnIndex>>(table->ann_index);
    }

    const std::string vectors_table = vec0_vectors_table_name(*table);

    sqlite3_stmt* count_stmt = nullptr;
    std::string count_sql = "SELECT COUNT(*) FROM " + vectors_table;
    int rc = sqlite3_prepare_v2(table->db, count_sql.c_str(), -1, &count_stmt, nullptr);
    if (rc != SQLITE_OK) {
        return err<std::shared_ptr<Vec0AnnIndex>>(
            Error::sqlite_error("Failed to count vec0 rows", rc));
    }

    size_t row_count = 0;
    if (sqlite3_step(count_stmt) == SQLITE_ROW) {
        row_count = static_cast<size_t>(sqlite3_column_int64(count_stmt, 0));
    }
    sqlite3_finalize(count_stmt);

    auto cfg = Vec0AnnIndex::Config::for_corpus(std::max<size_t>(row_count, 1), table->dimensions);
    cfg.normalize_vectors = false;

    auto ann_index = std::make_shared<Vec0AnnIndex>(cfg);
    ann_index->reserve(row_count);

    sqlite3_stmt* scan_stmt = nullptr;
    std::string scan_sql = "SELECT rowid, \"" + table->embedding_column + "\" FROM " +
                           vectors_table + " ORDER BY rowid";
    rc = sqlite3_prepare_v2(table->db, scan_sql.c_str(), -1, &scan_stmt, nullptr);
    if (rc != SQLITE_OK) {
        return err<std::shared_ptr<Vec0AnnIndex>>(
            Error::sqlite_error("Failed to scan vec0 vectors", rc));
    }

    while ((rc = sqlite3_step(scan_stmt)) == SQLITE_ROW) {
        std::int64_t rowid = sqlite3_column_int64(scan_stmt, 0);
        const void* blob = sqlite3_column_blob(scan_stmt, 1);
        int bytes = sqlite3_column_bytes(scan_stmt, 1);

        if (!blob || bytes <= 0) {
            continue;
        }

        const int expected_bytes = static_cast<int>(table->dimensions * sizeof(float));
        if (bytes != expected_bytes) {
            sqlite3_finalize(scan_stmt);
            return err<std::shared_ptr<Vec0AnnIndex>>(Error::invalid_argument(
                "vec0 stored vector size mismatch for rowid " + std::to_string(rowid)));
        }

        std::vector<float> vec(table->dimensions);
        std::memcpy(vec.data(), blob, static_cast<size_t>(bytes));
        ann_index->insert_single_threaded(static_cast<size_t>(rowid), std::span<const float>(vec));
    }

    if (rc != SQLITE_DONE) {
        sqlite3_finalize(scan_stmt);
        return err<std::shared_ptr<Vec0AnnIndex>>(
            Error::sqlite_error("Failed while rebuilding vec0 ANN index", rc));
    }

    sqlite3_finalize(scan_stmt);
    table->ann_index = ann_index;
    table->ann_ready = true;
    return Result<std::shared_ptr<Vec0AnnIndex>>(ann_index);
}

inline Result<std::vector<float>> vec0_parse_query_vector(Vec0Table* table,
                                                          const Value& query_value) {
    auto parsed_query = parse_vector_from_value<float>(query_value);
    if (!parsed_query) {
        return err<std::vector<float>>(parsed_query.error());
    }
    if (parsed_query->size() != table->dimensions) {
        return err<std::vector<float>>(Error::invalid_argument(
            "vec0 query dimension mismatch: expected " + std::to_string(table->dimensions) +
            ", got " + std::to_string(parsed_query->size())));
    }
    return Result<std::vector<float>>(std::move(parsed_query.value()));
}

inline Result<std::vector<std::pair<int64_t, float>>>
vec0_run_ann_query(Vec0Table* table, const Value& query_value, size_t k, size_t ef_search,
                   const Vec0RowidFilter& rowid_filter,
                   const Vec0PhssRerankOptions& phss_options = {}) {
    auto parsed_query = vec0_parse_query_vector(table, query_value);
    if (!parsed_query) {
        return err<std::vector<std::pair<int64_t, float>>>(parsed_query.error());
    }

    auto ann_index = vec0_ensure_ann_index(table);
    if (!ann_index) {
        return err<std::vector<std::pair<int64_t, float>>>(ann_index.error());
    }

    std::vector<std::pair<int64_t, float>> results;

    if (rowid_filter.active) {
        results.reserve(rowid_filter.allowed_rowids.size());
        for (int64_t rowid : rowid_filter.allowed_rowids) {
            const auto* node = ann_index.value()->get_node(static_cast<size_t>(rowid));
            if (!node || node->vector.empty()) {
                continue;
            }

            float dist = distances::l2_distance(std::span<const float>(parsed_query.value()),
                                                std::span<const float>(node->vector));
            results.emplace_back(rowid, dist);
        }

        std::sort(results.begin(), results.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
        if (results.size() > k) {
            results.resize(k);
        }
    } else {
        std::vector<std::pair<size_t, float>> raw_results;
        if (phss_options.enabled) {
            typename Vec0AnnIndex::PhssRerankConfig cfg;
            cfg.enabled = true;
            cfg.candidates = std::max(k, phss_options.candidates);
            raw_results = ann_index.value()->search_phss_rerank(
                std::span<const float>(parsed_query.value()), k, ef_search, cfg);
        } else {
            raw_results = ann_index.value()->search_read_mostly(
                std::span<const float>(parsed_query.value()), k, ef_search);
        }
        results.reserve(raw_results.size());
        for (const auto& [id, dist] : raw_results) {
            results.emplace_back(static_cast<int64_t>(id), dist);
        }
    }

    return Result<std::vector<std::pair<int64_t, float>>>(std::move(results));
}

inline Result<std::vector<std::pair<int64_t, float>>>
vec0_run_exact_query(Vec0Table* table, const Value& query_value, std::optional<size_t> k,
                     const Vec0RowidFilter& rowid_filter) {
    auto parsed_query = vec0_parse_query_vector(table, query_value);
    if (!parsed_query) {
        return err<std::vector<std::pair<int64_t, float>>>(parsed_query.error());
    }

    sqlite3_stmt* stmt = nullptr;
    std::string sql = "SELECT rowid, \"" + table->embedding_column + "\" FROM " +
                      vec0_vectors_table_name(*table) + " ORDER BY rowid";
    int rc = sqlite3_prepare_v2(table->db, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return err<std::vector<std::pair<int64_t, float>>>(
            Error::sqlite_error("Failed to scan vec0 vectors for exact query", rc));
    }

    std::vector<std::pair<int64_t, float>> results;
    std::vector<float> buffer(table->dimensions);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        int64_t rowid = sqlite3_column_int64(stmt, 0);
        if (!rowid_filter.matches(rowid)) {
            continue;
        }

        const void* blob = sqlite3_column_blob(stmt, 1);
        int bytes = sqlite3_column_bytes(stmt, 1);
        if (!blob || bytes != static_cast<int>(table->dimensions * sizeof(float))) {
            sqlite3_finalize(stmt);
            return err<std::vector<std::pair<int64_t, float>>>(Error::invalid_argument(
                "vec0 exact query encountered stored vector size mismatch for rowid " +
                std::to_string(rowid)));
        }

        std::memcpy(buffer.data(), blob, static_cast<size_t>(bytes));
        float dist = distances::l2_distance(std::span<const float>(parsed_query.value()),
                                            std::span<const float>(buffer));
        results.emplace_back(rowid, dist);
    }

    if (rc != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return err<std::vector<std::pair<int64_t, float>>>(
            Error::sqlite_error("Failed during vec0 exact query scan", rc));
    }

    sqlite3_finalize(stmt);
    std::sort(results.begin(), results.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
    if (k && results.size() > *k) {
        results.resize(*k);
    }
    return Result<std::vector<std::pair<int64_t, float>>>(std::move(results));
}

inline Result<Vec0RowidFilter> vec0_extract_rowid_filter(int idxNum, int argc, sqlite3_value** argv,
                                                         int& arg_idx) {
    Vec0RowidFilter filter;

    if ((idxNum & kVec0PlanHasRowidEq) != 0) {
        if (arg_idx >= argc) {
            return err<Vec0RowidFilter>(
                Error::invalid_argument("vec0 ANN query missing rowid equality argument"));
        }
        filter.active = true;
        filter.allowed_rowids.insert(sqlite3_value_int64(argv[arg_idx++]));
    }

    if ((idxNum & kVec0PlanHasRowidIn) != 0) {
        if (arg_idx >= argc) {
            return err<Vec0RowidFilter>(
                Error::invalid_argument("vec0 ANN query missing rowid IN argument"));
        }

        filter.active = true;
        sqlite3_value* list_value = argv[arg_idx++];
        sqlite3_value* item_value = nullptr;
        int rc = sqlite3_vtab_in_first(list_value, &item_value);
        if (rc != SQLITE_OK && rc != SQLITE_DONE) {
            return err<Vec0RowidFilter>(
                Error::sqlite_error("Failed to iterate vec0 rowid IN values", rc));
        }
        while (rc == SQLITE_OK && item_value) {
            filter.allowed_rowids.insert(sqlite3_value_int64(item_value));
            rc = sqlite3_vtab_in_next(list_value, &item_value);
        }
        if (rc != SQLITE_DONE) {
            return err<Vec0RowidFilter>(
                Error::sqlite_error("Failed to advance vec0 rowid IN values", rc));
        }
    }

    if ((idxNum & kVec0PlanHasRowidGt) != 0) {
        if (arg_idx >= argc) {
            return err<Vec0RowidFilter>(
                Error::invalid_argument("vec0 query missing rowid greater-than argument"));
        }
        filter.active = true;
        filter.min_rowid = sqlite3_value_int64(argv[arg_idx++]);
        filter.include_min = false;
    }

    if ((idxNum & kVec0PlanHasRowidGe) != 0) {
        if (arg_idx >= argc) {
            return err<Vec0RowidFilter>(
                Error::invalid_argument("vec0 query missing rowid greater-equal argument"));
        }
        filter.active = true;
        filter.min_rowid = sqlite3_value_int64(argv[arg_idx++]);
        filter.include_min = true;
    }

    if ((idxNum & kVec0PlanHasRowidLt) != 0) {
        if (arg_idx >= argc) {
            return err<Vec0RowidFilter>(
                Error::invalid_argument("vec0 query missing rowid less-than argument"));
        }
        filter.active = true;
        filter.max_rowid = sqlite3_value_int64(argv[arg_idx++]);
        filter.include_max = false;
    }

    if ((idxNum & kVec0PlanHasRowidLe) != 0) {
        if (arg_idx >= argc) {
            return err<Vec0RowidFilter>(
                Error::invalid_argument("vec0 query missing rowid less-equal argument"));
        }
        filter.active = true;
        filter.max_rowid = sqlite3_value_int64(argv[arg_idx++]);
        filter.include_max = true;
    }

    return Result<Vec0RowidFilter>(std::move(filter));
}

// Helper: Parse CREATE VIRTUAL TABLE statement for embedding column and dimensions
inline bool parse_vec0_schema(int argc, const char* const* argv, std::string& embedding_col,
                              size_t& dims) {
    // Format: CREATE VIRTUAL TABLE name USING vec0(embedding float[dims])
    embedding_col = "embedding"; // default
    dims = 384;                  // default

    for (int i = 3; i < argc; i++) {
        std::string arg(argv[i]);

        // Look for pattern: column_name float[dimensions]
        auto float_pos = arg.find("float[");
        if (float_pos != std::string::npos) {
            // Extract column name (everything before "float[")
            embedding_col = arg.substr(0, float_pos);
            // Trim whitespace
            while (!embedding_col.empty() && std::isspace(embedding_col.back())) {
                embedding_col.pop_back();
            }

            // Extract dimensions
            auto end_bracket = arg.find(']', float_pos);
            if (end_bracket != std::string::npos) {
                try {
                    std::string dim_str = arg.substr(float_pos + 6, end_bracket - float_pos - 6);
                    dims = std::stoul(dim_str);
                    return true;
                } catch (...) {
                    // Failed to parse dimensions - continue with defaults
                    continue;
                }
            }
        }
    }
    return false;
}

// Helper: Create shadow tables for vec0 storage
inline int create_shadow_tables(sqlite3* db, const char* schema, const char* table,
                                const char* embedding_col, [[maybe_unused]] size_t dims,
                                char** pzErr) {
    // Note: dims parameter is reserved for future use (validation, typed vectors)
    (void)dims;

    // Shadow table for metadata
    std::ostringstream meta_sql;
    meta_sql << "CREATE TABLE IF NOT EXISTS \"" << schema << "\".\"" << table << "_metadata\" ("
             << "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
             << "data BLOB)";

    char* err = nullptr;
    int rc = sqlite3_exec(db, meta_sql.str().c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        if (pzErr && err) {
            *pzErr = sqlite3_mprintf("Failed to create metadata shadow table: %s", err);
        }
        sqlite3_free(err);
        return rc;
    }

    // Shadow table for vectors
    std::ostringstream vec_sql;
    vec_sql << "CREATE TABLE IF NOT EXISTS \"" << schema << "\".\"" << table << "_vectors\" ("
            << "rowid INTEGER PRIMARY KEY, "
            << "\"" << embedding_col << "\" BLOB)";

    rc = sqlite3_exec(db, vec_sql.str().c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        if (pzErr && err) {
            *pzErr = sqlite3_mprintf("Failed to create vectors shadow table: %s", err);
        }
        sqlite3_free(err);
        return rc;
    }

    return SQLITE_OK;
}

// xCreate: Called when CREATE VIRTUAL TABLE is executed
inline int vec0Create(sqlite3* db, void* pAux, int argc, const char* const* argv,
                      sqlite3_vtab** ppVTab, char** pzErr) {
    (void)pAux;

    // Enable shadow table writes through xUpdate
    sqlite3_vtab_config(db, SQLITE_VTAB_DIRECTONLY);

    if (argc < 3) {
        *pzErr = sqlite3_mprintf("vec0: insufficient arguments");
        return SQLITE_ERROR;
    }

    std::string embedding_col;
    size_t dims;
    parse_vec0_schema(argc, argv, embedding_col, dims);

    auto* table = new Vec0Table();
    table->db = db;
    table->schema_name = argv[1];
    table->table_name = argv[2];
    table->embedding_column = embedding_col;
    table->dimensions = dims;

    // Create shadow tables for actual storage
    int rc = create_shadow_tables(db, argv[1], argv[2], embedding_col.c_str(), dims, pzErr);
    if (rc != SQLITE_OK) {
        delete table;
        return rc;
    }

    // Declare virtual table schema
    std::ostringstream schema;
    schema << "CREATE TABLE x(rowid INTEGER PRIMARY KEY, "
           << "\"" << embedding_col << "\", "
           << "distance HIDDEN, "
           << "k HIDDEN, "
           << "ef_search HIDDEN, "
           << "query HIDDEN, "
           << "phss HIDDEN, "
           << "phss_candidates HIDDEN)";

    rc = sqlite3_declare_vtab(db, schema.str().c_str());
    if (rc != SQLITE_OK) {
        delete table;
        *pzErr = sqlite3_mprintf("Failed to declare vtab schema");
        return rc;
    }

    *ppVTab = &table->base;
    return SQLITE_OK;
}

// xConnect: Called when connecting to existing virtual table
inline int vec0Connect(sqlite3* db, void* pAux, int argc, const char* const* argv,
                       sqlite3_vtab** ppVTab, char** pzErr) {
    // For vec0, Connect behaves same as Create (shadow tables already exist)
    return vec0Create(db, pAux, argc, argv, ppVTab, pzErr);
}

// xDisconnect: Called when disconnecting from table
inline int vec0Disconnect(sqlite3_vtab* pVTab) {
    auto* table = reinterpret_cast<Vec0Table*>(pVTab);
    delete table;
    return SQLITE_OK;
}

// xDestroy: Called when DROP TABLE is executed
inline int vec0Destroy(sqlite3_vtab* pVTab) {
    auto* table = reinterpret_cast<Vec0Table*>(pVTab);

    // Drop shadow tables
    std::ostringstream drop_meta;
    drop_meta << "DROP TABLE IF EXISTS \"" << table->schema_name << "\".\"" << table->table_name
              << "_metadata\"";
    sqlite3_exec(table->db, drop_meta.str().c_str(), nullptr, nullptr, nullptr);

    std::ostringstream drop_vec;
    drop_vec << "DROP TABLE IF EXISTS \"" << table->schema_name << "\".\"" << table->table_name
             << "_vectors\"";
    sqlite3_exec(table->db, drop_vec.str().c_str(), nullptr, nullptr, nullptr);

    delete table;
    return SQLITE_OK;
}

// xBestIndex: Query planner callback
inline int vec0BestIndex(sqlite3_vtab* pVTab, sqlite3_index_info* pInfo) {
    (void)pVTab;

    int match_constraint = -1;
    int query_constraint = -1;
    int k_constraint = -1;
    int ef_constraint = -1;
    int phss_constraint = -1;
    int phss_candidates_constraint = -1;
    int rowid_eq_constraint = -1;
    int rowid_in_constraint = -1;
    int rowid_gt_constraint = -1;
    int rowid_ge_constraint = -1;
    int rowid_lt_constraint = -1;
    int rowid_le_constraint = -1;

    for (int i = 0; i < pInfo->nConstraint; ++i) {
        const auto& constraint = pInfo->aConstraint[i];
        if (!constraint.usable) {
            continue;
        }

        if (constraint.iColumn == kVec0ColEmbedding &&
            constraint.op == SQLITE_INDEX_CONSTRAINT_MATCH && match_constraint < 0) {
            match_constraint = i;
        } else if (constraint.iColumn == kVec0ColQuery &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_EQ && query_constraint < 0) {
            query_constraint = i;
        } else if (constraint.iColumn == kVec0ColK && constraint.op == SQLITE_INDEX_CONSTRAINT_EQ &&
                   k_constraint < 0) {
            k_constraint = i;
        } else if (constraint.iColumn == kVec0ColEfSearch &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_EQ && ef_constraint < 0) {
            ef_constraint = i;
        } else if (constraint.iColumn == kVec0ColPhss &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_EQ && phss_constraint < 0) {
            phss_constraint = i;
        } else if (constraint.iColumn == kVec0ColPhssCandidates &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_EQ && phss_candidates_constraint < 0) {
            phss_candidates_constraint = i;
        } else if ((constraint.iColumn == kVec0ColRowid || constraint.iColumn < 0) &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_EQ) {
            if (sqlite3_vtab_in(pInfo, i, -1)) {
                if (rowid_in_constraint < 0 && rowid_eq_constraint < 0) {
                    rowid_in_constraint = i;
                }
            } else if (rowid_eq_constraint < 0 && rowid_in_constraint < 0) {
                rowid_eq_constraint = i;
            }
        } else if ((constraint.iColumn == kVec0ColRowid || constraint.iColumn < 0) &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_GT && rowid_gt_constraint < 0) {
            rowid_gt_constraint = i;
        } else if ((constraint.iColumn == kVec0ColRowid || constraint.iColumn < 0) &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_GE && rowid_ge_constraint < 0) {
            rowid_ge_constraint = i;
        } else if ((constraint.iColumn == kVec0ColRowid || constraint.iColumn < 0) &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_LT && rowid_lt_constraint < 0) {
            rowid_lt_constraint = i;
        } else if ((constraint.iColumn == kVec0ColRowid || constraint.iColumn < 0) &&
                   constraint.op == SQLITE_INDEX_CONSTRAINT_LE && rowid_le_constraint < 0) {
            rowid_le_constraint = i;
        }
    }

    const bool has_query_constraint = (match_constraint >= 0 || query_constraint >= 0);
    const bool has_rowid_range = (rowid_gt_constraint >= 0 || rowid_ge_constraint >= 0 ||
                                  rowid_lt_constraint >= 0 || rowid_le_constraint >= 0);
    const bool safe_ann_order =
        (pInfo->nOrderBy == 0) ||
        (pInfo->nOrderBy == 1 && pInfo->aOrderBy[0].iColumn == kVec0ColDistance &&
         !pInfo->aOrderBy[0].desc);
    const bool ann_eligible =
        has_query_constraint && k_constraint >= 0 && !has_rowid_range && safe_ann_order;

    int next_argv = 1;
    if (ann_eligible && match_constraint >= 0) {
        pInfo->idxNum |= kVec0PlanAnnMatch;
        pInfo->aConstraintUsage[match_constraint].argvIndex = next_argv++;
        pInfo->aConstraintUsage[match_constraint].omit = 1;
        if (query_constraint >= 0) {
            pInfo->aConstraintUsage[query_constraint].omit = 1;
        }
    } else if (ann_eligible && query_constraint >= 0) {
        pInfo->idxNum |= kVec0PlanAnnHiddenQuery;
        pInfo->aConstraintUsage[query_constraint].argvIndex = next_argv++;
        pInfo->aConstraintUsage[query_constraint].omit = 1;
    } else if (match_constraint >= 0) {
        pInfo->idxNum |= kVec0PlanExactMatch;
        pInfo->aConstraintUsage[match_constraint].argvIndex = next_argv++;
        pInfo->aConstraintUsage[match_constraint].omit = 1;
        if (query_constraint >= 0) {
            pInfo->aConstraintUsage[query_constraint].omit = 1;
        }
    } else if (query_constraint >= 0) {
        pInfo->idxNum |= kVec0PlanExactHiddenQuery;
        pInfo->aConstraintUsage[query_constraint].argvIndex = next_argv++;
        pInfo->aConstraintUsage[query_constraint].omit = 1;
    }

    if ((pInfo->idxNum & (kVec0PlanAnnMatch | kVec0PlanAnnHiddenQuery | kVec0PlanExactMatch |
                          kVec0PlanExactHiddenQuery)) != 0) {
        if (k_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasK;
            pInfo->aConstraintUsage[k_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[k_constraint].omit = 1;
        }
        if (ef_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasEfSearch;
            pInfo->aConstraintUsage[ef_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[ef_constraint].omit = 1;
        }
        if (phss_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasPhss;
            pInfo->aConstraintUsage[phss_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[phss_constraint].omit = 1;
        }
        if (phss_candidates_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasPhssCandidates;
            pInfo->aConstraintUsage[phss_candidates_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[phss_candidates_constraint].omit = 1;
        }
        if (rowid_eq_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasRowidEq;
            pInfo->aConstraintUsage[rowid_eq_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[rowid_eq_constraint].omit = 1;
        }
        if (rowid_in_constraint >= 0) {
            sqlite3_vtab_in(pInfo, rowid_in_constraint, 1);
            pInfo->idxNum |= kVec0PlanHasRowidIn;
            pInfo->aConstraintUsage[rowid_in_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[rowid_in_constraint].omit = 1;
        }
        if (rowid_gt_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasRowidGt;
            pInfo->aConstraintUsage[rowid_gt_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[rowid_gt_constraint].omit = 1;
        }
        if (rowid_ge_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasRowidGe;
            pInfo->aConstraintUsage[rowid_ge_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[rowid_ge_constraint].omit = 1;
        }
        if (rowid_lt_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasRowidLt;
            pInfo->aConstraintUsage[rowid_lt_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[rowid_lt_constraint].omit = 1;
        }
        if (rowid_le_constraint >= 0) {
            pInfo->idxNum |= kVec0PlanHasRowidLe;
            pInfo->aConstraintUsage[rowid_le_constraint].argvIndex = next_argv++;
            pInfo->aConstraintUsage[rowid_le_constraint].omit = 1;
        }

        bool orders_by_distance =
            (pInfo->nOrderBy == 1 && pInfo->aOrderBy[0].iColumn == kVec0ColDistance &&
             !pInfo->aOrderBy[0].desc);
        const bool ann_plan = (pInfo->idxNum & (kVec0PlanAnnMatch | kVec0PlanAnnHiddenQuery)) != 0;
        pInfo->orderByConsumed = (ann_plan && orders_by_distance) ? 1 : 0;
        pInfo->estimatedCost = ann_plan ? 1000.0 : 100000.0;
        pInfo->estimatedRows =
            static_cast<sqlite3_int64>(k_constraint >= 0 ? kVec0DefaultK : 1000000);
        const bool is_match = (pInfo->idxNum & (kVec0PlanAnnMatch | kVec0PlanExactMatch)) != 0;
        if (ann_plan) {
            pInfo->idxStr = sqlite3_mprintf(is_match ? "ann-match" : "ann-query");
        } else {
            pInfo->idxStr = sqlite3_mprintf(is_match ? "exact-match" : "exact-query");
        }
        pInfo->needToFreeIdxStr = 1;
    } else {
        pInfo->estimatedCost = 1000000.0;
        pInfo->estimatedRows = 1000000;
        pInfo->idxNum = kVec0PlanFullScan;
        pInfo->idxStr = sqlite3_mprintf("fullscan");
        pInfo->needToFreeIdxStr = 1;
    }

    return SQLITE_OK;
}

// xOpen: Create a new cursor
inline int vec0Open(sqlite3_vtab* pVTab, sqlite3_vtab_cursor** ppCursor) {
    auto* table = reinterpret_cast<Vec0Table*>(pVTab);
    auto* cursor = new Vec0Cursor();
    cursor->table = table;
    *ppCursor = &cursor->base;
    return SQLITE_OK;
}

// xClose: Destroy cursor
inline int vec0Close(sqlite3_vtab_cursor* pCursor) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);
    delete cursor;
    return SQLITE_OK;
}

// xFilter: Begin iteration
inline int vec0Filter(sqlite3_vtab_cursor* pCursor, int idxNum, const char* idxStr, int argc,
                      sqlite3_value** argv) {
    (void)idxStr;

    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);
    auto* table = cursor->table;

    cursor->ann_mode = false;
    cursor->ann_results.clear();
    cursor->ann_index.reset();
    cursor->ann_pos = 0;
    cursor->current_distance = 0.0f;

    // Prepare query on shadow table
    if (cursor->stmt) {
        sqlite3_finalize(cursor->stmt);
        cursor->stmt = nullptr;
    }

    if ((idxNum & (kVec0PlanAnnMatch | kVec0PlanAnnHiddenQuery | kVec0PlanExactMatch |
                   kVec0PlanExactHiddenQuery)) != 0) {
        if (argc < 1) {
            cursor->eof = true;
            return vec0_set_error(table, "vec0 vector query missing vector argument",
                                  SQLITE_MISUSE);
        }

        int arg_idx = 0;
        Value query_value(argv[arg_idx++]);
        std::optional<size_t> k;
        size_t ef_search = kVec0DefaultEfSearch;
        Vec0PhssRerankOptions phss_options;

        if ((idxNum & kVec0PlanHasK) != 0) {
            if (arg_idx >= argc) {
                cursor->eof = true;
                return vec0_set_error(table, "vec0 vector query missing k argument", SQLITE_MISUSE);
            }
            k = static_cast<size_t>(std::max<int64_t>(1, sqlite3_value_int64(argv[arg_idx++])));
        }

        if ((idxNum & kVec0PlanHasEfSearch) != 0) {
            if (arg_idx >= argc) {
                cursor->eof = true;
                return vec0_set_error(table, "vec0 ANN query missing ef_search argument",
                                      SQLITE_MISUSE);
            }
            ef_search = std::max<int64_t>(static_cast<int64_t>(k.value_or(kVec0DefaultK)),
                                          sqlite3_value_int64(argv[arg_idx++]));
        } else {
            ef_search = std::max(ef_search, k.value_or(kVec0DefaultK));
        }

        if ((idxNum & kVec0PlanHasPhss) != 0) {
            if (arg_idx >= argc) {
                cursor->eof = true;
                return vec0_set_error(table, "vec0 ANN query missing phss argument", SQLITE_MISUSE);
            }
            phss_options.enabled = sqlite3_value_int(argv[arg_idx++]) != 0;
        }

        if ((idxNum & kVec0PlanHasPhssCandidates) != 0) {
            if (arg_idx >= argc) {
                cursor->eof = true;
                return vec0_set_error(table, "vec0 ANN query missing phss_candidates argument",
                                      SQLITE_MISUSE);
            }
            phss_options.candidates = static_cast<size_t>(
                std::max<int64_t>(static_cast<int64_t>(k.value_or(kVec0DefaultK)),
                                  sqlite3_value_int64(argv[arg_idx++])));
        }

        auto rowid_filter = vec0_extract_rowid_filter(idxNum, argc, argv, arg_idx);
        if (!rowid_filter) {
            cursor->eof = true;
            return vec0_set_error(table, rowid_filter.error().message, SQLITE_ERROR);
        }

        const bool ann_plan = (idxNum & (kVec0PlanAnnMatch | kVec0PlanAnnHiddenQuery)) != 0;
        auto buffered_results =
            ann_plan ? vec0_run_ann_query(table, query_value, *k, ef_search, rowid_filter.value(),
                                          phss_options)
                     : vec0_run_exact_query(table, query_value, k, rowid_filter.value());
        if (!buffered_results) {
            cursor->eof = true;
            return vec0_set_error(table, buffered_results.error().message, SQLITE_ERROR);
        }

        if (ann_plan) {
            auto ann_index = vec0_ensure_ann_index(table);
            if (!ann_index) {
                cursor->eof = true;
                return vec0_set_error(table, ann_index.error().message, SQLITE_ERROR);
            }
            cursor->ann_index = ann_index.value();
        } else {
            cursor->ann_index.reset();
        }

        cursor->ann_mode = true;
        cursor->ann_results = std::move(buffered_results.value());
        cursor->eof = cursor->ann_results.empty();
        if (!cursor->eof) {
            cursor->current_rowid = cursor->ann_results[0].first;
            cursor->current_distance = cursor->ann_results[0].second;
        }

        return SQLITE_OK;
    }

    std::ostringstream sql;
    sql << "SELECT rowid, \"" << table->embedding_column << "\" FROM \"" << table->schema_name
        << "\".\"" << table->table_name << "_vectors\" ORDER BY rowid";

    int rc = sqlite3_prepare_v2(table->db, sql.str().c_str(), -1, &cursor->stmt, nullptr);
    if (rc != SQLITE_OK) {
        cursor->eof = true;
        return rc;
    }

    // Advance to first row
    rc = sqlite3_step(cursor->stmt);
    if (rc == SQLITE_ROW) {
        cursor->current_rowid = sqlite3_column_int64(cursor->stmt, 0);
        cursor->eof = false;
    } else {
        cursor->eof = true;
    }

    return SQLITE_OK;
}

// xNext: Advance to next row
inline int vec0Next(sqlite3_vtab_cursor* pCursor) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);

    if (cursor->ann_mode) {
        if (cursor->eof) {
            return SQLITE_OK;
        }

        ++cursor->ann_pos;
        if (cursor->ann_pos >= cursor->ann_results.size()) {
            cursor->eof = true;
        } else {
            cursor->current_rowid = cursor->ann_results[cursor->ann_pos].first;
            cursor->current_distance = cursor->ann_results[cursor->ann_pos].second;
        }
        return SQLITE_OK;
    }

    if (!cursor->stmt || cursor->eof) {
        return SQLITE_OK;
    }

    int rc = sqlite3_step(cursor->stmt);
    if (rc == SQLITE_ROW) {
        cursor->current_rowid = sqlite3_column_int64(cursor->stmt, 0);
        cursor->eof = false;
    } else {
        cursor->eof = true;
    }

    return SQLITE_OK;
}

// xEof: Check if at end of result set
inline int vec0Eof(sqlite3_vtab_cursor* pCursor) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);
    return cursor->eof ? 1 : 0;
}

// xColumn: Return column value
inline int vec0Column(sqlite3_vtab_cursor* pCursor, sqlite3_context* ctx, int col) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);

    if (cursor->ann_mode) {
        if (cursor->eof) {
            sqlite3_result_null(ctx);
            return SQLITE_OK;
        }

        if (col == kVec0ColRowid) {
            sqlite3_result_int64(ctx, cursor->current_rowid);
            return SQLITE_OK;
        }

        if (col == kVec0ColEmbedding) {
            if (cursor->ann_index) {
                const auto* node =
                    cursor->ann_index->get_node(static_cast<size_t>(cursor->current_rowid));
                if (node && !node->vector.empty()) {
                    sqlite3_result_blob(ctx, node->vector.data(),
                                        static_cast<int>(node->vector.size() * sizeof(float)),
                                        SQLITE_TRANSIENT);
                    return SQLITE_OK;
                }
            }

            sqlite3_stmt* stmt = nullptr;
            std::string sql = "SELECT \"" + cursor->table->embedding_column + "\" FROM " +
                              vec0_vectors_table_name(*cursor->table) + " WHERE rowid=?";
            int rc = sqlite3_prepare_v2(cursor->table->db, sql.c_str(), -1, &stmt, nullptr);
            if (rc != SQLITE_OK || !stmt) {
                sqlite3_result_null(ctx);
                if (stmt) {
                    sqlite3_finalize(stmt);
                }
                return SQLITE_OK;
            }
            sqlite3_bind_int64(stmt, 1, cursor->current_rowid);
            rc = sqlite3_step(stmt);
            if (rc == SQLITE_ROW) {
                const void* blob = sqlite3_column_blob(stmt, 0);
                int bytes = sqlite3_column_bytes(stmt, 0);
                if (blob && bytes > 0) {
                    sqlite3_result_blob(ctx, blob, bytes, SQLITE_TRANSIENT);
                } else {
                    sqlite3_result_null(ctx);
                }
            } else {
                sqlite3_result_null(ctx);
            }
            sqlite3_finalize(stmt);
            return SQLITE_OK;
        }

        if (col == kVec0ColDistance) {
            sqlite3_result_double(ctx, cursor->current_distance);
            return SQLITE_OK;
        }

        sqlite3_result_null(ctx);
        return SQLITE_OK;
    }

    if (!cursor->stmt || cursor->eof) {
        sqlite3_result_null(ctx);
        return SQLITE_OK;
    }

    if (col == kVec0ColRowid) {
        // rowid column
        sqlite3_result_int64(ctx, cursor->current_rowid);
    } else if (col == kVec0ColEmbedding) {
        // embedding column
        const void* blob = sqlite3_column_blob(cursor->stmt, 1);
        int bytes = sqlite3_column_bytes(cursor->stmt, 1);
        if (blob && bytes > 0) {
            sqlite3_result_blob(ctx, blob, bytes, SQLITE_TRANSIENT);
        } else {
            sqlite3_result_null(ctx);
        }
    } else if (col == kVec0ColDistance) {
        sqlite3_result_null(ctx);
    } else {
        sqlite3_result_null(ctx);
    }

    return SQLITE_OK;
}

// xRowid: Return current rowid
inline int vec0Rowid(sqlite3_vtab_cursor* pCursor, sqlite3_int64* pRowid) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);
    *pRowid = cursor->current_rowid;
    return SQLITE_OK;
}

// xUpdate: INSERT, UPDATE, DELETE operations
inline int vec0Update(sqlite3_vtab* pVTab, int argc, sqlite3_value** argv, sqlite3_int64* pRowid) {
    auto* table = reinterpret_cast<Vec0Table*>(pVTab);

    // DELETE: argc==1, argv[0]=old_rowid
    if (argc == 1) {
        sqlite3_stmt* stmt = nullptr;
        std::ostringstream sql;
        sql << "DELETE FROM \"" << table->schema_name << "\".\"" << table->table_name
            << "_vectors\" WHERE rowid=?";
        int rc = sqlite3_prepare_v2(table->db, sql.str().c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            return rc;
        }

        sqlite3_bind_int64(stmt, 1, sqlite3_value_int64(argv[0]));
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc == SQLITE_DONE) {
            vec0_invalidate_ann_index(table);
            return SQLITE_OK;
        }
        return rc;
    }

    // INSERT: argc>1, argv[0]=NULL, argv[1]=new_rowid or NULL
    // UPDATE: argc>1, argv[0]=old_rowid, argv[1]=new_rowid
    if (argc > 1) {
        bool is_insert = (sqlite3_value_type(argv[0]) == SQLITE_NULL);
        int64_t old_rowid = is_insert ? 0 : sqlite3_value_int64(argv[0]);
        bool has_declared_rowid = sqlite3_value_type(argv[2]) != SQLITE_NULL;
        bool has_new_rowid = sqlite3_value_type(argv[1]) != SQLITE_NULL;
        int64_t new_rowid = has_declared_rowid
                                ? sqlite3_value_int64(argv[2])
                                : (has_new_rowid ? sqlite3_value_int64(argv[1]) : old_rowid);

        // argv layout per SQLite vtab spec:
        // - argv[0]: old rowid (or NULL)
        // - argv[1]: new rowid (or NULL)
        // - argv[2..]: column values in declared order
        if (argc < 4) {
            return SQLITE_MISUSE;
        }

        // Our declared schema is: CREATE TABLE x(rowid INTEGER PRIMARY KEY, "embedding")
        // So the embedding column value is argv[3] (argv[2] corresponds to rowid column value).
        sqlite3_value* embedding_val = argv[3];
        const void* blob = sqlite3_value_blob(embedding_val);
        int bytes = sqlite3_value_bytes(embedding_val);

        // If xUpdate doesn't materialize function-returned blobs, parse JSON/text and bind as blob
        std::vector<float> parsed_vec;
        if (sqlite3_value_type(embedding_val) == SQLITE_TEXT) {
            Value value(embedding_val);
            auto parsed = parse_vector_from_value<float>(value);
            if (parsed) {
                if (parsed->size() != table->dimensions) {
                    return vec0_set_error(table, "vec0 dimension mismatch: expected " +
                                                     std::to_string(table->dimensions) + ", got " +
                                                     std::to_string(parsed->size()));
                }
                parsed_vec = std::move(parsed.value());
                blob = parsed_vec.data();
                bytes = static_cast<int>(parsed_vec.size() * sizeof(float));
            } else {
                return vec0_set_error(table, parsed.error().message, SQLITE_MISMATCH);
            }
        } else if (sqlite3_value_type(embedding_val) == SQLITE_BLOB && blob && bytes > 0) {
            if (bytes % static_cast<int>(sizeof(float)) != 0) {
                return vec0_set_error(table,
                                      "vec0 embeddings must be float32 blobs aligned to 4 bytes",
                                      SQLITE_MISMATCH);
            }

            std::size_t dimensions = static_cast<std::size_t>(bytes) / sizeof(float);
            if (dimensions != table->dimensions) {
                return vec0_set_error(table, "vec0 dimension mismatch: expected " +
                                                 std::to_string(table->dimensions) + ", got " +
                                                 std::to_string(dimensions));
            }
        }

        if (is_insert) {
            std::ostringstream sql;
            if (has_declared_rowid || has_new_rowid) {
                sql << "INSERT INTO \"" << table->schema_name << "\".\"" << table->table_name
                    << "_vectors\" (rowid, \"" << table->embedding_column << "\") VALUES (?, ?)";
            } else {
                sql << "INSERT INTO \"" << table->schema_name << "\".\"" << table->table_name
                    << "_vectors\" (\"" << table->embedding_column << "\") VALUES (?)";
            }

            sqlite3_stmt* stmt;
            int rc = sqlite3_prepare_v2(table->db, sql.str().c_str(), -1, &stmt, nullptr);
            if (rc != SQLITE_OK) {
                return rc;
            }

            int bind_idx = 1;
            if (has_declared_rowid || has_new_rowid) {
                sqlite3_bind_int64(stmt, bind_idx++, new_rowid);
            }

            if (blob && bytes > 0) {
                sqlite3_bind_blob(stmt, bind_idx, blob, bytes, SQLITE_TRANSIENT);
            } else {
                sqlite3_bind_null(stmt, bind_idx);
            }

            rc = sqlite3_step(stmt);
            if (rc == SQLITE_DONE) {
                *pRowid = (has_declared_rowid || has_new_rowid)
                              ? new_rowid
                              : sqlite3_last_insert_rowid(table->db);
                rc = SQLITE_OK;
            }

            sqlite3_finalize(stmt);
            if (rc == SQLITE_OK) {
                vec0_invalidate_ann_index(table);
            }
            return rc;
        } else {
            std::ostringstream sql;
            sql << "UPDATE \"" << table->schema_name << "\".\"" << table->table_name
                << "_vectors\" SET rowid=?, \"" << table->embedding_column << "\"=? WHERE rowid=?";

            sqlite3_stmt* stmt;
            int rc = sqlite3_prepare_v2(table->db, sql.str().c_str(), -1, &stmt, nullptr);
            if (rc != SQLITE_OK) {
                return rc;
            }

            sqlite3_bind_int64(stmt, 1, new_rowid);

            if (blob && bytes > 0) {
                sqlite3_bind_blob(stmt, 2, blob, bytes, SQLITE_TRANSIENT);
            } else {
                sqlite3_bind_null(stmt, 2);
            }

            sqlite3_bind_int64(stmt, 3, old_rowid);

            rc = sqlite3_step(stmt);
            if (rc == SQLITE_DONE) {
                *pRowid = new_rowid;
                rc = SQLITE_OK;
            }

            sqlite3_finalize(stmt);
            if (rc == SQLITE_OK) {
                vec0_invalidate_ann_index(table);
            }
            return rc;
        }
    }

    return SQLITE_OK;
}

// Module definition
// Note: sqlite3_module struct size varies by SQLite version:
// - xShadowName added in 3.26.0 (20 fields with iVersion=2)
// - xIntegrity added in 3.44.0 (21 fields with iVersion=3)
// We use designated initializers for forward compatibility
static sqlite3_module vec0_module = {
    /* iVersion      */ 2,
    /* xCreate       */ vec0Create,
    /* xConnect      */ vec0Connect,
    /* xBestIndex    */ vec0BestIndex,
    /* xDisconnect   */ vec0Disconnect,
    /* xDestroy      */ vec0Destroy,
    /* xOpen         */ vec0Open,
    /* xClose        */ vec0Close,
    /* xFilter       */ vec0Filter,
    /* xNext         */ vec0Next,
    /* xEof          */ vec0Eof,
    /* xColumn       */ vec0Column,
    /* xRowid        */ vec0Rowid,
    /* xUpdate       */ vec0Update,
    /* xBegin        */ nullptr,
    /* xSync         */ nullptr,
    /* xCommit       */ nullptr,
    /* xRollback     */ nullptr,
    /* xFindFunction */ nullptr,
    /* xRename       */ nullptr,
    /* xSavepoint    */ nullptr,
    /* xRelease      */ nullptr,
    /* xRollbackTo   */ nullptr,
    /* xShadowName   */ nullptr,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity    */ nullptr,
#endif
};

inline Result<void> register_vec0_module(sqlite3* db) {
    if (!db) {
        return err<void>(Error::invalid_argument("database handle is null"));
    }

    int rc = sqlite3_create_module_v2(db, "vec0", &vec0_module, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec0 module", rc));
    }

    return Result<void>();
}

} // namespace sqlite_vec_cpp::sqlite
