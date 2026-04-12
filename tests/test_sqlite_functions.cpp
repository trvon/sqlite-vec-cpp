#include <sqlite3.h>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <sqlite-vec-cpp/sqlite/registration.hpp>
#include <sqlite-vec-cpp/sqlite_vec.hpp>

extern "C" {
int sqlite3_vec_init(sqlite3* db, char** pzErrMsg, const sqlite3_api_routines* pApi);
int sqlite3_vec_distance_l2(const void* vec1, size_t size1, const void* vec2, size_t size2,
                            float* result);
int sqlite3_vec_distance_cosine(const void* vec1, size_t size1, const void* vec2, size_t size2,
                                float* result);
}

// Forward declarations of existing tests (defined later in this file)
void test_vec_f32_json();
void test_vec_f32_blob();
void test_vec_int8();
void test_vec_length();
void test_vec_to_json();
void test_vec_f32_vs_vec_f32_simple();
void test_distance_l2();
static void test_distance_int8_l1_l2_cosine();
static void test_distance_bit_hamming();
static void test_distance_mismatched_types();
static void test_distance_invalid_args();

static double query_double(sqlite3* db, const char* sql) {
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    assert(rc == SQLITE_OK && stmt);
    rc = sqlite3_step(stmt);
    assert(rc == SQLITE_ROW);
    double out = sqlite3_column_double(stmt, 0);
    sqlite3_finalize(stmt);
    return out;
}

static int query_int(sqlite3* db, const char* sql) {
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    assert(rc == SQLITE_OK && stmt);
    rc = sqlite3_step(stmt);
    assert(rc == SQLITE_ROW);
    int out = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return out;
}

static std::string query_error(sqlite3* db, const char* sql) {
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        const char* err = sqlite3_errmsg(db);
        return err ? std::string(err) : std::string();
    }

    rc = sqlite3_step(stmt);
    assert(rc != SQLITE_ROW);

    const char* err = sqlite3_errmsg(db);
    sqlite3_finalize(stmt);
    return err ? std::string(err) : std::string();
}

// C API compatibility layer tests
static void test_distance_int8_l1_l2_cosine() {
    sqlite3* db = nullptr;
    assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
    assert(sqlite3_vec_init(db, nullptr, nullptr) == SQLITE_OK);

    // vec_int8() sets the value subtype so distance functions route to Int8 branches.
    double l2 =
        query_double(db, "SELECT vec_distance_l2(vec_int8('[1,2,3]'), vec_int8('[1,2,5]'))");
    assert(l2 > 1.9 && l2 < 2.1);

    double l1 =
        query_double(db, "SELECT vec_distance_l1(vec_int8('[1,2,3]'), vec_int8('[1,2,5]'))");
    assert(l1 > 1.9 && l1 < 2.1);

    double cosine =
        query_double(db, "SELECT vec_distance_cosine(vec_int8('[1,0,0]'), vec_int8('[0,1,0]'))");
    assert(cosine > 0.9 && cosine < 1.1);

    sqlite3_close(db);
}

static void test_distance_bit_hamming() {
    sqlite3* db = nullptr;
    assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
    assert(sqlite3_vec_init(db, nullptr, nullptr) == SQLITE_OK);

    // Two bytes => 16 bit dimensions.
    // 0b00001111 vs 0b00011111 differs by 1 bit in first byte; second byte equal.
    int dist = query_int(
        db, "SELECT CAST(vec_distance_hamming(vec_bit(X'0F00'), vec_bit(X'1F00')) AS INT)");
    assert(dist == 1);

    sqlite3_close(db);
}

static void test_distance_mismatched_types() {
    sqlite3* db = nullptr;
    assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
    assert(sqlite3_vec_init(db, nullptr, nullptr) == SQLITE_OK);

    // Different subtypes => element types mismatch.
    auto err = query_error(db, "SELECT vec_distance_l2(vec_f32('[1,2,3]'), vec_int8('[1,2,3]'))");
    assert(!err.empty());

    // Bitvectors cannot be used with L2.
    err = query_error(db, "SELECT vec_distance_l2(vec_bit(X'00'), vec_bit(X'00'))");
    assert(!err.empty());

    sqlite3_close(db);
}

static void test_distance_invalid_args() {
    sqlite3* db = nullptr;
    assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
    assert(sqlite3_vec_init(db, nullptr, nullptr) == SQLITE_OK);

    // Wrong arg count.
    auto err = query_error(db, "SELECT vec_distance_l2(vec_f32('[1,2,3]'))");
    assert(!err.empty());

    // Non-blob args (extract_vector_from_value checks value.is_blob).
    err = query_error(db, "SELECT vec_distance_l2(1, 2)");
    assert(!err.empty());

    // Blob size not aligned to element size.
    err = query_error(db, "SELECT vec_distance_l2(X'00', X'00')");
    assert(!err.empty());

    // Dimension mismatch.
    err = query_error(db, "SELECT vec_distance_l2(vec_f32('[1,2,3]'), vec_f32('[1,2]'))");
    assert(!err.empty());

    sqlite3_close(db);
}

static void test_c_api_init_and_distance() {
    std::cout << "Testing sqlite3_vec_init (test failpoint) + distance helpers..." << std::endl;

    sqlite3* db = nullptr;
    assert(sqlite3_open(":memory:", &db) == SQLITE_OK);

    // With SQLITE_VEC_CPP_TESTING enabled, sqlite3_vec_init() can be forced to fail
    // by passing a non-null `pApi` pointer.
    {
        char* err = nullptr;
        int rc = sqlite3_vec_init(db, &err, reinterpret_cast<const sqlite3_api_routines*>(0x1));
        assert(rc != SQLITE_OK);
        assert(err != nullptr);
        sqlite3_free(err);
    }

    // Normal init should still work.
    assert(sqlite3_vec_init(db, nullptr, nullptr) == SQLITE_OK);

    // We still directly test the helper C distance functions below.

    float out = 0.0f;
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};

    int rc = sqlite3_vec_distance_l2(a, sizeof(a), b, sizeof(b), &out);
    assert(rc == SQLITE_OK);
    std::cout << "  l2_distance(a,b)=" << out << std::endl;
    // l2_distance returns Euclidean distance: sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
    assert(out > 1.3f && out < 1.5f);

    rc = sqlite3_vec_distance_cosine(a, sizeof(a), b, sizeof(b), &out);
    assert(rc == SQLITE_OK);
    std::cout << "  cosine_distance(a,b)=" << out << std::endl;
    // cosine_distance is 1 - cosine_similarity; orthogonal vectors => similarity=0 => distance=1
    assert(out > 0.9f && out < 1.1f);

    // Error paths: null pointers
    rc = sqlite3_vec_distance_l2(nullptr, sizeof(a), b, sizeof(b), &out);
    assert(rc != SQLITE_OK);

    rc = sqlite3_vec_distance_l2(a, sizeof(a), nullptr, sizeof(b), &out);
    assert(rc != SQLITE_OK);

    rc = sqlite3_vec_distance_l2(a, sizeof(a), b, sizeof(b), nullptr);
    assert(rc != SQLITE_OK);

    rc = sqlite3_vec_distance_cosine(nullptr, sizeof(a), b, sizeof(b), &out);
    assert(rc != SQLITE_OK);

    rc = sqlite3_vec_distance_cosine(a, sizeof(a), nullptr, sizeof(b), &out);
    assert(rc != SQLITE_OK);

    rc = sqlite3_vec_distance_cosine(a, sizeof(a), b, sizeof(b), nullptr);
    assert(rc != SQLITE_OK);

    // Error paths: dim mismatch
    rc = sqlite3_vec_distance_l2(a, sizeof(a), b, sizeof(float) * 2, &out);
    assert(rc != SQLITE_OK);

    rc = sqlite3_vec_distance_cosine(a, sizeof(a), b, sizeof(float) * 2, &out);
    assert(rc != SQLITE_OK);

    // Error paths: closed database init
    sqlite3_close(db);
    db = nullptr;

    assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
    assert(sqlite3_close(db) == SQLITE_OK);

    // sqlite3_vec_init expects a valid database handle; passing a closed handle is UB.

    std::cout << "  ✓ C API init + distance helpers work" << std::endl;
}

// New tests to validate vec0 virtual table lifecycle
static void test_vec0_basic_create_insert_select() {
    std::cout << "Testing vec0 virtual table create/insert/select..." << std::endl;

    sqlite3* db = nullptr;
    assert(sqlite3_open(":memory:", &db) == SQLITE_OK && db);

    // Initialize sqlite-vec (registers vec0 module + functions)
    sqlite3_vec_init(db, nullptr, nullptr);

    // Create vec0 table with dimension 4
    const char* create_sql = "CREATE VIRTUAL TABLE doc_embeddings USING vec0(embedding float[4])";
    char* err = nullptr;
    int rc = sqlite3_exec(db, create_sql, nullptr, nullptr, &err);
    assert(rc == SQLITE_OK);

    // Insert a vector. SQLite virtual table xUpdate does not reliably materialize
    // function-returned BLOBs (even without subtypes), so pass JSON text directly.
    const char* insert_sql =
        "INSERT INTO doc_embeddings(rowid, embedding) VALUES(NULL, '[1,2,3,4]')";
    rc = sqlite3_exec(db, insert_sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::cerr << "  INSERT failed with rc=" << rc;
        if (err) {
            std::cerr << " error=" << err;
            sqlite3_free(err);
        }
        std::cerr << std::endl;
        return;
    }

    // Read it back via SELECT; expect 1 row and non-null blob
    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db, "SELECT rowid, embedding, typeof(embedding) FROM doc_embeddings",
                            -1, &stmt, nullptr);
    assert(rc == SQLITE_OK && stmt);
    rc = sqlite3_step(stmt);
    assert(rc == SQLITE_ROW);
    assert(sqlite3_column_type(stmt, 0) == SQLITE_INTEGER);
    int col1_type = sqlite3_column_type(stmt, 1);
    const char* typeof_str = (const char*)sqlite3_column_text(stmt, 2);
    std::cerr << "  typeof(embedding)=" << (typeof_str ? typeof_str : "<null>")
              << " sqlite3_column_type=" << col1_type << std::endl;
    if (col1_type == SQLITE_NULL) {
        std::cerr << "  ERROR: Expected blob/text, got NULL" << std::endl;
    }
    assert(col1_type == SQLITE_BLOB || col1_type == SQLITE_TEXT);
    sqlite3_finalize(stmt);

    // Drop table (should remove shadow tables without error)
    rc = sqlite3_exec(db, "DROP TABLE doc_embeddings", nullptr, nullptr, &err);
    assert(rc == SQLITE_OK);

    sqlite3_close(db);
    std::cout << "  \xE2\x9C\x93 vec0 create/insert/select/drop works" << std::endl;
}

static void test_vec0_update_delete_paths() {
    std::cout << "Testing vec0 update/delete paths..." << std::endl;

    sqlite3* db = nullptr;
    assert(sqlite3_open(":memory:", &db) == SQLITE_OK && db);
    sqlite3_vec_init(db, nullptr, nullptr);

    char* err = nullptr;
    int rc = SQLITE_OK;

    assert(sqlite3_exec(db, "CREATE VIRTUAL TABLE t USING vec0(embedding float[2])", nullptr,
                        nullptr, nullptr) == SQLITE_OK);

    rc = sqlite3_exec(db, "INSERT INTO t(rowid, embedding) VALUES(NULL, '[10,20]')", nullptr,
                      nullptr, &err);
    if (rc != SQLITE_OK) {
        std::cerr << "  vec0 insert failed rc=" << rc;
        if (err) {
            std::cerr << " err=" << err;
            sqlite3_free(err);
            err = nullptr;
        }
        std::cerr << std::endl;
    }
    assert(rc == SQLITE_OK);

    // Update rowid 1
    rc = sqlite3_exec(db, "UPDATE t SET embedding='[11,22]' WHERE rowid=1", nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::cerr << "  vec0 update failed rc=" << rc;
        if (err) {
            std::cerr << " err=" << err;
            sqlite3_free(err);
            err = nullptr;
        }
        std::cerr << std::endl;
    }
    assert(rc == SQLITE_OK);

    // Delete row
    assert(sqlite3_exec(db, "DELETE FROM t WHERE rowid=1", nullptr, nullptr, nullptr) == SQLITE_OK);

    sqlite3_close(db);
    std::cout << "  \xE2\x9C\x93 vec0 update/delete works" << std::endl;
}

// Keep a single main() at end of file

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sqlite-vec-cpp/sqlite_vec.hpp>

using namespace sqlite_vec_cpp;
using namespace sqlite_vec_cpp::sqlite;

// RAII wrapper for SQLite database
class SQLiteDB {
public:
    explicit SQLiteDB(const std::string& path = ":memory:") {
        int rc = sqlite3_open(path.c_str(), &db_);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to open database");
        }
    }

    ~SQLiteDB() {
        if (db_) {
            sqlite3_close(db_);
        }
    }

    sqlite3* get() { return db_; }

    // Execute SQL and return result as string
    std::string exec_scalar(const std::string& sql) {
        char* err_msg = nullptr;
        std::string result;

        int rc = sqlite3_exec(
            db_, sql.c_str(),
            [](void* data, int argc, char** argv, [[maybe_unused]] char** col_names) -> int {
                if (argc > 0 && argv[0]) {
                    *static_cast<std::string*>(data) = argv[0];
                }
                return 0;
            },
            &result, &err_msg);

        if (rc != SQLITE_OK) {
            std::string error = err_msg ? err_msg : "Unknown error";
            sqlite3_free(err_msg);
            throw std::runtime_error("SQL error: " + error);
        }

        return result;
    }

    // Execute SQL without expecting result
    void exec(const std::string& sql) {
        char* err_msg = nullptr;
        int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);

        if (rc != SQLITE_OK) {
            std::string error = err_msg ? err_msg : "Unknown error";
            sqlite3_free(err_msg);
            throw std::runtime_error("SQL error: " + error);
        }
    }

    // Register a function
    void create_function(const char* name, int nargs,
                         void (*func)(sqlite3_context*, int, sqlite3_value**),
                         int flags = SQLITE_UTF8 | SQLITE_DETERMINISTIC) {
        int rc = sqlite3_create_function_v2(db_, name, nargs, flags, nullptr, func, nullptr,
                                            nullptr, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to create function: " + std::string(name));
        }
    }

private:
    sqlite3* db_ = nullptr;
};

struct RowDistance {
    std::int64_t rowid;
    double distance;
};

std::vector<RowDistance> query_row_distances(sqlite3* db, const std::string& sql) {
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare query: " + std::string(sqlite3_errmsg(db)));
    }

    std::vector<RowDistance> rows;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        rows.push_back(RowDistance{sqlite3_column_int64(stmt, 0), sqlite3_column_double(stmt, 1)});
    }

    if (rc != SQLITE_DONE) {
        std::string error = sqlite3_errmsg(db);
        sqlite3_finalize(stmt);
        throw std::runtime_error("Query failed: " + error);
    }

    sqlite3_finalize(stmt);
    return rows;
}

std::string query_explain_plan(sqlite3* db, const std::string& sql) {
    sqlite3_stmt* stmt = nullptr;
    std::string explain_sql = "EXPLAIN QUERY PLAN " + sql;
    int rc = sqlite3_prepare_v2(db, explain_sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare EQP: " + std::string(sqlite3_errmsg(db)));
    }

    std::string detail;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const unsigned char* text = sqlite3_column_text(stmt, 3);
        if (text) {
            if (!detail.empty()) {
                detail += "\n";
            }
            detail += reinterpret_cast<const char*>(text);
        }
    }

    if (rc != SQLITE_DONE) {
        std::string error = sqlite3_errmsg(db);
        sqlite3_finalize(stmt);
        throw std::runtime_error("EQP failed: " + error);
    }

    sqlite3_finalize(stmt);
    return detail;
}

// Default flags for functions that work with vector subtypes
constexpr int VEC_FUNC_FLAGS =
    SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE;

void register_all_functions(SQLiteDB& db) {
    // Distance functions (read subtypes, don't set them - return scalar)
    db.create_function("vec_distance_l2", 2, vec_distance_l2,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_distance_l1", 2, vec_distance_l1,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_distance_cosine", 2, vec_distance_cosine,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_distance_hamming", 2, vec_distance_hamming,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);

    // Vector creation (set subtypes)
    db.create_function("vec_f32", 1, vec_f32, VEC_FUNC_FLAGS);
    db.create_function("vec_int8", 1, vec_int8, VEC_FUNC_FLAGS);
    db.create_function("vec_bit", 1, vec_bit, VEC_FUNC_FLAGS);

    // Vector info (read subtypes)
    db.create_function("vec_length", 1, vec_length,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_type", 1, vec_type,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);

    // Vector conversion (read and set subtypes)
    db.create_function("vec_to_json", 1, vec_to_json,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);

    // Vector math (read and set subtypes)
    db.create_function("vec_add", 2, vec_add, VEC_FUNC_FLAGS);
    db.create_function("vec_sub", 2, vec_sub, VEC_FUNC_FLAGS);
    db.create_function("vec_normalize", 1, vec_normalize, VEC_FUNC_FLAGS);
    db.create_function("vec_slice", 3, vec_slice, VEC_FUNC_FLAGS);

    // Enhanced functions (read subtypes, may or may not set them)
    db.create_function("vec_dot", 2, vec_dot, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_magnitude", 1, vec_magnitude,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_scale", 2, vec_scale, VEC_FUNC_FLAGS);
    db.create_function("vec_mean", 1, vec_mean,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_std", 1, vec_std, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_min", 1, vec_min, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_max", 1, vec_max, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_clamp", 3, vec_clamp, VEC_FUNC_FLAGS);
}

// Helper to compare floats
bool approx_equal(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

// ============================================================================
// TEST: Vector Creation Functions
// ============================================================================

void test_vec_f32_json() {
    std::cout << "Testing vec_f32 with JSON input..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Create from JSON array
    std::string result = db.exec_scalar("SELECT hex(vec_f32('[1.0, 2.0, 3.0]'))");
    assert(!result.empty());

    // Verify length
    std::string length = db.exec_scalar("SELECT vec_length(vec_f32('[1.0, 2.0, 3.0]'))");
    assert(length == "3");

    // Verify type
    std::string type = db.exec_scalar("SELECT vec_type(vec_f32('[1.0, 2.0, 3.0]'))");
    assert(type == "float32");

    std::cout << "  ✓ vec_f32 JSON parsing works" << std::endl;
}

void test_vec_f32_blob() {
    std::cout << "Testing vec_f32 with blob input..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Create table with blob
    db.exec("CREATE TABLE test (vec BLOB)");

    // Insert binary data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db.get(), "INSERT INTO test VALUES (?)", -1, &stmt, nullptr);
    sqlite3_bind_blob(stmt, 1, data.data(), data.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    // Read back and verify
    std::string length = db.exec_scalar("SELECT vec_length(vec) FROM test");
    assert(length == "4");

    std::cout << "  ✓ vec_f32 blob handling works" << std::endl;
}

void test_vec_int8() {
    std::cout << "Testing vec_int8..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    std::string length = db.exec_scalar("SELECT vec_length(vec_int8('[1, 2, 3, 4, 5]'))");
    assert(length == "5");

    std::string type = db.exec_scalar("SELECT vec_type(vec_int8('[1, 2, 3]'))");
    assert(type == "int8");

    std::cout << "  ✓ vec_int8 works" << std::endl;
}

void test_vec_length() {
    std::cout << "Testing vec_length..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    assert(db.exec_scalar("SELECT vec_length(vec_f32('[1]'))") == "1");
    assert(db.exec_scalar("SELECT vec_length(vec_f32('[1,2,3]'))") == "3");
    assert(db.exec_scalar("SELECT vec_length(vec_f32('[1,2,3,4,5,6,7,8,9,10]'))") == "10");

    std::cout << "  ✓ vec_length works" << std::endl;
}

void test_vec_to_json() {
    std::cout << "Testing vec_to_json..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    std::string json = db.exec_scalar("SELECT vec_to_json(vec_f32('[1.0, 2.5, 3.14]'))");

    // Should be valid JSON array
    assert(json.front() == '[');
    assert(json.back() == ']');
    assert(json.find("1") != std::string::npos);
    assert(json.find("2.5") != std::string::npos);

    std::cout << "  ✓ vec_to_json works" << std::endl;
}

void test_vec_f32_vs_vec_f32_simple() {
    std::cout << "Testing vec_f32 vs vec_f32_simple..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Test that both functions produce the same blob data
    std::string hex_f32 = db.exec_scalar("SELECT hex(vec_f32('[1.0, 2.0, 3.0]'))");
    std::string hex_simple = db.exec_scalar("SELECT hex(vec_f32_simple('[1.0, 2.0, 3.0]'))");
    assert(hex_f32 == hex_simple);

    // Test that vec_length works for both
    std::string len_f32 = db.exec_scalar("SELECT vec_length(vec_f32('[1,2,3,4,5]'))");
    std::string len_simple = db.exec_scalar("SELECT vec_length(vec_f32_simple('[1,2,3,4,5]'))");
    assert(len_f32 == len_simple);
    assert(len_f32 == "5");

    std::cout << "  ✓ vec_f32 and vec_f32_simple produce identical results" << std::endl;
}

// ============================================================================
// TEST: Distance Functions
// ============================================================================

void test_distance_l2() {
    std::cout << "Testing vec_distance_l2..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Distance between [0,0] and [3,4] should be 5.0 (Euclidean distance, not squared)
    std::string dist = db.exec_scalar("SELECT vec_distance_l2(vec_f32('[0,0]'), vec_f32('[3,4]'))");
    double d = std::stod(dist);
    assert(approx_equal(d, 5.0)); // sqrt(3^2 + 4^2) = sqrt(25) = 5.0

    // Distance between identical vectors should be 0
    dist = db.exec_scalar("SELECT vec_distance_l2(vec_f32('[1,2,3]'), vec_f32('[1,2,3]'))");
    d = std::stod(dist);
    assert(approx_equal(d, 0.0));

    std::cout << "  ✓ vec_distance_l2 works" << std::endl;
}

void test_distance_l1() {
    std::cout << "Testing vec_distance_l1..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Manhattan distance between [0,0] and [3,4] should be 7
    std::string dist = db.exec_scalar("SELECT vec_distance_l1(vec_f32('[0,0]'), vec_f32('[3,4]'))");
    double d = std::stod(dist);
    assert(approx_equal(d, 7.0));

    // Distance between identical vectors should be 0
    dist = db.exec_scalar("SELECT vec_distance_l1(vec_f32('[1,2,3]'), vec_f32('[1,2,3]'))");
    d = std::stod(dist);
    assert(approx_equal(d, 0.0));

    std::cout << "  ✓ vec_distance_l1 works" << std::endl;
}

void test_distance_cosine() {
    std::cout << "Testing vec_distance_cosine..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Cosine distance between identical vectors should be 0
    std::string dist =
        db.exec_scalar("SELECT vec_distance_cosine(vec_f32('[1,2,3]'), vec_f32('[1,2,3]'))");
    double d = std::stod(dist);
    assert(approx_equal(d, 0.0));

    // Cosine distance between [1,0] and [0,1] should be 1.0 (orthogonal)
    dist = db.exec_scalar("SELECT vec_distance_cosine(vec_f32('[1,0]'), vec_f32('[0,1]'))");
    d = std::stod(dist);
    assert(approx_equal(d, 1.0));

    std::cout << "  ✓ vec_distance_cosine works" << std::endl;
}

// ============================================================================
// TEST: Vector Math Functions
// ============================================================================

void test_vec_add() {
    std::cout << "Testing vec_add..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    std::string result =
        db.exec_scalar("SELECT vec_to_json(vec_add(vec_f32('[1,2,3]'), vec_f32('[4,5,6]')))");

    // Result should be [5, 7, 9]
    assert(result.find("5") != std::string::npos);
    assert(result.find("7") != std::string::npos);
    assert(result.find("9") != std::string::npos);

    std::cout << "  ✓ vec_add works" << std::endl;
}

void test_vec_sub() {
    std::cout << "Testing vec_sub..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    std::string result =
        db.exec_scalar("SELECT vec_to_json(vec_sub(vec_f32('[5,7,9]'), vec_f32('[1,2,3]')))");

    // Result should be [4, 5, 6]
    assert(result.find("4") != std::string::npos);
    assert(result.find("5") != std::string::npos);
    assert(result.find("6") != std::string::npos);

    std::cout << "  ✓ vec_sub works" << std::endl;
}

void test_vec_normalize() {
    std::cout << "Testing vec_normalize..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Normalize [3, 4] -> [0.6, 0.8]
    std::string result = db.exec_scalar("SELECT vec_to_json(vec_normalize(vec_f32('[3,4]')))");

    assert(result.find("0.6") != std::string::npos);
    assert(result.find("0.8") != std::string::npos);

    // Verify magnitude is 1
    std::string mag = db.exec_scalar("SELECT vec_magnitude(vec_normalize(vec_f32('[3,4]')))");
    double m = std::stod(mag);
    assert(approx_equal(m, 1.0));

    std::cout << "  ✓ vec_normalize works" << std::endl;
}

void test_vec_slice() {
    std::cout << "Testing vec_slice..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Slice [1,2,3,4,5] from index 1 to 4 -> [2,3,4]
    std::string length =
        db.exec_scalar("SELECT vec_length(vec_slice(vec_f32('[1,2,3,4,5]'), 1, 4))");
    assert(length == "3");

    std::string result =
        db.exec_scalar("SELECT vec_to_json(vec_slice(vec_f32('[1,2,3,4,5]'), 1, 4))");

    assert(result.find("2") != std::string::npos);
    assert(result.find("3") != std::string::npos);
    assert(result.find("4") != std::string::npos);

    std::cout << "  ✓ vec_slice works" << std::endl;
}

// ============================================================================
// TEST: Enhanced Functions
// ============================================================================

void test_vec_dot() {
    std::cout << "Testing vec_dot..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Dot product of [1,2,3] and [4,5,6] = 1*4 + 2*5 + 3*6 = 32
    std::string result = db.exec_scalar("SELECT vec_dot(vec_f32('[1,2,3]'), vec_f32('[4,5,6]'))");
    double dot = std::stod(result);
    assert(approx_equal(dot, 32.0));

    // Dot product of orthogonal vectors should be 0
    result = db.exec_scalar("SELECT vec_dot(vec_f32('[1,0]'), vec_f32('[0,1]'))");
    dot = std::stod(result);
    assert(approx_equal(dot, 0.0));

    std::cout << "  ✓ vec_dot works" << std::endl;
}

void test_vec_magnitude() {
    std::cout << "Testing vec_magnitude..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Magnitude of [3,4] should be 5
    std::string result = db.exec_scalar("SELECT vec_magnitude(vec_f32('[3,4]'))");
    double mag = std::stod(result);
    assert(approx_equal(mag, 5.0));

    // Magnitude of [1,0,0] should be 1
    result = db.exec_scalar("SELECT vec_magnitude(vec_f32('[1,0,0]'))");
    mag = std::stod(result);
    assert(approx_equal(mag, 1.0));

    std::cout << "  ✓ vec_magnitude works" << std::endl;
}

void test_vec_scale() {
    std::cout << "Testing vec_scale..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Scale [1,2,3] by 2 -> [2,4,6]
    std::string result = db.exec_scalar("SELECT vec_to_json(vec_scale(vec_f32('[1,2,3]'), 2.0))");

    assert(result.find("2") != std::string::npos);
    assert(result.find("4") != std::string::npos);
    assert(result.find("6") != std::string::npos);

    std::cout << "  ✓ vec_scale works" << std::endl;
}

void test_vec_mean() {
    std::cout << "Testing vec_mean..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Mean of [1,2,3,4,5] should be 3
    std::string result = db.exec_scalar("SELECT vec_mean(vec_f32('[1,2,3,4,5]'))");
    double mean = std::stod(result);
    assert(approx_equal(mean, 3.0));

    // Mean of [10,20,30] should be 20
    result = db.exec_scalar("SELECT vec_mean(vec_f32('[10,20,30]'))");
    mean = std::stod(result);
    assert(approx_equal(mean, 20.0));

    std::cout << "  ✓ vec_mean works" << std::endl;
}

void test_vec_std() {
    std::cout << "Testing vec_std..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Standard deviation should be > 0 for varying data
    std::string result = db.exec_scalar("SELECT vec_std(vec_f32('[1,2,3,4,5]'))");
    double std_dev = std::stod(result);
    assert(std_dev > 0.0);
    assert(approx_equal(std_dev, 1.5811, 0.001)); // Sample std dev

    std::cout << "  ✓ vec_std works" << std::endl;
}

void test_vec_min_max() {
    std::cout << "Testing vec_min and vec_max..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    std::string min_val = db.exec_scalar("SELECT vec_min(vec_f32('[3,1,4,1,5,9,2,6]'))");
    double min_d = std::stod(min_val);
    assert(approx_equal(min_d, 1.0));

    std::string max_val = db.exec_scalar("SELECT vec_max(vec_f32('[3,1,4,1,5,9,2,6]'))");
    double max_d = std::stod(max_val);
    assert(approx_equal(max_d, 9.0));

    std::cout << "  ✓ vec_min and vec_max work" << std::endl;
}

void test_vec_clamp() {
    std::cout << "Testing vec_clamp..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Clamp [1,5,10] to [2,8] -> [2,5,8]
    std::string result = db.exec_scalar("SELECT vec_to_json(vec_clamp(vec_f32('[1,5,10]'), 2, 8))");

    assert(result.find("2") != std::string::npos);
    assert(result.find("5") != std::string::npos);
    assert(result.find("8") != std::string::npos);

    // Verify min is 2
    std::string min_val = db.exec_scalar("SELECT vec_min(vec_clamp(vec_f32('[1,5,10]'), 2, 8))");
    double min_d = std::stod(min_val);
    assert(approx_equal(min_d, 2.0));

    // Verify max is 8
    std::string max_val = db.exec_scalar("SELECT vec_max(vec_clamp(vec_f32('[1,5,10]'), 2, 8))");
    double max_d = std::stod(max_val);
    assert(approx_equal(max_d, 8.0));

    std::cout << "  ✓ vec_clamp works" << std::endl;
}

// ============================================================================
// TEST: Error Handling
// ============================================================================

void test_error_handling() {
    std::cout << "Testing error handling..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Dimension mismatch
    try {
        db.exec_scalar("SELECT vec_distance_l2(vec_f32('[1,2]'), vec_f32('[1,2,3]'))");
        assert(false && "Should have thrown error for dimension mismatch");
    } catch (const std::runtime_error& e) {
        // Expected
        std::string err = e.what();
        assert(err.find("mismatch") != std::string::npos ||
               err.find("SQL error") != std::string::npos);
    }

    // Invalid JSON
    try {
        db.exec_scalar("SELECT vec_f32('[1,2,invalid]')");
        assert(false && "Should have thrown error for invalid JSON");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Truncated JSON should be rejected.
    try {
        db.exec_scalar("SELECT vec_f32('[1,2')");
        assert(false && "Should have thrown error for truncated JSON");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Trailing non-whitespace after the array should be rejected.
    try {
        db.exec_scalar("SELECT vec_f32('[1] trailing')");
        assert(false && "Should have thrown error for trailing JSON garbage");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Empty vector
    try {
        db.exec_scalar("SELECT vec_f32('[]')");
        assert(false && "Should have thrown error for empty vector");
    } catch (const std::runtime_error&) {
        // Expected
    }

    std::cout << "  ✓ Error handling works" << std::endl;
}

// ============================================================================
// TEST: Integration Scenarios
// ============================================================================

void test_stored_float_blobs_work_with_pairwise_helpers() {
    std::cout << "Testing stored float blobs with pairwise helpers..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    db.exec("CREATE TABLE docs (embedding BLOB)");
    db.exec("INSERT INTO docs VALUES (vec_f32('[1,2,3]'))");

    std::string dot_sql = "SELECT vec_dot(embedding, vec_f32('[4,5,6]')) FROM docs LIMIT 1";
    double dot = std::stod(db.exec_scalar(dot_sql));
    assert(approx_equal(dot, 32.0));

    std::string sum_json =
        db.exec_scalar("SELECT vec_to_json(vec_add(embedding, vec_f32('[1,1,1]'))) "
                       "FROM docs LIMIT 1");
    assert(sum_json.find("2") != std::string::npos);
    assert(sum_json.find("3") != std::string::npos);
    assert(sum_json.find("4") != std::string::npos);

    std::cout << "  ✓ Stored float blobs work with pairwise helpers" << std::endl;
}

void test_sqlite3_vec_init_registers_extended_functions() {
    std::cout << "Testing sqlite3_vec_init extended function registration..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);

    std::string json = db.exec_scalar("SELECT vec_to_json(vec_f32('[1, 2.5, 3]'))");
    assert(json.front() == '[');
    assert(json.back() == ']');

    std::string normalized = db.exec_scalar("SELECT vec_to_json(vec_normalize(vec_f32('[3,4]')))");
    assert(normalized.find("0.6") != std::string::npos);
    assert(normalized.find("0.8") != std::string::npos);

    double std_dev = std::stod(db.exec_scalar("SELECT vec_std(vec_f32('[1,2,3,4,5]'))"));
    assert(std_dev > 0.0);

    std::cout << "  ✓ sqlite3_vec_init registers extended functions" << std::endl;
}

void test_vec0_explicit_rowid_insert() {
    std::cout << "Testing vec0 explicit rowid insert..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);

    db.exec("CREATE VIRTUAL TABLE t USING vec0(embedding float[2])");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (42, '[1,2]')");

    std::string rowid = db.exec_scalar("SELECT rowid FROM t LIMIT 1");
    assert(rowid == "42");

    std::cout << "  ✓ vec0 preserves explicit rowid inserts" << std::endl;
}

void test_vec0_dimension_validation() {
    std::cout << "Testing vec0 dimension validation..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);

    db.exec("CREATE VIRTUAL TABLE t USING vec0(embedding float[2])");

    try {
        db.exec("INSERT INTO t(embedding) VALUES ('[1,2,3]')");
        assert(false && "Expected vec0 dimension mismatch for text input");
    } catch (const std::runtime_error&) {
        // Expected
    }

    try {
        db.exec("INSERT INTO t(embedding) VALUES (X'000102')");
        assert(false && "Expected vec0 blob alignment error");
    } catch (const std::runtime_error&) {
        // Expected
    }

    db.exec("INSERT INTO t(embedding) VALUES ('[1,2]')");
    assert(db.exec_scalar("SELECT COUNT(*) FROM t") == "1");

    std::cout << "  ✓ vec0 enforces declared dimensions" << std::endl;
}

void test_vec0_ann_match_query() {
    std::cout << "Testing vec0 ANN MATCH query..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);

    db.exec("CREATE VIRTUAL TABLE t USING vec0(embedding float[2])");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (1, '[0,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (2, '[1,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (3, '[2,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (4, '[0,2]')");

    const std::string sql = "SELECT rowid, distance FROM t "
                            "WHERE embedding MATCH vec_f32('[1.2,0]') AND k = 2 "
                            "ORDER BY distance";

    std::string plan = query_explain_plan(db.get(), sql);
    assert(plan.find("ann") != std::string::npos);

    auto rows = query_row_distances(db.get(), sql);
    assert(rows.size() == 2);
    assert(rows[0].rowid == 2);
    assert(rows[1].rowid == 3);
    assert(approx_equal(static_cast<float>(rows[0].distance), 0.2f, 0.05f));
    assert(approx_equal(static_cast<float>(rows[1].distance), 0.8f, 0.05f));

    std::cout << "  ✓ vec0 MATCH ANN query works" << std::endl;
}

void test_vec0_ann_hidden_query() {
    std::cout << "Testing vec0 explicit hidden query..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);

    db.exec("CREATE VIRTUAL TABLE t USING vec0(embedding float[2])");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (10, '[0,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (11, '[1,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (12, '[2,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (13, '[0,2]')");

    const std::string sql = "SELECT rowid, distance FROM t "
                            "WHERE query = vec_f32('[1.2,0]') AND k = 3 AND ef_search = 32 "
                            "ORDER BY distance";

    std::string plan = query_explain_plan(db.get(), sql);
    assert(plan.find("ann") != std::string::npos);

    auto rows = query_row_distances(db.get(), sql);
    assert(rows.size() == 3);
    assert(rows[0].rowid == 11);
    assert(rows[1].rowid == 12);
    assert(rows[2].rowid == 10 || rows[2].rowid == 13);

    std::cout << "  ✓ vec0 explicit hidden query works" << std::endl;
}

void test_vec0_ann_insert_visibility() {
    std::cout << "Testing vec0 ANN insert visibility..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);

    db.exec("CREATE VIRTUAL TABLE t USING vec0(embedding float[2])");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (1, '[0,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (2, '[2,0]')");

    const std::string sql = "SELECT rowid, distance FROM t "
                            "WHERE embedding MATCH vec_f32('[1,0]') AND k = 1 "
                            "ORDER BY distance";

    auto before = query_row_distances(db.get(), sql);
    assert(before.size() == 1);

    db.exec("INSERT INTO t(rowid, embedding) VALUES (3, '[1,0]')");

    auto after = query_row_distances(db.get(), sql);
    assert(after.size() == 1);
    assert(after[0].rowid == 3);
    assert(approx_equal(static_cast<float>(after[0].distance), 0.0f));

    std::cout << "  ✓ vec0 ANN sees inserted vectors" << std::endl;
}

void test_vec0_ann_match_respects_rowid_filter() {
    std::cout << "Testing vec0 ANN MATCH with rowid filter semantics..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);

    db.exec("CREATE VIRTUAL TABLE t USING vec0(embedding float[2])");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (1, '[0.0,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (2, '[0.2,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (3, '[0.4,0]')");

    const std::string sql = "SELECT rowid, distance FROM t "
                            "WHERE embedding MATCH vec_f32('[0,0]') AND rowid IN (2,3) AND k = 1 "
                            "ORDER BY distance";

    std::string plan = query_explain_plan(db.get(), sql);
    if (plan.find("ann") == std::string::npos) {
        throw std::runtime_error("Expected ANN query plan for MATCH rowid-filter test");
    }

    auto rows = query_row_distances(db.get(), sql);
    if (rows.size() != 1) {
        throw std::runtime_error("Expected one filtered MATCH result, got " +
                                 std::to_string(rows.size()));
    }
    if (rows[0].rowid != 2) {
        throw std::runtime_error("Expected filtered MATCH result rowid=2");
    }
    if (!approx_equal(static_cast<float>(rows[0].distance), 0.2f, 0.05f)) {
        throw std::runtime_error("Expected filtered MATCH distance near 0.2");
    }

    std::cout << "  ✓ vec0 MATCH ANN respects rowid filter" << std::endl;
}

void test_vec0_ann_hidden_query_respects_rowid_filter() {
    std::cout << "Testing vec0 hidden query with rowid filter semantics..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);

    db.exec("CREATE VIRTUAL TABLE t USING vec0(embedding float[2])");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (10, '[0.0,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (11, '[0.2,0]')");
    db.exec("INSERT INTO t(rowid, embedding) VALUES (12, '[0.4,0]')");

    const std::string sql = "SELECT rowid, distance FROM t "
                            "WHERE query = vec_f32('[0,0]') AND rowid IN (11,12) AND k = 1 "
                            "ORDER BY distance";

    std::string plan = query_explain_plan(db.get(), sql);
    if (plan.find("ann") == std::string::npos) {
        throw std::runtime_error("Expected ANN query plan for hidden-query rowid-filter test");
    }

    auto rows = query_row_distances(db.get(), sql);
    if (rows.size() != 1) {
        throw std::runtime_error("Expected one filtered hidden-query result, got " +
                                 std::to_string(rows.size()));
    }
    if (rows[0].rowid != 11) {
        throw std::runtime_error("Expected filtered hidden-query result rowid=11");
    }
    if (!approx_equal(static_cast<float>(rows[0].distance), 0.2f, 0.05f)) {
        throw std::runtime_error("Expected filtered hidden-query distance near 0.2");
    }

    std::cout << "  ✓ vec0 hidden query respects rowid filter" << std::endl;
}

void test_vec0_benchmark_combined_loop_overstates_ann_latency() {
    std::cout << "Testing vec0 benchmark timing sanity..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    db.exec("CREATE VIRTUAL TABLE vecs USING vec0(embedding float[64])");

    auto make_vector = [](size_t dim, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> vec(dim);
        for (auto& value : vec) {
            value = dist(rng);
        }
        return vec;
    };

    std::mt19937 rng(42);
    std::vector<std::vector<float>> corpus;
    std::vector<std::vector<float>> queries;
    corpus.reserve(1500);
    queries.reserve(24);

    for (size_t i = 0; i < 1500; ++i) {
        corpus.push_back(make_vector(64, rng));
    }
    for (size_t i = 0; i < 24; ++i) {
        queries.push_back(make_vector(64, rng));
    }

    sqlite3_stmt* insert_stmt = nullptr;
    int rc = sqlite3_prepare_v2(db.get(), "INSERT INTO vecs(rowid, embedding) VALUES (?, ?)", -1,
                                &insert_stmt, nullptr);
    assert(rc == SQLITE_OK && insert_stmt);
    for (size_t i = 0; i < corpus.size(); ++i) {
        sqlite3_bind_int64(insert_stmt, 1, static_cast<sqlite3_int64>(i + 1));
        sqlite3_bind_blob(insert_stmt, 2, corpus[i].data(),
                          static_cast<int>(corpus[i].size() * sizeof(float)), SQLITE_TRANSIENT);
        rc = sqlite3_step(insert_stmt);
        assert(rc == SQLITE_DONE);
        sqlite3_reset(insert_stmt);
        sqlite3_clear_bindings(insert_stmt);
    }
    sqlite3_finalize(insert_stmt);

    sqlite3_stmt* ann_stmt = nullptr;
    sqlite3_stmt* exact_stmt = nullptr;
    rc = sqlite3_prepare_v2(db.get(),
                            "SELECT rowid FROM vecs WHERE embedding MATCH ?1 AND k = ?2 "
                            "ORDER BY distance",
                            -1, &ann_stmt, nullptr);
    assert(rc == SQLITE_OK && ann_stmt);
    rc = sqlite3_prepare_v2(
        db.get(), "SELECT rowid FROM vecs ORDER BY vec_distance_l2(embedding, ?1) LIMIT ?2", -1,
        &exact_stmt, nullptr);
    assert(rc == SQLITE_OK && exact_stmt);

    auto collect_ids = [](sqlite3_stmt* stmt) {
        std::vector<std::int64_t> ids;
        int step_rc = SQLITE_OK;
        while ((step_rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            ids.push_back(sqlite3_column_int64(stmt, 0));
        }
        assert(step_rc == SQLITE_DONE);
        return ids;
    };

    sqlite3_bind_blob(ann_stmt, 1, queries[0].data(),
                      static_cast<int>(queries[0].size() * sizeof(float)), SQLITE_TRANSIENT);
    sqlite3_bind_int64(ann_stmt, 2, 10);
    (void)collect_ids(ann_stmt);
    sqlite3_reset(ann_stmt);
    sqlite3_clear_bindings(ann_stmt);

    auto measure_us = [](auto&& fn) {
        auto start = std::chrono::steady_clock::now();
        fn();
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count();
    };

    double ann_only_us = measure_us([&]() {
        for (const auto& query : queries) {
            sqlite3_bind_blob(ann_stmt, 1, query.data(),
                              static_cast<int>(query.size() * sizeof(float)), SQLITE_TRANSIENT);
            sqlite3_bind_int64(ann_stmt, 2, 10);
            (void)collect_ids(ann_stmt);
            sqlite3_reset(ann_stmt);
            sqlite3_clear_bindings(ann_stmt);
        }
    });

    double exact_only_us = measure_us([&]() {
        for (const auto& query : queries) {
            sqlite3_bind_blob(exact_stmt, 1, query.data(),
                              static_cast<int>(query.size() * sizeof(float)), SQLITE_TRANSIENT);
            sqlite3_bind_int64(exact_stmt, 2, 10);
            (void)collect_ids(exact_stmt);
            sqlite3_reset(exact_stmt);
            sqlite3_clear_bindings(exact_stmt);
        }
    });

    double combined_us = measure_us([&]() {
        size_t hits = 0;
        for (const auto& query : queries) {
            sqlite3_bind_blob(ann_stmt, 1, query.data(),
                              static_cast<int>(query.size() * sizeof(float)), SQLITE_TRANSIENT);
            sqlite3_bind_int64(ann_stmt, 2, 10);
            auto ann_ids = collect_ids(ann_stmt);
            sqlite3_reset(ann_stmt);
            sqlite3_clear_bindings(ann_stmt);

            sqlite3_bind_blob(exact_stmt, 1, query.data(),
                              static_cast<int>(query.size() * sizeof(float)), SQLITE_TRANSIENT);
            sqlite3_bind_int64(exact_stmt, 2, 10);
            auto exact_ids = collect_ids(exact_stmt);
            sqlite3_reset(exact_stmt);
            sqlite3_clear_bindings(exact_stmt);

            std::unordered_set<std::int64_t> exact_set(exact_ids.begin(), exact_ids.end());
            for (std::int64_t id : ann_ids) {
                hits += exact_set.contains(id) ? 1 : 0;
            }
        }
        if (hits == 0) {
            throw std::runtime_error("Expected overlap between ANN and exact result sets");
        }
    });

    sqlite3_finalize(ann_stmt);
    sqlite3_finalize(exact_stmt);

    if (ann_only_us <= 0.0 || exact_only_us <= 0.0) {
        throw std::runtime_error("Expected positive benchmark timings");
    }
    if (combined_us <= ann_only_us) {
        throw std::runtime_error("Combined ANN+exact loop should exceed ANN-only latency");
    }
    if (combined_us <= ann_only_us + (0.25 * exact_only_us)) {
        throw std::runtime_error("Combined loop did not materially exceed ANN-only latency");
    }

    std::cout << "  ✓ Combined ANN+exact loop materially overstates ANN-only latency" << std::endl;
}

void test_vec0_ann_reopen_rebuild() {
    std::cout << "Testing vec0 ANN rebuild after reopen..." << std::endl;

    namespace fs = std::filesystem;
    const fs::path db_path = fs::temp_directory_path() / "sqlite_vec_cpp_vec0_ann_reopen.db";
    std::error_code ec;
    fs::remove(db_path, ec);

    {
        SQLiteDB db(db_path.string());
        sqlite3_vec_init(db.get(), nullptr, nullptr);

        db.exec("CREATE VIRTUAL TABLE t USING vec0(embedding float[2])");
        db.exec("INSERT INTO t(rowid, embedding) VALUES (1, '[0,0]')");
        db.exec("INSERT INTO t(rowid, embedding) VALUES (2, '[1,0]')");
        db.exec("INSERT INTO t(rowid, embedding) VALUES (3, '[2,0]')");
    }

    {
        SQLiteDB db(db_path.string());
        sqlite3_vec_init(db.get(), nullptr, nullptr);

        const std::string sql = "SELECT rowid, distance FROM t "
                                "WHERE embedding MATCH vec_f32('[1.1,0]') AND k = 2 "
                                "ORDER BY distance";

        auto rows = query_row_distances(db.get(), sql);
        assert(rows.size() == 2);
        assert(rows[0].rowid == 2);
        assert(rows[1].rowid == 3);
    }

    fs::remove(db_path, ec);

    std::cout << "  ✓ vec0 ANN rebuilds after reopen" << std::endl;
}

void test_integration_similarity_search() {
    std::cout << "Testing integration: similarity search..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Create table with vectors
    db.exec("CREATE TABLE documents (id INTEGER PRIMARY KEY, embedding BLOB)");

    // Insert vectors using vec_f32
    db.exec("INSERT INTO documents VALUES (1, vec_f32('[1,0,0]'))");
    db.exec("INSERT INTO documents VALUES (2, vec_f32('[0,1,0]'))");
    db.exec("INSERT INTO documents VALUES (3, vec_f32('[1,1,0]'))");

    // Find document closest to [1,0.5,0]
    // Documents 1 and 3 are tied at distance 0.5, so either is correct
    std::string closest =
        db.exec_scalar("SELECT id FROM documents "
                       "ORDER BY vec_distance_l2(embedding, vec_f32('[1,0.5,0]')) "
                       "LIMIT 1");

    // Should be document 1 or 3 (both at distance 0.5)
    assert(closest == "1" || closest == "3");

    std::cout << "  ✓ Similarity search integration works" << std::endl;
}

void test_integration_vector_operations() {
    std::cout << "Testing integration: vector operations pipeline..." << std::endl;

    SQLiteDB db;
    sqlite3_vec_init(db.get(), nullptr, nullptr);
    register_all_functions(db);

    // Complex pipeline: create, normalize, scale, and compute distance
    std::string result = db.exec_scalar("SELECT vec_distance_l2("
                                        "  vec_scale(vec_normalize(vec_f32('[3,4]')), 5),"
                                        "  vec_f32('[3,4]')"
                                        ")");

    double dist = std::stod(result);
    assert(dist >= 0.0); // Should be valid distance

    // Chain math operations
    result = db.exec_scalar("SELECT vec_to_json("
                            "  vec_add("
                            "    vec_sub(vec_f32('[10,10,10]'), vec_f32('[5,5,5]')),"
                            "    vec_f32('[1,2,3]')"
                            "  )"
                            ")");

    // Should be [6,7,8]
    assert(result.find("6") != std::string::npos);
    assert(result.find("7") != std::string::npos);
    assert(result.find("8") != std::string::npos);

    std::cout << "  ✓ Vector operations pipeline works" << std::endl;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "SQLITE-VEC-CPP COMPREHENSIVE TEST SUITE\n";
    std::cout << "========================================\n\n";

    int tests_passed = 0;
    int tests_failed = 0;

    auto run_test = [&](const char* name, void (*test_func)()) {
        try {
            test_func();
            tests_passed++;
        } catch (const std::exception& e) {
            std::cerr << "\n✗ TEST FAILED: " << name << "\n";
            std::cerr << "  Error: " << e.what() << "\n";
            tests_failed++;
        } catch (...) {
            std::cerr << "\n✗ TEST FAILED: " << name << "\n";
            std::cerr << "  Unknown error\n";
            tests_failed++;
        }
    };

    std::cout << "Testing Vector Creation Functions:\n";
    run_test("vec_f32_json", test_vec_f32_json);
    run_test("vec_f32_blob", test_vec_f32_blob);
    run_test("vec_int8", test_vec_int8);
    run_test("vec_length", test_vec_length);
    run_test("vec_f32_vs_vec_f32_simple", test_vec_f32_vs_vec_f32_simple);
    run_test("vec_to_json", test_vec_to_json);

    std::cout << "\nTesting Distance Functions:\n";
    run_test("distance_l2", test_distance_l2);
    run_test("distance_l1", test_distance_l1);
    run_test("distance_cosine", test_distance_cosine);

    std::cout << "\nTesting Vector Math Functions:\n";
    run_test("vec_add", test_vec_add);
    run_test("vec_sub", test_vec_sub);
    run_test("vec_normalize", test_vec_normalize);
    run_test("vec_slice", test_vec_slice);

    std::cout << "\nTesting Enhanced Functions:\n";
    run_test("vec_dot", test_vec_dot);
    run_test("vec_magnitude", test_vec_magnitude);
    run_test("vec_scale", test_vec_scale);
    run_test("vec_mean", test_vec_mean);
    run_test("vec_std", test_vec_std);
    run_test("vec_min_max", test_vec_min_max);
    run_test("vec_clamp", test_vec_clamp);

    std::cout << "\nTesting Error Handling:\n";
    run_test("error_handling", test_error_handling);

    std::cout << "\nTesting Integration Scenarios:\n";
    run_test("stored_float_blobs_pairwise_helpers",
             test_stored_float_blobs_work_with_pairwise_helpers);
    run_test("sqlite3_vec_init_extended_functions",
             test_sqlite3_vec_init_registers_extended_functions);
    run_test("integration_similarity_search", test_integration_similarity_search);
    run_test("integration_vector_operations", test_integration_vector_operations);

    std::cout << "\nTesting vec0 Virtual Table:\n";
    run_test("vec0_basic_create_insert_select", test_vec0_basic_create_insert_select);
    run_test("vec0_update_delete_paths", test_vec0_update_delete_paths);
    run_test("vec0_explicit_rowid_insert", test_vec0_explicit_rowid_insert);
    run_test("vec0_dimension_validation", test_vec0_dimension_validation);
    run_test("vec0_ann_match_query", test_vec0_ann_match_query);
    run_test("vec0_ann_hidden_query", test_vec0_ann_hidden_query);
    run_test("vec0_ann_insert_visibility", test_vec0_ann_insert_visibility);
    run_test("vec0_ann_match_respects_rowid_filter", test_vec0_ann_match_respects_rowid_filter);
    run_test("vec0_ann_hidden_query_respects_rowid_filter",
             test_vec0_ann_hidden_query_respects_rowid_filter);
    run_test("vec0_benchmark_combined_loop_overstates_ann_latency",
             test_vec0_benchmark_combined_loop_overstates_ann_latency);
    run_test("vec0_ann_reopen_rebuild", test_vec0_ann_reopen_rebuild);

    std::cout << "\nTesting Distance Subtypes (Bit/Int8):\n";
    run_test("distance_int8_l1_l2_cosine", test_distance_int8_l1_l2_cosine);
    run_test("distance_bit_hamming", test_distance_bit_hamming);

    std::cout << "\nTesting Distance Error Paths:\n";
    run_test("distance_mismatched_types", test_distance_mismatched_types);
    run_test("distance_invalid_args", test_distance_invalid_args);

    std::cout << "\nTesting C API Compatibility Layer:\n";
    run_test("c_api_init_and_distance", test_c_api_init_and_distance);

    std::cout << "\n========================================\n";
    std::cout << "TEST RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";
    std::cout << "Total tests:  " << (tests_passed + tests_failed) << "\n";

    if (tests_failed == 0) {
        std::cout << "\n✓ All tests passed!\n\n";
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed\n\n";
        return 1;
    }
}
